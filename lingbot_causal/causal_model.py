"""
Causal LingBot-World Model
Full DiT model with block-wise causal attention for streaming generation
"""

import sys
sys.path.insert(0, '/workspace/lingbot-world')

import torch
import torch.nn as nn
import math
from typing import Optional, Dict

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

# Import Wan modules
from wan.modules.model import (
    WanRMSNorm, WanLayerNorm, rope_apply, rope_params, sinusoidal_embedding_1d,
    Head, WanSelfAttention, WanAttentionBlock, WanT2VCrossAttention
)
WanCrossAttention = WanT2VCrossAttention  # Alias for compatibility
from wan.modules.animate.model_animate import MLPProj

from .causal_attention import CausalWanSelfAttention, BlockWiseCausalMask, KVCacheManager


class CausalWanAttentionBlock(nn.Module):
    """
    Attention block with support for block-wise causal attention and KV caching
    Includes camera control layers to match LingBot-World-Base-Cam weights
    """

    def __init__(self, dim, ffn_dim, num_heads,
                 window_size=(-1, -1), qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()

        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        # Self-attention (causal variant)
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, window_size, qk_norm, eps
        )

        # Cross-attention
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)

        # FFN
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        # Modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # Camera control layers
        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                block_mask=None, kv_cache=None, current_start=0, current_end=0,
                c2ws_plucker_emb=None):
        """Forward pass with optional causal masking and KV caching"""
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]

        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # Self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes, freqs,
            block_mask=block_mask,
            kv_cache=kv_cache,
            current_start=current_start,
            current_end=current_end,
            is_causal=kv_cache is not None
        )

        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # Camera injection
        if c2ws_plucker_emb is not None:
            c2ws_hidden = self.cam_injector_layer2(torch.nn.functional.silu(self.cam_injector_layer1(c2ws_plucker_emb)))
            c2ws_hidden = c2ws_hidden + c2ws_plucker_emb
            cam_scale = self.cam_scale_layer(c2ws_hidden)
            cam_shift = self.cam_shift_layer(c2ws_hidden)
            x = (1.0 + cam_scale) * x + cam_shift

        # Cross-attention
        x = x + self.cross_attn(self.norm3(x), context, context_lens)

        # FFN
        y = self.ffn((self.norm2(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2))
        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e[5]).flatten(1, 2)

        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    """Causal LingBot-World model with streaming support"""

    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size']
    _no_split_modules = ['CausalWanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='i2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=36,
                 dim=5120,
                 ffn_dim=13824,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=40,
                 num_layers=40,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()

        assert model_type in ['i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )

        # Attention blocks
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # Head
        self.head = Head(dim, out_dim, patch_size, eps)

        # RoPE frequencies
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        # Image embedding
        self.img_emb = MLPProj(1280, dim)
        self._use_img_emb = True

        # Block-wise causal mask
        self.block_mask = None
        self.num_frame_per_block = 1

        # KV cache manager
        self.kv_cache_manager = KVCacheManager(num_layers, num_heads, d)

        self.init_weights()
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)

    def _prepare_block_mask(self, num_frames, frame_seqlen, device):
        if self.block_mask is None:
            self.block_mask = BlockWiseCausalMask.create_mask(
                num_frames, frame_seqlen, self.num_frame_per_block, device
            )
        return self.block_mask

    def forward_train(self, x, t, context, seq_len, clip_fea, y):
        """Training forward pass"""
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        block_mask = None
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, sizes=t.shape)

        context_lens = None
        context = self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        for block in self.blocks:
            x = block(x, e0, seq_lens, grid_sizes, self.freqs, context, context_lens, block_mask=block_mask)

        x = self.head(x, e.unflatten(0, sizes=t.shape))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward_inference(self, x, t, context, seq_len, clip_fea, y,
                         kv_cache, crossattn_cache, current_start, current_end):
        """Streaming inference with KV caching"""
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat(x)

        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, sizes=t.shape)

        context_lens = None
        if True:
            context = self.text_embedding(torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
            if clip_fea is not None:
                context_clip = self.img_emb(clip_fea)
                context = torch.concat([context_clip, context], dim=1)
        else:
            context = None

        for block_idx, block in enumerate(self.blocks):
            x = block(x, e0, seq_lens, grid_sizes, self.freqs, context, context_lens,
                     kv_cache=kv_cache[block_idx], current_start=current_start, current_end=current_end)

        x = self.head(x, e.unflatten(0, sizes=t.shape))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        if kwargs.get('kv_cache') is not None:
            return self.forward_inference(*args, **kwargs)
        else:
            return self.forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def create_kv_cache(self, batch_size, device, dtype):
        return self.kv_cache_manager.create_cache(batch_size, device, dtype)
