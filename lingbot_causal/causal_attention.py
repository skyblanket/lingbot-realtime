"""
Block-wise Causal Attention for LingBot-World
Enables streaming generation by processing frames sequentially
"""

import sys
sys.path.insert(0, '/workspace/lingbot-world')

import torch
import torch.nn as nn
import math

# Try to import flex_attention (PyTorch 2.0+)
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None

# Try to import flash_attention, fall back to native PyTorch if not available
try:
    from wan.modules.attention import flash_attention, FLASH_ATTN_2_AVAILABLE
except ImportError:
    FLASH_ATTN_2_AVAILABLE = False
    flash_attention = None

def native_attention(q, k, v, causal=False):
    """Fallback using PyTorch's native scaled_dot_product_attention"""
    # Ensure same dtype (qk_norm may convert to float32)
    dtype = v.dtype
    q = q.to(dtype)
    k = k.to(dtype)

    # q, k, v: [B, seq_len, num_heads, head_dim]
    # Need to transpose for sdpa: [B, num_heads, seq_len, head_dim]
    q = q.transpose(1, 2)  # [B, num_heads, seq, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        is_causal=causal,
        dropout_p=0.0
    )
    return out.transpose(1, 2)  # Back to [B, seq, num_heads, head_dim]


class CausalWanSelfAttention(nn.Module):
    """
    Modified self-attention with block-wise causal masking and KV caching
    """
    
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__()
        
        try:
            from wan.modules.model import WanRMSNorm
        except ImportError:
            # Fallback implementation
            class WanRMSNorm(nn.Module):
                def __init__(self, dim, eps=1e-6):
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(dim))
                    self.eps = eps
                def forward(self, x):
                    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        
        # Layers (same as original)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        
    def forward(self, x, seq_lens, grid_sizes, freqs, 
                block_mask=None, kv_cache=None, 
                current_start=0, current_end=0, is_causal=False):
        """
        Args:
            x: [B, L, C]
            seq_lens: [B]
            grid_sizes: [B, 3] - (F, H, W)
            freqs: RoPE freqs
            block_mask: BlockMask for training (block-wise causal)
            kv_cache: Dict with 'k', 'v' for inference streaming
            current_start/end: Frame indices for KV cache
            is_causal: If True, use causal mask (for inference without block mask)
        """
        from wan.modules.model import rope_apply
        
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        
        # QKV projection
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        
        # Apply RoPE
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        
        if kv_cache is not None:
            # Inference mode with KV caching
            kv_cache['k'][:, current_start:current_end] = k
            kv_cache['v'][:, current_start:current_end] = v
            
            # Use all cached keys/values up to current_end
            k_full = kv_cache['k'][:, :current_end]
            v_full = kv_cache['v'][:, :current_end]
            
            # Standard causal attention with full cache
            if FLASH_ATTN_2_AVAILABLE and flash_attention is not None:
                x = flash_attention(q, k_full, v_full, causal=True)
            else:
                x = native_attention(q, k_full, v_full, causal=True)
        else:
            # Training mode with block mask
            if block_mask is not None and FLEX_ATTENTION_AVAILABLE:
                # Use flex_attention with block-wise causal mask
                x = self._flex_attention(q, k, v, block_mask)
            else:
                # Fall back to flash attention or native PyTorch
                if FLASH_ATTN_2_AVAILABLE and flash_attention is not None:
                    x = flash_attention(q, k, v, causal=is_causal)
                else:
                    x = native_attention(q, k, v, causal=is_causal)
        
        x = x.flatten(2)
        x = self.o(x)
        return x
    
    def _flex_attention(self, q, k, v, block_mask):
        """Use PyTorch's flex_attention for block-wise causal masking"""
        # Pad to multiple of 128 for performance
        pad_len = (128 - (q.shape[1] % 128)) % 128
        if pad_len > 0:
            q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))
        
        # Apply flex attention
        x = flex_attention(
            query=q.transpose(1, 2),
            key=k.transpose(1, 2),
            value=v.transpose(1, 2),
            block_mask=block_mask
        )
        
        # Remove padding
        if pad_len > 0:
            x = x[:, :, :-pad_len]
        
        return x.transpose(1, 2)


class BlockWiseCausalMask:
    """
    Creates block-wise causal attention mask for training
    Allows bidirectional attention within blocks, causal across blocks
    """
    
    @staticmethod
    def create_mask(num_frames, frame_seqlen, num_frame_per_block=1, device='cuda'):
        """
        Create block-wise causal mask
        
        Args:
            num_frames: Total number of frames
            frame_seqlen: Tokens per frame
            num_frame_per_block: Frames per causal block (1 = frame-wise causal)
            device: Device to create mask on
        """
        if not FLEX_ATTENTION_AVAILABLE:
            return None
        
        total_length = num_frames * frame_seqlen
        
        # Pad to multiple of 128
        padded_length = math.ceil(total_length / 128) * 128
        
        # Create mask tensor
        # For each position, mask determines which keys it can attend to
        ends = torch.zeros(padded_length, device=device, dtype=torch.long)
        
        # Block boundaries
        block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(0, total_length, block_size, device=device)
        
        for start in frame_indices:
            end = min(start + block_size, total_length)
            ends[start:end] = end
        
        # Mask function: can attend to positions < ends[idx]
        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
        
        block_mask = create_block_mask(
            attention_mask,
            B=None, H=None,
            Q_LEN=padded_length,
            KV_LEN=padded_length,
            _compile=False,
            device=device
        )
        
        print(f"[BlockMask] Created with block size: {num_frame_per_block} frames "
              f"({block_size} tokens), total: {num_frames} frames")
        
        return block_mask


class KVCacheManager:
    """
    Manages KV caches for streaming inference
    """
    
    def __init__(self, num_layers, num_heads, head_dim, max_seq_len=65536):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.caches = {}
        
    def create_cache(self, batch_size, device, dtype):
        """Create fresh KV caches for all layers"""
        self.caches = {
            i: {
                'k': torch.zeros(
                    batch_size, self.max_seq_len, self.num_heads, self.head_dim,
                    device=device, dtype=dtype
                ),
                'v': torch.zeros(
                    batch_size, self.max_seq_len, self.num_heads, self.head_dim,
                    device=device, dtype=dtype
                )
            }
            for i in range(self.num_layers)
        }
        return self.caches
    
    def clear(self):
        """Clear all caches"""
        self.caches = {}
    
    def get_cache(self, layer_idx):
        return self.caches.get(layer_idx)
