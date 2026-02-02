# Real-Time LingBot-World Implementation Plan

## Overview
This document outlines how to convert LingBot-World from a 50-step bidirectional diffusion model to a 4-step causal streaming model capable of real-time WASD interaction.

## Current State
- **LingBot-World Base (Cam)**: 50-step bidirectional diffusion, ~20-30 sec generation
- **Target**: 3-4 step causal streaming, 16+ FPS real-time

## The Core Technologies (Already Researched)

### 1. Block-Wise Causal Attention
Located in: `/home/sky/lingbot-realtime/CausVid/causvid/models/wan/causal_model.py`

```python
def _prepare_blockwise_causal_attn_mask(
    device: torch.device | str, num_frames: int = 21,
    frame_seqlen: int = 1560, num_frame_per_block=1
) -> BlockMask:
    # Block-wise causal mask will attend to all elements that are before the end of the current chunk
    # Within chunk: bidirectional (maintains consistency)
    # Across chunks: causal (enables streaming)
```

Key insight: Process frames in blocks (e.g., 1 frame per block), bidirectional within block, causal across blocks.

### 2. Distribution Matching Distillation (DMD)
Located in: `/home/sky/lingbot-realtime/CausVid/causvid/dmd.py`

- Distills 50-step teacher → 3-4 step student
- Uses ODE trajectory initialization for stability
- Asymmetric distillation: bidirectional teacher → causal student

### 3. KV Caching for Streaming
From CausVid causal_model.py:
```python
def _forward_inference(self, x, t, context, ..., 
                       kv_cache: dict = None,
                       crossattn_cache: dict = None,
                       current_start: int = 0, 
                       current_end: int = 0):
    # Processes latent frames one by one with KV caching
    # Enables streaming generation at 9.4 FPS on single GPU
```

## Hardware Advantage
Your setup (2× RTX PRO 6000 96GB) can:
- Run full 720p at 16+ FPS (vs 9.4 FPS on single consumer GPU)
- Run 480p at 30+ FPS
- Train AND inference simultaneously

## Implementation Steps

### Phase 1: Adapt LingBot-World to Causal Architecture (Week 1)

1. **Copy LingBot-World DiT architecture** → Add block causal attention
2. **Modify attention mask** in `wan/modules/attention.py`:
   ```python
   # Current: Full bidirectional
   # New: Block-wise causal
   if self.training:
       # Training: use block mask
   else:
       # Inference: use KV cache
   ```

3. **Add KV cache support** to existing blocks

### Phase 2: Distillation Training (Week 2-3)

1. **Generate ODE pairs** from LingBot-World teacher (bidirectional)
2. **Pretrain causal student** on ODE trajectories
3. **Run DMD training** to distill to 3-4 steps

Training command from CausVid:
```bash
torchrun --nproc_per_node=2 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    causvid/train_distillation.py \
    --config_path configs/wan_causal_dmd.yaml
```

### Phase 3: Real-Time WASD Demo (Week 4)

1. **Create streaming inference loop**:
   - Generate frame 0-1 from initial image
   - Decode to pixels
   - Show to user
   - Capture WASD input
   - Generate next frame conditioned on action

2. **Action conditioning** (from LingBot-World Base-Act):
   - Inject action embeddings into DiT blocks
   - WASD → action vectors → adaptive normalization

## Code Structure to Create

```
lingbot-realtime/
├── lingbot_causal/          # Modified LingBot with causal attention
│   ├── models/
│   │   ├── causal_dit.py   # Block causal DiT
│   │   └── kv_cache.py     # KV cache manager
│   └── inference/
│       └── streaming.py    # Real-time generation loop
├── distillation/
│   ├── generate_ode_pairs.py
│   └── train_dmd.py
└── demo/
    └── wasd_controller.py  # Keyboard input → world generation
```

## Key Files to Modify

1. **LingBot-World**: `wan/modules/model.py` → Add causal attention option
2. **LingBot-World**: `generate.py` → Add streaming mode
3. **New**: `causal_inference.py` → Streaming with WASD

## References Already Cloned

- `/home/sky/lingbot-realtime/CausVid` - Full causal video diffusion implementation
- `/home/sky/lingbot-realtime/FastGen` - NVIDIA's distillation framework
- `/home/sky/lingbot-realtime/Causal-Forcing` - Alternative causal approach

## Performance Targets

| Metric | Current | Target | Your Hardware |
|--------|---------|--------|---------------|
| Steps | 50 | 3-4 | ✓ |
| FPS | ~0.5 | 16+ | ✓ |
| Latency | 20-30s | <1s | ✓ |
| Resolution | 720p | 720p/480p | ✓ |

## Next Immediate Actions

1. Compare LingBot-World's `wan/modules/model.py` with CausVid's `causal_model.py`
2. Identify attention mask locations
3. Create block causal variant
4. Test with pre-trained LingBot-World weights

Ready to start implementation.
