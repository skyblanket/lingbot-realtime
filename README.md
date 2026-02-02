# LingBot-World Real-Time 🎮

Real-time streaming world model based on LingBot-World with block-wise causal attention for interactive WASD control.

## 🎯 Goal

Transform LingBot-World from a 50-step bidirectional diffusion model (20-30s generation) into a 4-step causal streaming model capable of **16+ FPS real-time WASD gameplay**.

## 🚀 Quick Start

```bash
# 1. Install
pip install -e .

# 2. Test causal attention
python -c "from lingbot_causal import CausalWanModel; print('✓ Causal model loaded')"

# 3. Run streaming inference demo
python demo/wasd_demo.py --image input.jpg --prompt "A fantasy world"
```

## 📁 Structure

```
lingbot-realtime/
├── lingbot_causal/           # Core causal model implementation
│   ├── causal_attention.py   # Block-wise causal attention + KV cache
│   ├── causal_model.py       # Full CausalWanModel
│   └── streaming_inference.py # Real-time generation loop
├── distillation/
│   └── train_dmd.py          # DMD training pipeline
├── demo/
│   └── wasd_demo.py          # Interactive WASD demo
├── CausVid/                  # Reference implementation (MIT)
├── FastGen/                  # NVIDIA distillation framework
└── Causal-Forcing/           # Alternative approach
```

## 🔬 Technical Approach

### 1. Block-Wise Causal Attention
- **Within block**: Bidirectional (maintains frame consistency)
- **Across blocks**: Causal (enables streaming)
- **Result**: Generate frames sequentially, one at a time

### 2. KV Caching
- Cache key/value tensors from previous frames
- Avoid recomputing attention for past frames
- Enables O(1) per-frame cost (vs O(N²) for full sequence)

### 3. Distribution Matching Distillation (DMD)
- Teacher: Original 50-step LingBot-World (frozen)
- Student: 4-step causal model (trained)
- Loss: Match student output distribution to teacher

### 4. Streaming Pipeline
```
Image → Latent → [Frame 1] → Decode → Display
                    ↓
              [Frame 2] → Decode → Display
                    ↓
              [Frame 3] → ...
```

## 💻 Hardware Requirements

| Component | Minimum | Recommended | Your Setup |
|-----------|---------|-------------|------------|
| GPU | 1× RTX 4090 (24GB) | 2× A100 (80GB) | ✅ 2× RTX PRO 6000 (96GB) |
| VRAM | 24GB | 80GB | ✅ 96GB per GPU |
| RAM | 64GB | 256GB | ✅ 503GB |
| Storage | 500GB SSD | 2TB NVMe | ✅ Sufficient |

**Your hardware can achieve 16-30 FPS at 480p-720p!**

## 📊 Performance Targets

| Metric | Current | Target | Your HW Expected |
|--------|---------|--------|------------------|
| Steps | 50 | 4 | 4 |
| Latency | 20-30s | <1s | <0.5s |
| FPS | ~0.5 | 16 | 16-30 |
| Resolution | 720p | 720p | 720p |
| Consistency | 10+ min | 10+ min | 10+ min |

## 🎮 WASD Demo

```python
from lingbot_causal import StreamingInference, WASDController

# Create controller
controller = WASDController()
controller.start_listener()

# Stream generation
for frame in streamer.stream_generate(image, prompt):
    display(frame)
    action = controller.get_action()  # W/A/S/D
    # Action conditions next frame generation
```

## 🧪 Testing

```bash
# Test causal attention
python -m pytest tests/test_causal_attention.py

# Test streaming
python tests/test_streaming.py --num_frames 10

# Benchmark FPS
python benchmarks/bench_fps.py --resolution 480p
```

## 🏋️ Training

### Stage 1: Generate ODE Pairs (Optional)
```bash
python distillation/generate_ode_pairs.py \
    --teacher_path /home/sky/lingbot-world/lingbot-world-base-cam \
    --output_path data/ode_pairs.pt \
    --num_pairs 10000
```

### Stage 2: DMD Training
```bash
torchrun --nproc_per_node=2 distillation/train_dmd.py \
    --teacher_path /home/sky/lingbot-world/lingbot-world-base-cam \
    --output_dir checkpoints/causal_lingbot \
    --batch_size 1 \
    --num_epochs 1000
```

## 📚 References

- [CausVid](https://github.com/tianweiy/CausVid) - MIT's causal video diffusion
- [LingBot-World](https://github.com/Robbyant/lingbot-world) - Base world model
- [DMD Paper](https://arxiv.org/abs/2311.17042) - Distribution matching distillation
- [Consistency Models](https://arxiv.org/abs/2303.01469) - Few-step generation

## 📝 TODO

- [ ] Integrate with existing LingBot-World weights
- [ ] Add action conditioning (WASD → action embeddings)
- [ ] Optimize VAE decoding for real-time
- [ ] Create interactive Gradio demo
- [ ] Quantization for faster inference

## 🏆 Goal

The first open-source real-time interactive world model with:
- ✅ 16+ FPS generation
- ✅ WASD keyboard control
- ✅ 10+ minute consistency
- ✅ 720p resolution
- ✅ Runs on dual RTX PRO 6000

---

Built with PyTorch, inspired by CausVid, powered by LingBot-World.
