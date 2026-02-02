# Downloads Required for Real-Time LingBot-World

## ✅ Already Available on Your System

| Item | Location | Size | Status |
|------|----------|------|--------|
| LingBot-World Base (Cam) | `/home/sky/lingbot-world/lingbot-world-base-cam/` | ~60GB | ✅ Ready |
| - High noise model | `high_noise_model/` | 34GB | ✅ Ready |
| - Low noise model | `low_noise_model/` | 27GB | ✅ Ready |
| - VAE weights | `Wan2.1_VAE.pth` | 485MB | ✅ Ready |
| - Text encoder | `google/` folder | 21GB | ✅ Ready |
| LingBot-World code | `/home/sky/lingbot-world/` | 116GB total | ✅ Ready |

## 📥 Additional Downloads Needed

### 1. Training Dataset (Required for Distillation)

**Option A: MixKit Dataset (Toy/Small Scale)**
- **Size**: ~6K videos, ~50GB
- **Use**: Testing pipeline, quick experiments
- **Download**: `python distillation_data/download_mixkit.py`
- **Location**: CausVid repo already cloned

**Option B: Internal Ego4D/YouTube Videos (Recommended)**
- **Size**: 10K-100K videos, 500GB-5TB
- **Use**: Production-quality distillation
- **Download**: Need to curate from existing video datasets
- **Alternative**: Use LingBot-World's training data if available

### 2. Pre-computed ODE Pairs (Optional - Speeds Up Training)

These are teacher model outputs used for student warmup:
- **Size**: ~100GB for 10K pairs (480P)
- **Generate**: Use `generate_ode_pairs()` script
- **Alternative**: Skip and go straight to DMD training

### 3. Pre-trained Causal Checkpoints (Optional - Transfer Learning)

**CausVid Weights** (Same Wan2.1 architecture):
- Autoregressive checkpoint: ~5GB
- Bidirectional checkpoint: ~5GB
- **Download**: `huggingface-cli download tianweiy/CausVid`
- **Use**: Initialize student from CausVid instead of from scratch

## 🔧 Code Modules Created

| Module | Path | Purpose |
|--------|------|---------|
| Causal Attention | `lingbot_causal/causal_attention.py` | Block-wise causal attention + KV cache |
| Causal Model | `lingbot_causal/causal_model.py` | Full DiT with streaming support |
| Streaming Inference | `lingbot_causal/streaming_inference.py` | Real-time frame generation |
| DMD Trainer | `distillation/train_dmd.py` | Distillation training pipeline |

## 💾 Storage Summary

| Component | Size | Notes |
|-----------|------|-------|
| Existing LingBot-World | 60GB | Already downloaded |
| Training data (50K videos) | 2TB | Can use subset (100GB) |
| Precomputed ODE pairs | 100GB | Optional |
| Checkpoints during training | 120GB | 2x model size for EMA |
| **Total Recommended** | **~500GB** | For full training |

## ⚡ Your Hardware Assessment

**EXCELLENT for this project:**
- ✅ 2× RTX PRO 6000 (96GB each) - Can train with batch size 2-4
- ✅ 503GB RAM - Plenty for data loading
- ✅ Fast storage - For video dataset

**No cloud instance needed** - your local machine is perfect!

## 🎯 Next Steps

1. **Quick Test** (~1 hour):
   - Test causal attention module
   - Verify streaming works
   - Generate 10 test frames

2. **Small Scale Training** (~1 day):
   - Download MixKit dataset
   - Train on 1K videos
   - Validate 4-step generation quality

3. **Full Training** (~1 week):
   - Curate larger dataset
   - Run full DMD distillation
   - Fine-tune for WASD control

## 🔗 Download Commands

```bash
# 1. MixKit dataset (for testing)
cd /home/sky/lingbot-realtime/CausVid
python distillation_data/download_mixkit.py --local_dir ./data/mixkit

# 2. CausVid checkpoints (optional transfer learning)
huggingface-cli download tianweiy/CausVid \
    --local-dir ./checkpoints/causvid \
    --include "autoregressive_checkpoint/*"

# 3. Generate ODE pairs (from your existing LingBot-World)
# (Script will use existing model at /home/sky/lingbot-world/)
python distillation/generate_ode_pairs.py \
    --teacher_path /home/sky/lingbot-world/lingbot-world-base-cam \
    --output_path ./data/ode_pairs.pt \
    --num_pairs 10000
```

Ready to start when you are!
