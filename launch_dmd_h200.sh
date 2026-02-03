#!/bin/bash
# DMD Training Launcher for 8x H200 GPUs
# Optimized for 10-hour training window

set -e

echo "======================================================================"
echo "🚀 LingBot-World DMD Training on 8x H200 GPUs"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  - GPUs: 8x H200 (141GB each)"
echo "  - Training window: 10 hours"
echo "  - Batch size per GPU: 2 (total 16)"
echo "  - Gradient accumulation: 2 (effective batch 32)"
echo "  - Target: 10,000+ steps in 10 hours"
echo ""

# Create directories
mkdir -p ./dmd_checkpoints
mkdir -p ./dmd_logs
mkdir -p ./wandb_logs

# Environment variables for H200s
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Training hyperparameters optimized for H200s
NUM_STEPS=${NUM_STEPS:-15000}
BATCH_SIZE=${BATCH_SIZE:-2}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-2}
SAVE_EVERY=${SAVE_EVERY:-1000}
LOG_EVERY=${LOG_EVERY:-50}
NUM_WORKERS=${NUM_WORKERS:-8}

echo "Training configuration:"
echo "  - Steps: $NUM_STEPS"
echo "  - Batch size per GPU: $BATCH_SIZE"
echo "  - Total batch size: $((BATCH_SIZE * 8 * GRADIENT_ACCUMULATION))"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "  - Save every: $SAVE_EVERY steps"
echo "  - Workers: $NUM_WORKERS"
echo ""

# Calculate estimated training time
# ~2.5s per step on H200 = 10 hours for 14,400 steps
echo "Estimated training time: ~10 hours for $NUM_STEPS steps"
echo ""

# Check GPU availability
echo "Checking GPUs..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -8
echo ""

# Launch training with torchrun for multi-GPU
echo "🚀 Starting distributed DMD training..."
echo "Press Ctrl+C to stop (checkpoint will be saved)"
echo ""

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    lingbot_causal/dmd_trainer_distributed.py \
    --teacher_model_path /home/sky/lingbot-world/lingbot-world-base-cam \
    --training_video_dir ./training_data \
    --output_dir ./dmd_checkpoints \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --save_every $SAVE_EVERY \
    --log_every $LOG_EVERY \
    --num_workers $NUM_WORKERS \
    --num_frames 17 \
    --height 512 \
    --width 512 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --use_wandb \
    --wandb_project "lingbot-dmd" \
    --wandb_run_name "8xh200-${NUM_STEPS}steps-$(date +%Y%m%d)" \
    2>&1 | tee ./dmd_logs/training_h200_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "======================================================================"
echo "✅ Training complete!"
echo "======================================================================"
echo "Checkpoints: ./dmd_checkpoints/"
echo "Logs: ./dmd_logs/"
echo ""
echo "Next steps:"
echo "  1. Test distilled model: python3 test_distilled.py"
echo "  2. Launch WASD demo: python3 wasd_demo.py"
echo "======================================================================"
