#!/bin/bash
# Launch DMD Distillation Training for LingBot-World
# Converts 50-step teacher model to 4-step student

set -e

echo "======================================================================"
echo "LingBot-World DMD Distillation Training"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  - Teacher: 50-step LingBot-World (frozen)"
echo "  - Student: 4-step causal model (trainable)"
echo "  - Training data: ./training_data/*.mp4"
echo "  - Target FPS: 16+ (after distillation)"
echo ""

# Create directories
mkdir -p ./dmd_checkpoints
mkdir -p ./dmd_logs
mkdir -p ./training_data_latents

# Set environment variables for Blackwell GPUs
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Training hyperparameters
NUM_STEPS=${NUM_STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
SAVE_EVERY=${SAVE_EVERY:-500}
LOG_EVERY=${LOG_EVERY:-10}

echo "Training hyperparameters:"
echo "  - Steps: $NUM_STEPS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "  - Save every: $SAVE_EVERY steps"
echo ""

# Launch training
echo "Starting DMD distillation training..."
echo "Press Ctrl+C to stop (checkpoint will be saved)"
echo ""

python3 -m lingbot_causal.dmd_trainer \
    --teacher_model_path /home/sky/lingbot-world/lingbot-world-base-cam \
    --training_video_dir ./training_data \
    --output_dir ./dmd_checkpoints \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --save_every $SAVE_EVERY \
    --log_every $LOG_EVERY \
    --num_frames 17 \
    --height 512 \
    --width 512 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --use_8bit_adam \
    2>&1 | tee ./dmd_logs/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training complete!"
echo "Checkpoints saved to: ./dmd_checkpoints/"
echo "Logs saved to: ./dmd_logs/"
