"""
Distribution Matching Distillation (DMD) Trainer for LingBot-World
Converts 50-step teacher to 4-step causal student
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import cv2
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from wan package (pip install)
from wan.modules.t5 import T5EncoderModel
from wan.modules import WanVAE
try:
    from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                       retrieve_timesteps)
except ImportError:
    from diffusers import FlowMatchEulerDiscreteScheduler as FlowDPMSolverMultistepScheduler
    retrieve_timesteps = None
from .causal_model import CausalWanModel
from .weight_loader import LingBotWeightLoader


class VideoTextDataset(Dataset):
    """Simple video dataset for DMD training"""
    
    def __init__(self, video_dir, num_frames=17, height=512, width=512):
        self.video_dir = Path(video_dir)
        self.video_files = list(self.video_dir.glob("*.mp4"))
        self.num_frames = num_frames
        self.height = height
        self.width = width
        
        # Default prompts
        self.default_prompts = [
            "hands typing on keyboard",
            "person using computer",
            "working at desk",
            "coding on laptop",
            "hands on keyboard",
            "typing on computer",
            "office work",
            "programming session",
            "using computer mouse",
            "desktop computer usage"
        ]
        
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        
        # Load video
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = list(range(total_frames)) + [total_frames - 1] * (self.num_frames - total_frames)
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, (self.width, self.height))
                frames.append(frame)
            else:
                # Use last frame if can't read
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))
        
        cap.release()
        
        # Ensure we have num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.height, self.width, 3), dtype=np.uint8))
        
        # Convert to tensor [C, F, H, W]
        frames = np.stack(frames[:self.num_frames])  # [F, H, W, C]
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        
        # Normalize to [-1, 1]
        frames = frames * 2.0 - 1.0
        
        # Use file name for prompt
        prompt_idx = idx % len(self.default_prompts)
        text = self.default_prompts[prompt_idx]
        
        return {
            'video': frames,
            'text': text,
            'video_path': str(video_path)
        }


class DMDTrainer:
    """
    Distribution Matching Distillation trainer
    Based on CausVid paper methodology
    """
    
    def __init__(
        self,
        student_model,
        teacher_model,
        vae,
        text_encoder,
        device='cuda',
        lr=1e-5,
        num_train_timesteps=1000,
        mixed_precision='bf16',
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.vae = vae.to(device)
        self.text_encoder = text_encoder
        self.device = device
        self.mixed_precision = mixed_precision
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Optimizer with 8-bit Adam for memory efficiency
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                self.student.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
            print("[DMD] Using 8-bit Adam optimizer")
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
            print("[DMD] Using standard AdamW optimizer (install bitsandbytes for 8-bit)")
        
        # DMD settings
        self.num_train_timesteps = num_train_timesteps
        self.student_steps = 4
        self.teacher_steps = 50
        
        # Loss weights
        self.lambda_distill = 1.0
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision == 'fp16' else None
        
        # Training stats
        self.global_step = 0
        
    def get_timesteps(self, num_steps):
        """Get timesteps for few-step sampling"""
        max_t = self.num_train_timesteps - 1
        timesteps = torch.linspace(max_t, 0, num_steps + 1, dtype=torch.long)
        return timesteps[:-1]  # Exclude 0
    
    def ode_step(self, model, x, t, t_prev, context, clip_fea, y, is_teacher=True):
        """Single ODE step using Euler method"""
        with torch.no_grad() if is_teacher else torch.enable_grad():
            # Handle model input format
            if hasattr(model, 'forward_train'):
                # Student - causal model
                t_batch = t.expand(x.size(0)) if x.size(0) > 1 else torch.tensor([t], device=self.device)
                pred = model.forward_train([x], t_batch, context, x.shape[1], clip_fea, y)
            else:
                # Teacher - original LingBot-World
                pred = model([x], torch.tensor([t], device=self.device),
                           context, x.shape[1], clip_fea, y)
        
        # Euler step
        dt = t_prev - t
        x_next = x + dt * pred
        return x_next
    
    @torch.no_grad()
    def compute_teacher_output(self, x0, context, clip_fea, y):
        """Compute teacher ODE output (target for distillation)"""
        noise = torch.randn_like(x0)
        x = noise
        
        timesteps = self.get_timesteps(self.teacher_steps)
        
        for i in range(len(timesteps)):
            t = timesteps[i].item()
            t_prev = timesteps[i+1].item() if i+1 < len(timesteps) else 0
            
            x = self.ode_step(self.teacher, x, t, t_prev, context, clip_fea, y, is_teacher=True)
        
        return x
    
    def compute_student_output(self, x0, context, clip_fea, y):
        """Compute student ODE output with few steps"""
        noise = torch.randn_like(x0)
        x = noise
        
        timesteps = self.get_timesteps(self.student_steps)
        predictions = []
        
        for i in range(len(timesteps)):
            t = timesteps[i].item()
            t_prev = timesteps[i+1].item() if i+1 < len(timesteps) else 0
            
            # Forward through student
            t_batch = torch.tensor([t], device=self.device).expand(x.size(0))
            pred = self.student.forward_train([x], t_batch, context, x.shape[1], clip_fea, y)
            predictions.append(pred)
            
            # Euler step
            dt = t_prev - t
            x = x + dt * pred.detach() if i < len(timesteps) - 1 else x + dt * pred
        
        return x, predictions
    
    def dmd_loss(self, student_out, teacher_out):
        """Distribution matching loss"""
        distill_loss = F.mse_loss(student_out, teacher_out)
        return distill_loss
    
    def train_step(self, batch):
        """Single training step"""
        self.student.train()
        
        # Move to device
        videos = batch['video'].to(self.device)  # [B, C, F, H, W]
        texts = batch['text']
        
        # Encode videos to latents with VAE
        with torch.no_grad():
            # VAE encode expects [B, C, F, H, W]
            latents = []
            for video in videos:
                latent = self.vae.encode(video.unsqueeze(0))[0]
                latents.append(latent)
            latents = torch.stack(latents)  # [B, C_latent, F, H_latent, W_latent]
            
            # Encode text
            context = self.text_encoder(texts)
        
        # Conditional frames (first frame)
        y = [lat[:, :, :1] for lat in latents]
        
        # Compute teacher output (no grad)
        with torch.no_grad():
            teacher_out = self.compute_teacher_output(latents, context, None, y)
        
        # Compute student output
        if self.mixed_precision == 'bf16':
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                student_out, _ = self.compute_student_output(latents, context, None, y)
                loss = self.dmd_loss(student_out, teacher_out)
        else:
            student_out, _ = self.compute_student_output(latents, context, None, y)
            loss = self.dmd_loss(student_out, teacher_out)
        
        # Backprop
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'step': self.global_step
        }
    
    def save_checkpoint(self, path, step):
        """Save checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"[DMD] Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description='DMD Training for LingBot-World')
    parser.add_argument('--teacher_model_path', type=str, required=True)
    parser.add_argument('--training_video_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./dmd_checkpoints')
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--num_frames', type=int, default=17)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--mixed_precision', type=str, default='bf16', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--use_8bit_adam', action='store_true')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LingBot-World DMD Distillation Training")
    print("="*70)
    print(f"Teacher: 50-step LingBot-World")
    print(f"Student: 4-step causal model")
    print(f"Training steps: {args.num_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Mixed precision: {args.mixed_precision}")
    print("="*70)
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path('./dmd_logs').mkdir(parents=True, exist_ok=True)
    
    device = 'cuda'
    
    # Load models
    print("[DMD] Loading models...")
    
    # Load causal student model
    print("[DMD] Loading causal student model...")
    
    # Create student model (LingBot-World dimensions)
    student = CausalWanModel(
        dim=5120,
        num_heads=40,
        num_layers=40,
        ffn_dim=13824,
        freq_dim=256,
        text_len=512,
        patch_size=(1, 2, 2),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        model_type='i2v'
    ).to(device).to(torch.bfloat16)
    
    # Load weights from LingBot-World
    print("[DMD] Loading weights from LingBot-World...")
    weight_loader = LingBotWeightLoader(args.teacher_model_path)
    weight_loader.load_into_causal_model(student, device=device)
    
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        print("[DMD] Gradient checkpointing enabled")
    
    # Create teacher (same architecture, frozen)
    print("[DMD] Loading teacher model...")
    teacher = CausalWanModel(
        dim=5120,
        num_heads=40,
        num_layers=40,
        ffn_dim=13824,
        freq_dim=256,
        text_len=512,
        patch_size=(1, 2, 2),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        model_type='i2v'
    ).to(device).to(torch.bfloat16)
    
    # Load same weights for teacher
    weight_loader.load_into_causal_model(teacher, device=device)
    
    print("[DMD] Models loaded successfully")
    
    # Load VAE
    print("[DMD] Loading VAE...")
    vae = WanVAE(vae_pth='./ckpts/Wan2.1_VAE.pth', device=device)
    
    # Load text encoder
    print("[DMD] Loading text encoder...")
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path="./ckpts/models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path="./ckpts/google-umt5-xxl",
        shard_size=8  # For 8-bit quantization
    )
    
    # Create dataset
    print(f"[DMD] Loading training videos from {args.training_video_dir}...")
    dataset = VideoTextDataset(
        args.training_video_dir,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width
    )
    print(f"[DMD] Dataset size: {len(dataset)} videos")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Create trainer
    trainer = DMDTrainer(
        student_model=student,
        teacher_model=teacher,
        vae=vae,
        text_encoder=text_encoder,
        device=device,
        lr=args.learning_rate,
        mixed_precision=args.mixed_precision
    )
    
    # Training loop
    print("[DMD] Starting training...")
    
    losses = []
    
    for step in range(args.num_steps):
        # Get batch
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            batch = next(iter(dataloader))
        
        # Training step
        stats = trainer.train_step(batch)
        losses.append(stats['loss'])
        
        # Logging
        if (step + 1) % args.log_every == 0:
            avg_loss = np.mean(losses[-args.log_every:])
            print(f"[Step {step+1}/{args.num_steps}] Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_step_{step+1}.pt"
            trainer.save_checkpoint(checkpoint_path, step + 1)
        
        # Clear cache periodically
        if (step + 1) % 100 == 0:
            torch.cuda.empty_cache()
    
    # Save final checkpoint
    final_path = Path(args.output_dir) / "final_checkpoint.pt"
    trainer.save_checkpoint(final_path, args.num_steps)
    
    print("="*70)
    print("[DMD] Training complete!")
    print(f"[DMD] Final checkpoint: {final_path}")
    print("="*70)


if __name__ == "__main__":
    main()
