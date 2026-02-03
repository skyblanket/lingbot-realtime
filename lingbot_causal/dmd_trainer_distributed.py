#!/usr/bin/env python3
"""
Distributed DMD Training for LingBot-World
Optimized for 8x H200/H100 GPUs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import cv2
import sys
import os
import wandb
import time

# Add paths
sys.path.insert(0, '/workspace/lingbot-world')
sys.path.insert(0, str(Path(__file__).parent.parent)

# Use diffusers for VAE and text encoder
try:
    from diffusers import AutoencoderKLWan, WanPipeline
    from transformers import UMT5EncoderModel
    USE_DIFFUSERS = True
    print("Using diffusers for VAE and text encoder")
except ImportError:
    USE_DIFFUSERS = False
    print("Diffusers not available - using simplified training")

from lingbot_causal.causal_model import CausalWanModel
from lingbot_causal.weight_loader import LingBotWeightLoader


def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


class VideoTextDataset(Dataset):
    """Video dataset for DMD training"""
    
    def __init__(self, video_dir, num_frames=17, height=512, width=512):
        self.video_dir = Path(video_dir)
        self.video_files = list(self.video_dir.glob("*.mp4"))
        self.num_frames = num_frames
        self.height = height
        self.width = width
        
        self.default_prompts = [
            "hands typing on keyboard", "person using computer", "working at desk",
            "coding on laptop", "hands on keyboard", "typing on computer",
            "office work", "programming session", "using computer mouse",
            "desktop computer usage", "first person view walking", "indoor navigation",
            "moving through room", "walking forward", "turning left",
            "turning right", "looking around", "exploring environment"
        ]
        
    def __len__(self):
        return len(self.video_files) * 10  # Augment with different prompts
    
    def __getitem__(self, idx):
        video_idx = idx // 10
        prompt_idx = idx % 10
        
        video_path = self.video_files[video_idx]
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = list(range(total_frames)) + [total_frames - 1] * (self.num_frames - total_frames)
        
        frames = []
        for i in indices[:self.num_frames]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.width, self.height))
                frames.append(frame)
            else:
                frames.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))
        
        cap.release()
        
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.height, self.width, 3), dtype=np.uint8))
        
        frames = np.stack(frames[:self.num_frames])
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        frames = frames * 2.0 - 1.0
        
        text = self.default_prompts[prompt_idx % len(self.default_prompts)]
        
        return {'video': frames, 'text': text}


class DistributedDMDTrainer:
    """Distributed DMD Trainer"""
    
    def __init__(self, student, teacher, vae, text_encoder, args, local_rank):
        self.student = student.to(local_rank)
        self.teacher = teacher.to(local_rank)
        self.vae = vae.to(local_rank)
        self.text_encoder = text_encoder
        self.args = args
        self.local_rank = local_rank
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # DDP wrapper
        self.student = DDP(self.student, device_ids=[local_rank])
        
        # Optimizer
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                self.student.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
        
        self.num_train_timesteps = 1000
        self.student_steps = 4
        self.teacher_steps = 50
        self.global_step = 0
        
        # Wandb logging (only rank 0)
        self.use_wandb = args.use_wandb and local_rank == 0
        if self.use_wandb:
            wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    
    def get_timesteps(self, num_steps):
        max_t = self.num_train_timesteps - 1
        timesteps = torch.linspace(max_t, 0, num_steps + 1, dtype=torch.long)
        return timesteps[:-1]
    
    @torch.no_grad()
    def compute_teacher_output(self, latents, context):
        noise = torch.randn_like(latents)
        x = noise
        timesteps = self.get_timesteps(self.teacher_steps)
        
        for i in range(len(timesteps)):
            t = timesteps[i].item()
            t_prev = timesteps[i+1].item() if i+1 < len(timesteps) else 0
            t_batch = torch.tensor([t], device=x.device).expand(x.size(0))
            pred = self.teacher([x], t_batch, context, x.shape[1], None, [x[:, :, :1]])
            dt = t_prev - t
            x = x + dt * pred
        
        return x
    
    def compute_student_output(self, latents, context):
        noise = torch.randn_like(latents)
        x = noise
        timesteps = self.get_timesteps(self.student_steps)
        
        for i in range(len(timesteps)):
            t = timesteps[i].item()
            t_prev = timesteps[i+1].item() if i+1 < len(timesteps) else 0
            t_batch = torch.tensor([t], device=x.device).expand(x.size(0))
            pred = self.student([x], t_batch, context, x.shape[1], None, [x[:, :, :1]])
            dt = t_prev - t
            x = x + dt * pred.detach() if i < len(timesteps) - 1 else x + dt * pred
        
        return x
    
    def train_step(self, batch):
        self.student.train()
        
        videos = batch['video'].to(self.local_rank)
        texts = batch['text']
        
        # Encode to latents
        with torch.no_grad():
            latents = []
            for video in videos:
                latent = self.vae.encode(video.unsqueeze(0))[0]
                latents.append(latent)
            latents = torch.stack(latents)
            context = self.text_encoder(texts, device=self.local_rank)
        
        # Teacher output
        with torch.no_grad():
            teacher_out = self.compute_teacher_output(latents, context)
        
        # Student output
        student_out = self.compute_student_output(latents, context)
        
        # Loss
        loss = F.mse_loss(student_out, teacher_out)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.global_step += 1
        
        return {'loss': loss.item()}
    
    def save_checkpoint(self, path, step):
        if self.local_rank == 0:
            checkpoint = {
                'step': step,
                'model_state_dict': self.student.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, path)
            print(f"[DMD] Saved checkpoint to {path}")
            
            if self.use_wandb:
                wandb.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_model_path', type=str, required=True)
    parser.add_argument('--training_video_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./dmd_checkpoints')
    parser.add_argument('--num_steps', type=int, default=15000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_frames', type=int, default=17)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--mixed_precision', type=str, default='bf16')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--use_8bit_adam', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='lingbot-dmd')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup distributed
    local_rank = setup_distributed()
    
    if local_rank == 0:
        print("="*70)
        print("🚀 LingBot-World Distributed DMD Training")
        print("="*70)
        print(f"GPUs: {torch.cuda.device_count()}")
        print(f"Steps: {args.num_steps}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * torch.cuda.device_count() * args.gradient_accumulation_steps}")
        print("="*70)
    
    # Create output dir
    if local_rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dist.barrier()
    
    # Load models
    if local_rank == 0:
        print("[DMD] Loading models...")
    
    student = CausalWanModel(
        dim=5120, num_heads=40, num_layers=40,
        ffn_dim=13824, freq_dim=256, text_len=512,
        patch_size=(1, 2, 2), model_type='i2v'
    ).to(local_rank).to(torch.bfloat16)
    
    loader = LingBotWeightLoader(args.teacher_model_path)
    loader.load_into_causal_model(student, device=f'cuda:{local_rank}')
    
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
    
    teacher = CausalWanModel(
        dim=5120, num_heads=40, num_layers=40,
        ffn_dim=13824, freq_dim=256, text_len=512,
        patch_size=(1, 2, 2), model_type='i2v'
    ).to(local_rank).to(torch.bfloat16)
    loader.load_into_causal_model(teacher, device=f'cuda:{local_rank}')
    teacher.eval()
    
    # Load VAE and text encoder (only rank 0, then broadcast)
    if local_rank == 0:
        print("[DMD] Loading VAE and text encoder...")
    
    vae = WanVAE(
        vae_pth=f'{args.teacher_model_path}/Wan2.1_VAE.pth',
        device=f'cuda:{local_rank}'
    )
    
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=f'cuda:{local_rank}',
        checkpoint_path=f'{args.teacher_model_path}/models_t5_umt5-xxl-enc-bf16.pth',
        tokenizer_path="google/umt5-xxl",
    )
    
    # Create dataset
    dataset = VideoTextDataset(
        args.training_video_dir,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create trainer
    trainer = DistributedDMDTrainer(student, teacher, vae, text_encoder, args, local_rank)
    
    # Training loop
    if local_rank == 0:
        print("[DMD] Starting training...")
    
    step = 0
    epoch = 0
    
    while step < args.num_steps:
        sampler.set_epoch(epoch)
        
        for batch in dataloader:
            if step >= args.num_steps:
                break
            
            start_time = time.time()
            stats = trainer.train_step(batch)
            step_time = time.time() - start_time
            
            # Logging
            if step % args.log_every == 0 and local_rank == 0:
                print(f"[Step {step}/{args.num_steps}] Loss: {stats['loss']:.6f} | Time: {step_time:.2f}s")
                
                if trainer.use_wandb:
                    wandb.log({'loss': stats['loss'], 'step_time': step_time}, step=step)
            
            # Save checkpoint
            if step % args.save_every == 0 and step > 0 and local_rank == 0:
                trainer.save_checkpoint(f"{args.output_dir}/checkpoint_step_{step}.pt", step)
            
            step += 1
        
        epoch += 1
    
    # Final save
    if local_rank == 0:
        trainer.save_checkpoint(f"{args.output_dir}/final_checkpoint.pt", step)
        print("="*70)
        print("✅ Training complete!")
        print(f"Checkpoints: {args.output_dir}/")
        print("="*70)
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
