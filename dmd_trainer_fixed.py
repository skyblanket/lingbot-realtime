#!/usr/bin/env python3
"""Fixed DMD trainer using diffusers"""
import sys
sys.path.insert(0, '/workspace/lingbot-world')
sys.path.insert(0, '/workspace/lingbot-realtime')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

from lingbot_causal.causal_model import CausalWanModel
from lingbot_causal.weight_loader import LingBotWeightLoader
from diffusers import AutoencoderKLWan
from transformers import UMT5EncoderModel

def setup():
    dist.init_process_group('nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

class DMDTrainer:
    def __init__(self, student, teacher, vae, text_enc, args, rank):
        self.student = DDP(student, device_ids=[rank])
        self.teacher = teacher
        self.vae = vae
        self.text_enc = text_enc
        self.rank = rank
        self.args = args
        self.opt = torch.optim.AdamW(self.student.parameters(), lr=args.lr)
        
    def train_step(self, real_noise, t, ctx, v0):
        """Single DMD training step"""
        # Student forward (4 steps)
        with torch.no_grad():
            teacher_real = self.teacher.forward_train([real_noise], t, ctx, 
                                                       self.args.num_frames, None, [v0])
        
        student_pred = self.student.module.forward_train([real_noise], t, ctx,
                                                          self.args.num_frames, None, [v0])
        
        # Distribution matching loss
        loss = F.mse_loss(student_pred, teacher_real)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.item()

def main():
    local_rank = setup()
    
    args = type('Args', (), {
        'lr': 2e-5,
        'num_frames': 17,
        'batch_size': 2,
        'steps': 20000
    })()
    
    if local_rank == 0:
        print(f"🚀 DMD Training on {torch.cuda.device_count()} GPUs")
    
    student = CausalWanModel(dim=5120, num_heads=40, num_layers=40,
                            ffn_dim=13824, text_len=512).cuda(local_rank).to(torch.bfloat16)
    teacher = CausalWanModel(dim=5120, num_heads=40, num_layers=40,
                            ffn_dim=13824, text_len=512).cuda(local_rank).to(torch.bfloat16)
    
    loader = LingBotWeightLoader('/workspace/lingbot-world')
    loader.load_into_causal_model(student, device=f'cuda:{local_rank}')
    loader.load_into_causal_model(teacher, device=f'cuda:{local_rank}')
    
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    if local_rank == 0:
        print("Loading VAE and text encoder...")
        vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-T2V-14B", 
                                                subfolder="vae",
                                                torch_dtype=torch.bfloat16).cuda(local_rank)
        text_enc = UMT5EncoderModel.from_pretrained("google/umt5-xxl",
                                                     torch_dtype=torch.bfloat16).cuda(local_rank)
        print("Models loaded!")
    else:
        vae = None
        text_enc = None
    
    dist.barrier()
    
    trainer = DMDTrainer(student, teacher, vae, text_enc, args, local_rank)
    
    for step in range(args.steps):
        real_noise = torch.randn(args.batch_size, 16, args.num_frames, 32, 32,
                                 device=f'cuda:{local_rank}', dtype=torch.bfloat16)
        t = torch.randint(0, 1000, (args.batch_size,), device=f'cuda:{local_rank}')
        ctx = [torch.randn(args.batch_size, 512, 5120, device=f'cuda:{local_rank}')]
        v0 = real_noise[:, :, :1]
        
        loss = trainer.train_step(real_noise, t, ctx, v0)
        
        if step % 50 == 0 and local_rank == 0:
            print(f"Step {step}/{args.steps}: loss={loss:.4f}")
        
        if step % 1000 == 0 and step > 0 and local_rank == 0:
            torch.save(student.module.state_dict(), f"dmd_checkpoint_{step}.pt")
            print(f"💾 Saved checkpoint at step {step}")
    
    if local_rank == 0:
        torch.save(student.module.state_dict(), "dmd_student_final.pt")
        print("✅ Training complete!")
    
    cleanup()

if __name__ == "__main__":
    main()
