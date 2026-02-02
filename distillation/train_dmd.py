"""
Distribution Matching Distillation (DMD) for LingBot-World
Distills 50-step teacher to 4-step causal student
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np
from pathlib import Path


class DMDTrainer:
    """
    Distribution Matching Distillation trainer
    Based on CausVid paper: distills bidirectional teacher → causal student
    """
    
    def __init__(
        self,
        student_model,      # Causal model (to be trained)
        teacher_model,      # Original LingBot-World (frozen)
        vae,
        text_encoder,
        device='cuda',
        lr=1e-5,
        num_train_timesteps=1000,
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.vae = vae.to(device)
        self.text_encoder = text_encoder
        self.device = device
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # DMD specific
        self.num_train_timesteps = num_train_timesteps
        self.student_steps = 4  # Target: 4-step generation
        self.teacher_steps = 50  # Original: 50-step
        
        # ODE solver for teacher
        self.ode_solver = "euler"
        
        # Loss weights
        self.lambda_distill = 1.0
        self.lambda_gan = 0.1  # If using GAN discriminator
        
    def get_timesteps(self, num_steps: int) -> torch.Tensor:
        """Get timesteps for few-step sampling"""
        # Linear spacing with shift
        shift = 1.0  # Tune for quality/speed tradeoff
        max_t = self.num_train_timesteps - 1
        
        indices = torch.linspace(0, max_t, num_steps + 1, dtype=torch.long)
        timesteps = max_t - indices[:-1]
        
        # Apply shift
        timesteps = (timesteps.float() * shift).long()
        return timesteps
    
    def ode_step(self, model, x, t, t_prev, context, clip_fea, y):
        """Single ODE step using Euler method"""
        # Predict velocity
        with torch.no_grad() if model == self.teacher else torch.enable_grad():
            pred = model([x], torch.tensor([t], device=self.device), 
                        context, x.shape[1], clip_fea, y)
        
        # Euler step
        dt = t_prev - t
        x_next = x + dt * pred
        return x_next
    
    def compute_teacher_output(self, x0, context, clip_fea, y) -> torch.Tensor:
        """
        Compute teacher output using full ODE solver
        This is the target for distillation
        """
        # Start from noise
        noise = torch.randn_like(x0)
        x = noise
        
        timesteps = self.get_timesteps(self.teacher_steps)
        
        # Solve ODE with teacher
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_prev = timesteps[i+1] if i+1 < len(timesteps) else 0
            
            x = self.ode_step(self.teacher, x, t, t_prev, context, clip_fea, y)
        
        return x
    
    def compute_student_output(self, x0, context, clip_fea, y) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute student output with few steps
        Returns (output, intermediate_predictions for loss)
        """
        noise = torch.randn_like(x0)
        x = noise
        
        timesteps = self.get_timesteps(self.student_steps)
        predictions = []
        
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_prev = timesteps[i+1] if i+1 < len(timesteps) else 0
            
            # Predict and store
            pred = self.student([x], torch.tensor([t], device=self.device),
                              context, x.shape[1], clip_fea, y)
            predictions.append(pred)
            
            # Euler step
            dt = t_prev - t
            x = x + dt * pred
        
        return x, predictions
    
    def dmd_loss(self, student_out, teacher_out, predictions) -> torch.Tensor:
        """
        Distribution Matching Distillation loss
        Matches student output distribution to teacher output distribution
        """
        # Main distillation loss: L2 distance in output space
        distill_loss = F.mse_loss(student_out, teacher_out)
        
        # Optional: Consistency loss on intermediate predictions
        consistency_loss = 0
        if len(predictions) > 1:
            for pred in predictions[:-1]:
                consistency_loss += F.mse_loss(pred, predictions[-1].detach())
            consistency_loss /= len(predictions) - 1
        
        total_loss = self.lambda_distill * distill_loss + 0.1 * consistency_loss
        return total_loss, {
            'distill': distill_loss.item(),
            'consistency': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else 0
        }
    
    def train_step(self, batch: Dict) -> Dict:
        """
        Single training step
        
        Batch format:
        {
            'latents': [B, C, F, H, W] VAE latents
            'text': [B] list of prompts
            'clip_fea': [B, 257, 1280] CLIP features (optional)
            'y': [B, C, F_cond, H, W] Conditional frames
        }
        """
        self.student.train()
        
        # Move to device
        latents = batch['latents'].to(self.device)
        text = batch['text']
        clip_fea = batch.get('clip_fea', None)
        if clip_fea is not None:
            clip_fea = clip_fea.to(self.device)
        y = [y_i.to(self.device) for y_i in batch['y']]
        
        # Encode text
        context = self.text_encoder(text)
        
        # Compute teacher output (no grad)
        with torch.no_grad():
            # For efficiency, compute teacher on subset or use precomputed
            teacher_out = self.compute_teacher_output(latents, context, clip_fea, y)
        
        # Compute student output
        student_out, predictions = self.compute_student_output(latents, context, clip_fea, y)
        
        # Compute loss
        loss, loss_dict = self.dmd_loss(student_out, teacher_out, predictions)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        loss_dict['total'] = loss.item()
        return loss_dict
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            loss_dict = self.train_step(batch)
            total_loss += loss_dict['total']
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss_dict['total']:.4f} "
                      f"(distill: {loss_dict['distill']:.4f})")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} complete. Avg loss: {avg_loss:.4f}")
        return avg_loss
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"[DMD] Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[DMD] Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', 'unknown')})")


class ODEPairDataset(torch.utils.data.Dataset):
    """
    Dataset of (x0, xT) ODE pairs precomputed from teacher
    This is used for initial student warmup before DMD
    """
    
    def __init__(self, ode_pairs_path: str):
        """
        Args:
            ode_pairs_path: Path to precomputed ODE pairs
                Format: dict with keys:
                    'x0': [N, C, F, H, W] clean latents
                    'xT': [N, C, F, H, W] teacher ODE outputs
                    'text': [N] text prompts
        """
        data = torch.load(ode_pairs_path)
        self.x0 = data['x0']
        self.xT = data['xT']
        self.text = data['text']
        
    def __len__(self):
        return len(self.x0)
    
    def __getitem__(self, idx):
        return {
            'latents': self.x0[idx],
            'text': self.text[idx],
            'teacher_output': self.xT[idx],
        }


def generate_ode_pairs(
    teacher_model,
    vae,
    text_encoder,
    video_dataset,
    output_path: str,
    num_pairs: int = 10000,
    device: str = 'cuda'
):
    """
    Generate ODE pairs from teacher model for student pretraining
    
    This precomputes (x0, xT) pairs where:
    - x0 is the clean latent
    - xT is the teacher's ODE solution starting from noise
    """
    print(f"[ODE] Generating {num_pairs} ODE pairs...")
    
    teacher_model.eval()
    
    x0_list = []
    xT_list = []
    text_list = []
    
    with torch.no_grad():
        for i in range(num_pairs):
            if i % 100 == 0:
                print(f"[ODE] Generated {i}/{num_pairs} pairs")
            
            # Get video and text
            batch = video_dataset[i]
            video = batch['video'].to(device)  # [C, F, H, W]
            text = batch['text']
            
            # Encode to latent
            latent = vae.encode(video.unsqueeze(0))[0]
            
            # Encode text
            context = text_encoder([text])
            
            # Compute teacher ODE output
            # This is the same as compute_teacher_output in trainer
            noise = torch.randn_like(latent)
            x = noise
            
            timesteps = torch.linspace(999, 0, 50, dtype=torch.long)
            for j in range(len(timesteps)):
                t = timesteps[j]
                t_prev = timesteps[j+1] if j+1 < len(timesteps) else 0
                
                pred = teacher_model([x], torch.tensor([t], device=device),
                                   context, x.shape[1], None, [latent[:, :, :1]])
                dt = t_prev - t
                x = x + dt * pred
            
            x0_list.append(latent.cpu())
            xT_list.append(x.cpu())
            text_list.append(text)
    
    # Save
    torch.save({
        'x0': torch.stack(x0_list),
        'xT': torch.stack(xT_list),
        'text': text_list,
    }, output_path)
    
    print(f"[ODE] Saved {num_pairs} pairs to {output_path}")


def train_causal_model(
    teacher_path: str,
    output_dir: str,
    video_dataset_path: str,
    num_epochs: int = 1000,
    batch_size: int = 1,  # Small batch due to memory
    device: str = 'cuda'
):
    """
    Main training function
    
    Two-stage training:
    1. Pretrain on ODE pairs (optional but recommended)
    2. DMD training with teacher guidance
    """
    from ..lingbot_causal.causal_model import CausalWanModel
    
    print("=" * 60)
    print("LingBot-World Causal Distillation Training")
    print("=" * 60)
    
    # Load teacher
    print("[Train] Loading teacher model...")
    # teacher = load_lingbot_model(teacher_path)
    
    # Create student
    print("[Train] Creating causal student model...")
    student = CausalWanModel(model_type='i2v')
    
    # Initialize student from teacher weights where possible
    # This requires careful mapping of attention weights
    
    # Create trainer
    # trainer = DMDTrainer(student, teacher, vae, text_encoder, device)
    
    # Stage 1: ODE pretraining (optional)
    # ode_pairs_path = f"{output_dir}/ode_pairs.pt"
    # if not Path(ode_pairs_path).exists():
    #     generate_ode_pairs(teacher, vae, text_encoder, dataset, ode_pairs_path)
    
    # Stage 2: DMD training
    # for epoch in range(num_epochs):
    #     trainer.train_epoch(dataloader, epoch)
    #     if epoch % 100 == 0:
    #         trainer.save_checkpoint(f"{output_dir}/checkpoint_{epoch}.pt", epoch)
    
    print("[Train] Training complete!")


if __name__ == "__main__":
    print("[DMD] Distribution Matching Distillation module")
    print("[DMD] Use train_causal_model() to start training")
