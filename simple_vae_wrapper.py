#!/usr/bin/env python3
"""Simple VAE wrapper using diffusers or local implementation"""

import torch
import sys
from pathlib import Path

# Try to use existing WanVAE implementation
sys.path.insert(0, '/workspace/lingbot-realtime/weights')

try:
    # Try importing from diffusers
    from diffusers import AutoencoderKLWan
    print("Using diffusers Wan VAE")
    
    class SimpleVAE:
        def __init__(self, vae_path, device='cuda'):
            self.vae = AutoencoderKLWan.from_pretrained(
                vae_path,
                subfolder="vae",
                torch_dtype=torch.bfloat16
            ).to(device)
            self.device = device
            
        def encode(self, videos):
            """videos: list of [C,F,H,W] tensors"""
            with torch.no_grad():
                latents = []
                for video in videos:
                    # video: [C, F, H, W]
                    latent = self.vae.encode(video.unsqueeze(0)).latent_dist.sample()
                    latents.append(latent)
                return torch.stack(latents)
        
        def decode(self, latents):
            """latents: [B,C,F,H,W]"""
            with torch.no_grad():
                decoded = []
                for latent in latents:
                    video = self.vae.decode(latent.unsqueeze(0)).sample
                    decoded.append(video[0])
                return decoded
                
except ImportError:
    print("Diffusers not available, using placeholder VAE")
    
    class SimpleVAE:
        def __init__(self, vae_path, device='cuda'):
            self.device = device
            print(f"WARNING: Using dummy VAE - real VAE should be at {vae_path}")
            
        def encode(self, videos):
            # Dummy: just downsample
            return torch.randn(len(videos), 16, 17, 32, 32, device=self.device, dtype=torch.bfloat16)
        
        def decode(self, latents):
            # Dummy: return random noise
            return [torch.randn(3, 17, 512, 512, device=self.device) for _ in range(len(latents))]


if __name__ == "__main__":
    # Test
    vae = SimpleVAE('/workspace/lingbot-realtime/weights', device='cuda')
    print("VAE loaded successfully")
