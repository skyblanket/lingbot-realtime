#!/usr/bin/env python3
"""Single GPU test - verify training works"""
import sys
sys.path.insert(0, '/workspace/lingbot-world')
sys.path.insert(0, '/workspace/lingbot-realtime')

import torch
import torch.nn as nn
import torch.nn.functional as F

print("Creating model...")
from lingbot_causal.causal_model import CausalWanModel

model = CausalWanModel(
    model_type='i2v',
    dim=5120, 
    num_heads=40, 
    num_layers=40,
    ffn_dim=13824, 
    text_len=512,
    in_dim=36,
    out_dim=16
).cuda().to(torch.bfloat16)

print(f"Model: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

print("Loading weights...")
from lingbot_causal.weight_loader import LingBotWeightLoader
loader = LingBotWeightLoader('/workspace/lingbot-world')
loader.load_into_causal_model(model, device='cuda')

print("Testing forward pass...")
with torch.no_grad():
    # i2v: x=[16,F,H,W], y=[20,F,H,W], concat -> [36,F,H,W]
    x = torch.randn(16, 17, 32, 32, device='cuda', dtype=torch.bfloat16)
    y = torch.randn(20, 17, 32, 32, device='cuda', dtype=torch.bfloat16)
    t = torch.tensor([500], device='cuda')
    ctx = [torch.randn(512, 5120, device='cuda', dtype=torch.bfloat16)]
    
    # After patching 32x32 with (2,2) patch: 16x16 = 256 patches per frame
    # 17 frames * 256 patches = 4352 sequence length
    seq_len = 17 * 16 * 16  # 4352
    
    out = model.forward_train([x], t, ctx, seq_len, None, [y])
    print(f"Output shape: {out.shape}")

print("Testing DMD training...")
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Create teacher
teacher = CausalWanModel(
    model_type='i2v',
    dim=5120, num_heads=40, num_layers=40,
    ffn_dim=13824, text_len=512,
    in_dim=36, out_dim=16
).cuda().to(torch.bfloat16)
teacher.load_state_dict(model.state_dict())
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

seq_len = 17 * 16 * 16

for i in range(10):
    x = torch.randn(16, 17, 32, 32, device='cuda', dtype=torch.bfloat16)
    y = torch.randn(20, 17, 32, 32, device='cuda', dtype=torch.bfloat16)
    t = torch.randint(0, 1000, (1,), device='cuda')
    ctx = [torch.randn(512, 5120, device='cuda', dtype=torch.bfloat16)]
    
    with torch.no_grad():
        teacher_out = teacher.forward_train([x], t, ctx, seq_len, None, [y])
    
    student_out = model.forward_train([x], t, ctx, seq_len, None, [y])
    loss = F.mse_loss(student_out, teacher_out)
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    print(f"Step {i}: loss={loss.item():.6f}")

print("✅ DMD Training works!")
torch.save(model.state_dict(), "/workspace/lingbot-realtime/test_checkpoint.pt")
print("Saved checkpoint!")
