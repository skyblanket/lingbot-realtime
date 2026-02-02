#!/usr/bin/env python3
"""
Test script: Load LingBot-World weights into Causal Model
"""

import sys
sys.path.insert(0, '/home/sky/lingbot-world')

import torch
from lingbot_causal.weight_loader import load_lingbot_weights_into_causal

print("=" * 70)
print("LINGBOT-WORLD CAUSAL MODEL - WEIGHT LOADING TEST")
print("=" * 70)

# Step 1: Load weights
print("\n[1/4] Loading 74GB model weights...")
print("      This will take ~2-3 minutes...")

try:
    model = load_lingbot_weights_into_causal(
        lingbot_path="/home/sky/lingbot-world/lingbot-world-base-cam",
        device='cpu'  # Load on CPU first, then move to GPU
    )
    print("✅ Weights loaded successfully!")
    
    # Print model stats
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params / 1e9:.1f}B")
    
except Exception as e:
    print(f"❌ Failed to load weights: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Move to GPU with proper dtype
print("\n[2/4] Moving model to GPU...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use bfloat16 - Blackwell GPUs need it for Conv3d
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    print(f"✅ Model on {device} (bfloat16)")
    if device == 'cuda':
        print(f"   GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB")
except Exception as e:
    print(f"⚠️  Failed to move to GPU: {e}")
    print("   Continuing on CPU...")
    device = 'cpu'

# Step 3: Test forward pass (dummy input)
print("\n[3/4] Testing forward pass with dummy input...")
try:
    model.eval()
    
    # Create dummy input
    batch_size = 1
    num_frames = 1
    height, width = 480, 832
    
    # Latent dimensions (after VAE encoding)
    latent_h = height // 8
    latent_w = width // 8
    
    # Create dummy latent frame (16 channels) - the noised input
    dummy_latent = torch.randn(
        batch_size, 16, num_frames, latent_h, latent_w,
        device=device, dtype=torch.bfloat16
    )
    
    # Create conditional frame (16 channels) - usually the first frame
    dummy_cond = torch.randn(
        batch_size, 16, num_frames, latent_h, latent_w,
        device=device, dtype=torch.bfloat16
    )
    
    # Create camera feature (4 channels) - position + rotation
    # These are processed separately through patch_embedding_wancamctrl
    dummy_camera = torch.randn(
        batch_size, 4, num_frames, latent_h, latent_w,
        device=device, dtype=torch.bfloat16
    )
    
    # y should be concat of conditional (16) + camera (4) = 20 channels
    # Forward_train will concat x (16) + y (20) = 36 channels for patch embedding
    dummy_y = torch.cat([dummy_cond, dummy_camera], dim=1)
    
    # Timestep
    t = torch.tensor([500], device=device)
    
    # Text context (dummy) - T5 embedding
    # LingBot-World uses text_len=512 tokens, embedding dim=4096
    # The text_embedding layer projects 4096 -> dim (5120)
    # Use shorter context (77 tokens) for faster testing
    context = [torch.randn(77, 4096, device=device, dtype=torch.bfloat16)]
    
    print(f"   x shape: {dummy_latent.shape} (noised latent)")
    print(f"   y shape: {dummy_y.shape} (condition 16 + camera 4)")
    print(f"   Concat: {dummy_latent.shape[1]} + {dummy_y.shape[1]} = 36 channels")
    print(f"   Expected seq_len (after patchify): {num_frames} * {latent_h // 2} * {latent_w // 2} = {num_frames * (latent_h // 2) * (latent_w // 2)}")
    print(f"   Timestep: {t.item()}")
    
    # Calculate seq_len properly
    # After patch embedding with kernel_size=(1,2,2), stride=(1,2,2):
    # - Time: F stays same (1)
    # - Height: H/2 (60/2=30)
    # - Width: W/2 (104/2=52)
    # So seq_len = F * (H/2) * (W/2) = 1 * 30 * 52 = 1560
    patch_h = latent_h // 2
    patch_w = latent_w // 2
    seq_len = num_frames * patch_h * patch_w
    
    print(f"   Computed seq_len: {num_frames} * {patch_h} * {patch_w} = {seq_len}")
    
    # Forward pass with autocast for mixed precision
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output = model(
            [dummy_latent[0]],  # x: noised latent (16 channels)
            t,
            context,
            seq_len=seq_len,
            clip_fea=None,
            y=[dummy_y[0]]  # y: condition + camera (20 channels)
        )
    
    print(f"✅ Forward pass successful!")
    print(f"   Output shape: {output.shape}")
    
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test streaming mode
print("\n[4/4] Testing streaming inference mode...")
try:
    # Create KV cache (use bfloat16 for Blackwell)
    kv_cache = model.create_kv_cache(batch_size, device, torch.bfloat16)
    crossattn_cache = [{} for _ in range(len(model.blocks))]
    
    print("✅ KV cache created")
    print(f"   Cache layers: {len(kv_cache)}")
    print(f"   Cache shapes: k={kv_cache[0]['k'].shape}, v={kv_cache[0]['v'].shape}")
    
    # Test streaming forward with autocast
    # current_start/end should be the actual token positions in the cache
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output = model.forward_inference(
            [dummy_latent[0]],
            t,
            context,
            seq_len=seq_len,
            clip_fea=None,
            y=[dummy_y[0]],
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=0,
            current_end=seq_len  # Full sequence length for first frame
        )
    
    print(f"✅ Streaming forward pass successful!")
    print(f"   Output shape: {output.shape}")
    
except Exception as e:
    print(f"❌ Streaming mode failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nThe causal model is ready for:")
print("  • Real-time streaming generation")
print("  • 4-step distillation training")
print("  • WASD interactive demo")
