#!/usr/bin/env python3
"""
Test LingBot-World Streaming Performance
Measures FPS and latency for real-time generation
"""

import torch
import numpy as np
import time
import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, '/home/sky/lingbot-world')
sys.path.insert(0, str(Path(__file__).parent))

from wan.modules.t5 import T5EncoderModel
from lingbot_causal.causal_model import CausalWanModel
from lingbot_causal.weight_loader import LingBotWeightLoader


def test_streaming_performance(num_frames=10):
    """Test streaming generation performance"""
    
    device = 'cuda'
    dtype = torch.bfloat16
    
    print("="*70)
    print("LingBot-World Streaming Performance Test")
    print("="*70)
    
    # Load model
    print("\n[Perf] Loading model...")
    model = CausalWanModel(
        dim=5120, num_heads=40, num_layers=40,
        ffn_dim=13824, freq_dim=256, text_len=512,
        patch_size=(1, 2, 2), model_type='i2v'
    ).to(device).to(dtype)
    
    loader = LingBotWeightLoader('/home/sky/lingbot-world/lingbot-world-base-cam')
    loader.load_into_causal_model(model, device=device)
    model.eval()
    print("[Perf] Model loaded!")
    
    # Load text encoder
    print("[Perf] Loading text encoder...")
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device=device,
        checkpoint_path="/home/sky/lingbot-world/lingbot-world-base-cam/models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path="google/umt5-xxl",
    )
    print("[Perf] Text encoder loaded!")
    
    # Test prompt
    prompt = "first person view, walking forward, indoor environment"
    
    # Encode text (once, reused)
    with torch.no_grad():
        context = text_encoder([prompt], device=device)
    
    print(f"\n[Perf] Generating {num_frames} frames...")
    print("-"*70)
    
    # Warm up
    print("Warming up...")
    noise = torch.randn(1, 16, 1, 32, 32, device=device, dtype=dtype)
    t = torch.tensor([999], device=device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            _ = model.forward_train([noise], t, context, 1, None, [noise[:, :, :1]])
    torch.cuda.synchronize()
    
    # Initialize KV cache
    kv_cache = model.create_kv_cache(1, device, dtype)
    
    # Generation timing
    frame_times = []
    current_latent = None
    
    for i in range(num_frames):
        frame_start = time.time()
        
        if i == 0:
            # First frame - full generation
            noise = torch.randn(1, 16, 1, 32, 32, device=device, dtype=dtype)
            t = torch.tensor([999], device=device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    current_latent = model.forward_train(
                        [noise], t, context, 1, None, [noise[:, :, :1]]
                    )
        else:
            # Streaming generation with KV cache
            t = torch.tensor([500], device=device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    current_latent = model.forward_inference(
                        [current_latent], t, context, 1, None, [current_latent[:, :, :1]],
                        kv_cache=kv_cache,
                        current_start=0,
                        current_end=current_latent.shape[2]
                    )
        
        torch.cuda.synchronize()
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        
        fps = 1.0 / frame_time
        print(f"Frame {i+1}/{num_frames}: {frame_time*1000:.1f}ms | {fps:.2f} FPS")
    
    # Stats
    print("-"*70)
    print("\n📊 PERFORMANCE RESULTS:")
    print(f"  Average frame time: {np.mean(frame_times)*1000:.1f}ms")
    print(f"  Average FPS: {1.0/np.mean(frame_times):.2f}")
    print(f"  Min FPS: {1.0/np.max(frame_times):.2f}")
    print(f"  Max FPS: {1.0/np.min(frame_times):.2f}")
    print(f"  Std dev: {np.std(frame_times)*1000:.1f}ms")
    
    # Memory
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"\n💾 GPU Memory:")
    print(f"  Allocated: {allocated:.1f} GB")
    print(f"  Reserved: {reserved:.1f} GB")
    
    # Projections
    print("\n🎯 PROJECTIONS:")
    current_fps = 1.0 / np.mean(frame_times)
    target_fps = 16.0
    
    if current_fps < target_fps:
        speedup_needed = target_fps / current_fps
        print(f"  Current: {current_fps:.2f} FPS")
        print(f"  Target: {target_fps:.2f} FPS")
        print(f"  Speedup needed: {speedup_needed:.1f}x")
        print(f"  DMD (50→4 steps) provides: ~12.5x speedup")
        if speedup_needed <= 12.5:
            print(f"  ✅ DMD should achieve real-time!")
        else:
            print(f"  ⚠️  May need additional optimization")
    else:
        print(f"  ✅ Already real-time! ({current_fps:.2f} FPS)")
    
    print("\n" + "="*70)
    
    return {
        'avg_fps': 1.0 / np.mean(frame_times),
        'avg_latency_ms': np.mean(frame_times) * 1000,
        'memory_gb': allocated
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=10, help='Number of frames to generate')
    args = parser.parse_args()
    
    results = test_streaming_performance(args.frames)
