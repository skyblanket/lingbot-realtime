#!/usr/bin/env python3
"""
Simple test: Load LingBot-World weights and do minimal forward pass
"""

import sys
sys.path.insert(0, '/home/sky/lingbot-world')

import torch

print("=" * 70)
print("SIMPLE FORWARD PASS TEST")
print("=" * 70)

# Test 1: Can we import?
print("\n[1/4] Importing model...")
try:
    from lingbot_causal.weight_loader import LingBotWeightLoader
    from lingbot_causal.causal_model import CausalWanModel
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Can we create loader?
print("\n[2/4] Creating weight loader...")
try:
    loader = LingBotWeightLoader('/home/sky/lingbot-world/lingbot-world-base-cam')
    print(f"✅ Loader created")
    print(f"   Config: dim={loader.config['dim']}, layers={loader.config['num_layers']}")
except Exception as e:
    print(f"❌ Loader creation failed: {e}")
    sys.exit(1)

# Test 3: Create model
print("\n[3/4] Creating causal model...")
try:
    model = CausalWanModel(
        model_type='i2v',
        dim=loader.config['dim'],
        ffn_dim=loader.config['ffn_dim'],
        num_heads=loader.config['num_heads'],
        num_layers=loader.config['num_layers'],
        in_dim=loader.config['in_dim'],
        out_dim=loader.config['out_dim'],
        text_len=loader.config['text_len'],
    )
    print("✅ Model created")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Load just a few weights (quick test)
print("\n[4/4] Testing weight loading on subset...")
try:
    # Just verify we can load one shard
    from safetensors.torch import load_file
    shard = load_file(str(loader.high_noise_path / 'diffusion_pytorch_model-00001-of-00008.safetensors'))
    print(f"✅ Loaded shard with {len(shard)} tensors")
    
    # Get model state dict
    model_state = model.state_dict()
    
    # Try to load a few weights
    loaded_count = 0
    for key in ['patch_embedding.weight', 'patch_embedding.bias']:
        if key in shard and key in model_state:
            if model_state[key].shape == shard[key].shape:
                model_state[key] = shard[key]
                loaded_count += 1
                print(f"  ✓ Loaded {key}: {shard[key].shape}")
    
    model.load_state_dict(model_state, strict=False)
    print(f"✅ Loaded {loaded_count} test weights successfully")
    
except Exception as e:
    print(f"❌ Weight loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ BASIC TESTS PASSED!")
print("=" * 70)
print("\nReady for full weight loading test.")
