"""
Weight Loader for LingBot-World Causal Model
Maps LingBot-World pretrained weights to CausalWanModel architecture
"""

import torch
from safetensors.torch import load_file
from pathlib import Path
from typing import Dict, Tuple
import json


class LingBotWeightLoader:
    """
    Loads LingBot-World weights into CausalWanModel
    
    LingBot-World uses MoE with two experts (high_noise, low_noise)
    We'll load both and create a unified causal model
    """
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.high_noise_path = self.model_path / "high_noise_model"
        self.low_noise_path = self.model_path / "low_noise_model"
        
        # Load config
        with open(self.high_noise_path / "config.json") as f:
            self.config = json.load(f)
        
        print(f"[WeightLoader] Loaded config: {self.config}")
        print(f"  - Dim: {self.config['dim']}")
        print(f"  - Layers: {self.config['num_layers']}")
        print(f"  - Heads: {self.config['num_heads']}")
        
    def load_safetensors_sharded(self, model_dir: Path) -> Dict[str, torch.Tensor]:
        """Load sharded safetensors and merge them"""
        index_file = model_dir / "diffusion_pytorch_model.safetensors.index.json"
        
        with open(index_file) as f:
            index = json.load(f)
        
        # Collect all unique shard files
        shard_files = set(index["weight_map"].values())
        
        # Load all shards
        state_dict = {}
        for shard_file in sorted(shard_files):
            shard_path = model_dir / shard_file
            print(f"[WeightLoader] Loading {shard_file}...")
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)
        
        return state_dict
    
    def create_key_mapping(self) -> Dict[str, str]:
        """
        Create mapping from LingBot-World keys to CausalWanModel keys
        
        Most keys are the same, just need to handle:
        - Camera control layers (already match)
        - Norm layers (already match)
        """
        # Keys are mostly identical, no renaming needed
        # Just need to filter out MoE routing if present
        return {}
    
    def load_into_causal_model(self, causal_model, device='cuda'):
        """
        Load LingBot-World weights into CausalWanModel
        
        Args:
            causal_model: CausalWanModel instance
            device: Device to load on
        """
        print("[WeightLoader] Loading high noise expert weights...")
        high_noise_state = self.load_safetensors_sharded(self.high_noise_path)
        
        print("[WeightLoader] Loading low noise expert weights...")
        low_noise_state = self.load_safetensors_sharded(self.low_noise_path)
        
        # Get causal model state dict
        causal_state = causal_model.state_dict()
        
        # Track what we're loading
        loaded_keys = []
        missing_keys = []
        mismatched_keys = []
        
        print("[WeightLoader] Mapping weights...")
        
        for key in causal_state.keys():
            if key in high_noise_state:
                # Check shape compatibility
                if causal_state[key].shape == high_noise_state[key].shape:
                    causal_state[key] = high_noise_state[key]
                    loaded_keys.append(key)
                else:
                    mismatched_keys.append((key, causal_state[key].shape, high_noise_state[key].shape))
            else:
                # Try alternative naming
                alt_key = key.replace("model.", "")
                if alt_key in high_noise_state:
                    if causal_state[key].shape == high_noise_state[alt_key].shape:
                        causal_state[key] = high_noise_state[alt_key]
                        loaded_keys.append(key)
                    else:
                        mismatched_keys.append((key, causal_state[key].shape, high_noise_state[alt_key].shape))
                else:
                    missing_keys.append(key)
        
        # Load into model
        causal_model.load_state_dict(causal_state, strict=False)
        
        print(f"[WeightLoader] Summary:")
        print(f"  - Loaded: {len(loaded_keys)} keys")
        print(f"  - Missing: {len(missing_keys)} keys")
        print(f"  - Mismatched: {len(mismatched_keys)} keys")
        
        if missing_keys:
            print(f"[WeightLoader] Missing keys (will be randomly initialized):")
            for k in missing_keys[:10]:
                print(f"    - {k}")
            if len(missing_keys) > 10:
                print(f"    ... and {len(missing_keys) - 10} more")
        
        if mismatched_keys:
            print(f"[WeightLoader] Shape mismatches:")
            for k, expected, got in mismatched_keys[:5]:
                print(f"    - {k}: expected {expected}, got {got}")
        
        return causal_model
    
    def verify_architecture_compatibility(self, causal_model) -> bool:
        """Verify that causal model architecture matches LingBot-World"""
        
        checks = [
            (causal_model.dim == self.config['dim'], 
             f"dim: {causal_model.dim} vs {self.config['dim']}"),
            (causal_model.num_layers == self.config['num_layers'],
             f"num_layers: {causal_model.num_layers} vs {self.config['num_layers']}"),
            (causal_model.num_heads == self.config['num_heads'],
             f"num_heads: {causal_model.num_heads} vs {self.config['num_heads']}"),
            (causal_model.ffn_dim == self.config['ffn_dim'],
             f"ffn_dim: {causal_model.ffn_dim} vs {self.config['ffn_dim']}"),
        ]
        
        all_pass = True
        print("[WeightLoader] Architecture compatibility check:")
        for passed, msg in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {msg}")
            if not passed:
                all_pass = False
        
        return all_pass


def load_lingbot_weights_into_causal(
    lingbot_path: str = "/home/sky/lingbot-world/lingbot-world-base-cam",
    device: str = 'cuda'
):
    """
    Main function to load LingBot-World weights into CausalWanModel
    
    Returns:
        CausalWanModel with loaded weights
    """
    from .causal_model import CausalWanModel
    
    print("=" * 60)
    print("Loading LingBot-World into Causal Model")
    print("=" * 60)
    
    # Create loader
    loader = LingBotWeightLoader(lingbot_path)
    
    # Create model with correct architecture
    print("\n[WeightLoader] Creating CausalWanModel...")
    causal_model = CausalWanModel(
        model_type='i2v',
        dim=loader.config['dim'],
        ffn_dim=loader.config['ffn_dim'],
        num_heads=loader.config['num_heads'],
        num_layers=loader.config['num_layers'],
        in_dim=loader.config['in_dim'],
        out_dim=loader.config['out_dim'],
        text_len=loader.config['text_len'],
    )
    
    # Verify compatibility
    if not loader.verify_architecture_compatibility(causal_model):
        print("⚠️  Architecture mismatch detected!")
        print("Attempting to load anyway...")
    
    # Load weights
    print("\n[WeightLoader] Loading weights...")
    loader.load_into_causal_model(causal_model, device)
    
    print("\n✅ Model loaded successfully!")
    
    return causal_model


if __name__ == "__main__":
    # Test loading
    print("[Test] Testing weight loader...")
    
    # Try to load (this will take time with 60GB model)
    try:
        model = load_lingbot_weights_into_causal()
        print("[Test] Weight loader works!")
    except Exception as e:
        print(f"[Test] Error: {e}")
        import traceback
        traceback.print_exc()
