#!/usr/bin/env python3
"""Download LingBot-World model from HuggingFace"""

from huggingface_hub import snapshot_download
import sys

print("Downloading LingBot-World model...")
print("This will take ~10-15 minutes (74GB)")

try:
    snapshot_download(
        repo_id="robbyant/lingbot-world-base-cam",
        local_dir="/workspace/lingbot-world",
        resume_download=True,
        local_dir_use_symlinks=False
    )
    print("✅ Download complete!")
    print("Model saved to: /workspace/lingbot-world/")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
