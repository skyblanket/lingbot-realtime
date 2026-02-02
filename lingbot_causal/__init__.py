"""
LingBot-World Causal: Real-time streaming world model
Modified LingBot-World with block-wise causal attention for real-time WASD control
"""

import sys
import os

# Add lingbot-world to path for importing wan modules
LINGBOT_PATH = '/home/sky/lingbot-world'
if LINGBOT_PATH not in sys.path:
    sys.path.insert(0, LINGBOT_PATH)

try:
    from .causal_model import CausalWanModel
    from .streaming_inference import StreamingInference
    __all__ = ['CausalWanModel', 'StreamingInference']
except ImportError as e:
    print(f"[LingBot-Causal] Warning: Could not import models: {e}")
    print(f"[LingBot-Causal] Make sure lingbot-world is at {LINGBOT_PATH}")
    __all__ = []
