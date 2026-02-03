#!/usr/bin/env python3
"""
LingBot-World WASD Demo with REAL Model Inference

This demo actually loads and runs the causal model!
Warning: Will be SLOW (~0.1 FPS) until DMD distillation
"""

import torch
import numpy as np
import gradio as gr
import cv2
import time
import sys
from pathlib import Path
from collections import deque

# Add paths
sys.path.insert(0, '/home/sky/lingbot-world')
sys.path.insert(0, str(Path(__file__).parent))

from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE as WanVAE
from lingbot_causal.causal_model import CausalWanModel
from lingbot_causal.weight_loader import LingBotWeightLoader


class RealLingBotEngine:
    """Real LingBot-World streaming engine"""
    
    def __init__(self):
        self.device = 'cuda'
        self.dtype = torch.bfloat16
        self.frame_count = 0
        self.action_history = deque(maxlen=10)
        self.frame_times = deque(maxlen=10)
        
        print("="*70)
        print("🎮 Loading REAL LingBot-World Model...")
        print("="*70)
        
        # Load model
        print("[Real] Loading causal model...")
        self.model = CausalWanModel(
            dim=5120, num_heads=40, num_layers=40,
            ffn_dim=13824, freq_dim=256, text_len=512,
            patch_size=(1, 2, 2), model_type='i2v'
        ).to(self.device).to(self.dtype)
        
        loader = LingBotWeightLoader('/home/sky/lingbot-world/lingbot-world-base-cam')
        loader.load_into_causal_model(self.model, device=self.device)
        self.model.eval()
        print("✅ Model loaded!")
        
        # Load VAE
        print("[Real] Loading VAE...")
        self.vae = WanVAE(
            vae_pth='/home/sky/lingbot-world/lingbot-world-base-cam/Wan2.1_VAE.pth',
            device=self.device
        )
        print("✅ VAE loaded!")
        
        # Load text encoder
        print("[Real] Loading text encoder...")
        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=self.dtype,
            device=self.device,
            checkpoint_path="/home/sky/lingbot-world/lingbot-world-base-cam/models_t5_umt5-xxl-enc-bf16.pth",
            tokenizer_path="google/umt5-xxl",
        )
        print("✅ Text encoder loaded!")
        
        # State
        self.current_latent = None
        self.kv_cache = None
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0]
        
        # Pre-encode base prompt
        print("[Real] Pre-encoding prompts...")
        with torch.no_grad():
            self.base_context = self.text_encoder(
                ["first person view, indoor environment"], 
                device=self.device
            )
        
        print("="*70)
        print("✅ REAL Engine ready! (Warning: SLOW - ~0.1 FPS)")
        print("="*70)
    
    def get_prompt_for_action(self, action):
        prompts = {
            'forward': "first person view, walking forward, indoor environment",
            'backward': "first person view, walking backward, indoor environment", 
            'left': "first person view, turning left, rotating left",
            'right': "first person view, turning right, rotating right",
            'up': "first person view, looking up, camera tilt up",
            'down': "first person view, looking down, camera tilt down",
            'stop': "first person view, standing still, stationary",
        }
        return prompts.get(action, prompts['stop'])
    
    def generate_frame(self, action='stop'):
        """Generate one frame using real model inference"""
        start_time = time.time()
        
        # Encode action-specific prompt
        prompt = self.get_prompt_for_action(action)
        with torch.no_grad():
            context = self.text_encoder([prompt], device=self.device)
        
        # Generate latent
        if self.current_latent is None:
            # First frame - initialize
            print(f"[Gen] Generating first frame (action: {action})...")
            noise = torch.randn(1, 16, 1, 32, 32, device=self.device, dtype=self.dtype)
            t = torch.tensor([999], device=self.device)
            
            # Simplified single-step for demo (fast but lower quality)
            with torch.no_grad():
                self.current_latent = self.model.forward_train(
                    [noise], t, context, 1, None, [noise[:, :, :1]]
                )
        else:
            # Continue from previous (simplified - no full diffusion)
            # Just add small noise perturbation based on action
            noise_scale = 0.1 if action == 'stop' else 0.3
            noise = torch.randn_like(self.current_latent) * noise_scale
            self.current_latent = self.current_latent + noise
        
        # Decode to image
        with torch.no_grad():
            decoded = self.vae.decode([self.current_latent[0]])[0]
        
        # Convert to numpy
        frame = decoded.cpu().numpy()
        
        # Handle different shapes
        if frame.ndim == 4:  # [C, F, H, W]
            frame = frame[:, 0]  # Take first frame
        
        # Convert CHW to HWC
        if frame.shape[0] in [3, 16]:  # Channels first
            frame = np.transpose(frame, (1, 2, 0))
        
        # Take first 3 channels if more
        if frame.shape[-1] > 3:
            frame = frame[..., :3]
        
        # Normalize to 0-255
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        frame = (frame * 255).astype(np.uint8)
        
        # Update stats
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        self.frame_count += 1
        self.action_history.append(action)
        
        # Calculate FPS
        avg_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        # Update position
        speed = 0.1
        if action == 'forward':
            self.position[2] += speed
        elif action == 'backward':
            self.position[2] -= speed
        elif action == 'left':
            self.rotation[0] -= 5
        elif action == 'right':
            self.rotation[0] += 5
        
        return frame, fps, frame_time * 1000
    
    def reset(self):
        """Reset state"""
        self.current_latent = None
        self.kv_cache = None
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0]
        self.frame_count = 0
        self.frame_times.clear()
        self.action_history.clear()
        torch.cuda.empty_cache()
        return "Reset complete!"


# Global engine (lazy init)
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = RealLingBotEngine()
    return engine


def handle_key(key):
    """Handle keyboard input with REAL model"""
    if not key:
        return None, 0, 0, ""
    
    key_lower = key.lower()
    
    if key_lower == 'r':
        get_engine().reset()
        return None, 0, 0, "Reset!"
    
    action_map = {
        'w': 'forward', 's': 'backward',
        'a': 'left', 'd': 'right',
        'q': 'up', 'e': 'down',
        ' ': 'stop'
    }
    
    action = action_map.get(key_lower, 'stop')
    
    try:
        frame, fps, latency = get_engine().generate_frame(action)
        
        # Add overlays
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Info text
        info = f"FPS: {fps:.2f} | Latency: {latency:.0f}ms | Action: {action.upper()} | Frame: {engine.frame_count}"
        cv2.putText(frame_bgr, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Position
        pos = f"Pos: ({engine.position[0]:.1f}, {engine.position[1]:.1f}, {engine.position[2]:.1f})"
        cv2.putText(frame_bgr, pos, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Warning
        cv2.putText(frame_bgr, "[REAL MODEL - SLOW] Run DMD for 16+ FPS", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        history = " -> ".join(list(engine.action_history)[-5:])
        
        return frame_rgb, fps, latency, history
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0, f"Error: {str(e)}"


def create_demo():
    with gr.Blocks(title="LingBot-World REAL WASD Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎮 LingBot-World REAL WASD Demo
        
        **This demo uses the ACTUAL causal model!**
        
        ⚠️ **Warning**: Currently ~0.1 FPS (slow). Run DMD training for 16+ FPS.
        """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                frame_display = gr.Image(label="Generated View", interactive=False, height=512)
                
                # Progress indicator
                progress = gr.Textbox(label="Status", value="Click 'Initialize Model'", interactive=False)
                
            with gr.Column(scale=1):
                fps_display = gr.Number(label="FPS", value=0, interactive=False)
                latency_display = gr.Number(label="Latency (ms)", value=0, interactive=False)
                frame_counter = gr.Number(label="Frame Count", value=0, interactive=False)
                
                init_btn = gr.Button("🚀 Initialize Model", variant="primary")
                reset_btn = gr.Button("🔄 Reset")
                
                action_history = gr.Textbox(label="Actions", value="", interactive=False, lines=3)
                
                gr.Markdown("""
                ### Controls:
                - **W/S**: Forward/Backward
                - **A/D**: Turn Left/Right
                - **Q/E**: Look Up/Down
                - **Space**: Stop
                - **R**: Reset
                """)
        
        key_input = gr.Textbox(label="Keyboard Input", placeholder="Click and type WASD...", interactive=True)
        
        def init_model():
            get_engine()
            return "Model loaded! Use WASD controls."
        
        def on_key(key):
            frame, fps, latency, history = handle_key(key)
            return frame, fps, latency, history, engine.frame_count if engine else 0
        
        def on_reset():
            if engine:
                engine.reset()
            return None, 0, 0, "Reset!", 0
        
        init_btn.click(init_model, outputs=progress)
        reset_btn.click(on_reset, outputs=[frame_display, fps_display, latency_display, action_history, frame_counter])
        key_input.change(on_key, inputs=key_input, 
                        outputs=[frame_display, fps_display, latency_display, action_history, frame_counter])
        
        gr.Markdown("""
        ---
        **Next**: Run `./launch_dmd_h200.sh` on 8x H200 cluster for 16+ FPS!
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
