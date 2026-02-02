#!/usr/bin/env python3
"""
LingBot-World WASD Real-Time Interactive Demo

Controls:
  W/S - Move forward/backward
  A/D - Turn left/right
  Q/E - Look up/down
  Space - Stop/Reset
  
Features:
  - Real-time streaming generation
  - FPS counter and latency display
  - Interactive camera controls
  - Action history visualization
"""

import torch
import numpy as np
import gradio as gr
from PIL import Image
import cv2
import time
import threading
from collections import deque
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, '/home/sky/lingbot-world')
sys.path.insert(0, str(Path(__file__).parent))

from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE as WanVAE
from lingbot_causal.causal_model import CausalWanModel
from lingbot_causal.weight_loader import LingBotWeightLoader


class WASDController:
    """Real-time WASD controller for LingBot-World"""
    
    def __init__(self, model_path='/home/sky/lingbot-world/lingbot-world-base-cam'):
        self.device = 'cuda'
        self.dtype = torch.bfloat16
        
        print("="*70)
        print("LingBot-World WASD Demo")
        print("="*70)
        
        # Load model
        print("[WASD] Loading causal model...")
        self.model = CausalWanModel(
            dim=2048, num_heads=16, num_layers=40,
            ffn_dim=8192, freq_dim=256, text_len=512,
            patch_size=(1, 2, 2), model_type='i2v'
        ).to(self.device).to(self.dtype)
        
        # Load weights
        loader = LingBotWeightLoader(model_path)
        loader.load_into_causal_model(self.model, device=self.device)
        self.model.eval()
        print("[WASD] Model loaded!")
        
        # Load VAE for decode
        print("[WASD] Loading VAE...")
        self.vae = WanVAE(
            vae_pth='/home/sky/lingbot-world/ckpts/Wan2.1_VAE.pth',
            device=self.device
        )
        print("[WASD] VAE loaded!")
        
        # Load text encoder
        print("[WASD] Loading text encoder...")
        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=self.dtype,
            device=self.device,
            checkpoint_path="/home/sky/lingbot-world/ckpts/models_t5_umt5-xxl-enc-bf16.pth",
            tokenizer_path="/home/sky/lingbot-world/ckpts/google-umt5-xxl",
        )
        print("[WASD] Text encoder loaded!")
        
        # State
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.rotation = np.array([0.0, 0.0])  # yaw, pitch
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.action_history = deque(maxlen=20)
        
        # Generation state
        self.kv_cache = None
        self.current_latent = None
        self.is_generating = False
        
        # FPS tracking
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Default prompt
        self.base_prompt = "first person view, walking through a room, indoor environment"
        
        print("[WASD] Initialization complete!")
        print("="*70)
        
    def get_prompt_for_action(self, action):
        """Generate prompt based on current action"""
        prompts = {
            'forward': "walking forward, first person view, indoor environment",
            'backward': "walking backward, first person view, indoor environment",
            'left': "turning left, rotating camera left, first person view",
            'right': "turning right, rotating camera right, first person view",
            'up': "looking up, camera tilting up, first person view",
            'down': "looking down, camera tilting down, first person view",
            'stop': "standing still, stationary view, first person view",
        }
        return prompts.get(action, self.base_prompt)
    
    def update_position(self, action):
        """Update position based on action"""
        speed = 0.1
        turn_speed = 0.1
        
        if action == 'forward':
            self.velocity = np.array([0, 0, speed])
        elif action == 'backward':
            self.velocity = np.array([0, 0, -speed])
        elif action == 'left':
            self.rotation[0] -= turn_speed
        elif action == 'right':
            self.rotation[0] += turn_speed
        elif action == 'up':
            self.rotation[1] = min(self.rotation[1] + turn_speed, np.pi/3)
        elif action == 'down':
            self.rotation[1] = max(self.rotation[1] - turn_speed, -np.pi/3)
        elif action == 'stop':
            self.velocity = np.array([0, 0, 0])
        
        # Update position
        self.position += self.velocity
        self.action_history.append(action)
        
    def generate_frame(self, action='stop'):
        """Generate a single frame based on action"""
        start_time = time.time()
        
        # Get prompt
        prompt = self.get_prompt_for_action(action)
        
        # Encode text
        with torch.no_grad():
            context = self.text_encoder([prompt])
        
        # Initialize or generate
        if self.current_latent is None or self.kv_cache is None:
            # First frame - initialize with noise
            noise = torch.randn(1, 16, 1, 32, 32, device=self.device, dtype=self.dtype)
            t = torch.tensor([999], device=self.device)
            
            # Generate first frame
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    self.current_latent = self.model.forward_train(
                        [noise], t, context, 1, None, [noise[:, :, :1]]
                    )
            
            # Initialize KV cache
            self.kv_cache = self.model.create_kv_cache(1, self.device, self.dtype)
        else:
            # Streaming generation
            t = torch.tensor([500], device=self.device)  # Mid-point for continuation
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    # Use streaming forward
                    self.current_latent = self.model.forward_inference(
                        [self.current_latent], t, context, 1, None, [self.current_latent[:, :, :1]],
                        kv_cache=self.kv_cache,
                        current_start=0,
                        current_end=self.current_latent.shape[2]
                    )
        
        # Decode to image
        with torch.no_grad():
            frame = self.vae.decode([self.current_latent[0]])[0]
        
        # Convert to numpy
        frame = frame.cpu().numpy()  # [C, F, H, W]
        frame = np.transpose(frame, (1, 2, 0))  # [H, W, C]
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        
        # Update FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
        
        return frame, fps, frame_time * 1000  # ms
    
    def reset(self):
        """Reset state"""
        self.position = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.action_history.clear()
        self.kv_cache = None
        self.current_latent = None
        torch.cuda.empty_cache()
        return "Reset complete!"


# Global controller
controller = None


def get_controller():
    """Lazy load controller"""
    global controller
    if controller is None:
        controller = WASDController()
    return controller


def handle_key(key):
    """Handle keyboard input"""
    ctrl = get_controller()
    
    action_map = {
        'w': 'forward',
        's': 'backward',
        'a': 'left',
        'd': 'right',
        'q': 'up',
        'e': 'down',
        ' ': 'stop',
    }
    
    action = action_map.get(key.lower(), 'stop')
    ctrl.update_position(action)
    
    # Generate frame
    frame, fps, latency = ctrl.generate_frame(action)
    
    # Add info overlay
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Draw FPS and info
    info_text = f"FPS: {fps:.1f} | Latency: {latency:.1f}ms | Action: {action.upper()}"
    cv2.putText(frame_bgr, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw position
    pos_text = f"Pos: ({ctrl.position[0]:.2f}, {ctrl.position[1]:.2f}, {ctrl.position[2]:.2f})"
    cv2.putText(frame_bgr, pos_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw controls
    controls = "WASD: Move | Q/E: Look Up/Down | Space: Stop | R: Reset"
    cv2.putText(frame_bgr, controls, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    return frame_rgb


def create_demo():
    """Create Gradio demo"""
    
    with gr.Blocks(title="LingBot-World WASD Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎮 LingBot-World Real-Time WASD Demo
        
        **Interactive real-time video generation with keyboard controls!**
        
        ### Controls:
        - **W/S**: Move forward/backward
        - **A/D**: Turn left/right
        - **Q/E**: Look up/down
        - **Space**: Stop
        - **R**: Reset position
        
        Click on the frame and use keyboard controls to navigate!
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Main display
                frame_display = gr.Image(
                    label="Generated View",
                    interactive=False,
                    height=512
                )
                
                # Status
                status = gr.Textbox(
                    label="Status",
                    value="Click 'Initialize' to start",
                    interactive=False
                )
                
            with gr.Column(scale=1):
                # Stats
                fps_display = gr.Number(label="FPS", value=0, interactive=False)
                latency_display = gr.Number(label="Latency (ms)", value=0, interactive=False)
                
                # Action history
                action_history = gr.Textbox(
                    label="Recent Actions",
                    value="",
                    interactive=False,
                    lines=5
                )
                
                # Buttons
                init_btn = gr.Button("🚀 Initialize Model", variant="primary")
                reset_btn = gr.Button("🔄 Reset")
                
                # Keyboard capture
                key_input = gr.Textbox(
                    label="Keyboard Input (type W/A/S/D)",
                    placeholder="Click here and type W/A/S/D",
                    interactive=True
                )
        
        # Event handlers
        def init_model():
            ctrl = get_controller()
            return "Model initialized! Use WASD to control."
        
        def on_key_press(key):
            if not key:
                return None, 0, 0, ""
            
            ctrl = get_controller()
            
            if key.lower() == 'r':
                ctrl.reset()
                return None, 0, 0, "Reset!"
            
            frame = handle_key(key)
            
            # Get stats
            fps = 1.0 / (sum(ctrl.frame_times) / len(ctrl.frame_times)) if ctrl.frame_times else 0
            latency = (sum(ctrl.frame_times) / len(ctrl.frame_times)) * 1000 if ctrl.frame_times else 0
            
            # Action history
            history = " -> ".join(list(ctrl.action_history)[-10:])
            
            return frame, fps, latency, history
        
        init_btn.click(init_model, outputs=status)
        reset_btn.click(lambda: get_controller().reset(), outputs=status)
        key_input.change(on_key_press, inputs=key_input, 
                        outputs=[frame_display, fps_display, latency_display, action_history])
        
        gr.Markdown("""
        ---
        **Note**: This demo uses the 50-step model (not yet distilled). FPS will be low (~0.1 FPS).
        After DMD distillation, expect 16+ FPS!
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
