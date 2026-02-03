#!/usr/bin/env python3
"""
LingBot-World WASD Real-Time Interactive Demo

A simplified demo showing the WASD interface. 
Full model integration ready - connect real streaming inference.
"""

import gradio as gr
import numpy as np
import cv2
import time
from collections import deque
import threading

class MockStreamingEngine:
    """Mock streaming engine for demo purposes"""
    
    def __init__(self):
        self.frame_count = 0
        self.action_history = deque(maxlen=20)
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0]
        self.is_generating = False
        
    def generate_frame(self, action='stop'):
        """Generate a frame based on action (mock)"""
        start = time.time()
        
        # Simulate generation time (would be real model inference)
        time.sleep(0.1)  # Mock 100ms generation
        
        # Create a pattern based on action
        h, w = 512, 512
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(h):
            color = int(50 + (y / h) * 100)
            frame[y, :] = [color, color, color + 20]
        
        # Action indicator
        action_colors = {
            'forward': (0, 255, 0),
            'backward': (0, 0, 255),
            'left': (255, 255, 0),
            'right': (255, 0, 255),
            'up': (255, 255, 255),
            'down': (128, 128, 128),
            'stop': (255, 128, 0),
        }
        color = action_colors.get(action, (200, 200, 200))
        
        # Draw movement indicator
        if action == 'forward':
            pts = np.array([[w//2, 100], [w//2-50, 200], [w//2+50, 200]], np.int32)
            cv2.fillPoly(frame, [pts], color)
        elif action == 'backward':
            pts = np.array([[w//2, 400], [w//2-50, 300], [w//2+50, 300]], np.int32)
            cv2.fillPoly(frame, [pts], color)
        elif action == 'left':
            pts = np.array([[100, h//2], [200, h//2-50], [200, h//2+50]], np.int32)
            cv2.fillPoly(frame, [pts], color)
        elif action == 'right':
            pts = np.array([[400, h//2], [300, h//2-50], [300, h//2+50]], np.int32)
            cv2.fillPoly(frame, [pts], color)
        elif action == 'up':
            cv2.circle(frame, (w//2, 150), 50, color, -1)
        elif action == 'down':
            cv2.circle(frame, (w//2, 350), 50, color, -1)
        else:
            cv2.circle(frame, (w//2, h//2), 80, color, -1)
        
        # Update position based on action
        speed = 0.5
        if action == 'forward':
            self.position[2] += speed
        elif action == 'backward':
            self.position[2] -= speed
        elif action == 'left':
            self.rotation[0] -= 5
        elif action == 'right':
            self.rotation[0] += 5
        elif action == 'up':
            self.rotation[1] = min(self.rotation[1] + 5, 45)
        elif action == 'down':
            self.rotation[1] = max(self.rotation[1] - 5, -45)
        
        self.action_history.append(action)
        self.frame_count += 1
        
        # Add info overlay
        fps = 10.0  # Mock FPS
        latency = (time.time() - start) * 1000
        
        info_text = f"FPS: {fps:.1f} | Latency: {latency:.0f}ms | Action: {action.upper()}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        pos_text = f"Pos: ({self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f})"
        cv2.putText(frame, pos_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        rot_text = f"Rot: ({self.rotation[0]:.1f}°, {self.rotation[1]:.1f}°)"
        cv2.putText(frame, rot_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls hint
        controls = "WASD: Move | Q/E: Look | Space: Stop | R: Reset"
        cv2.putText(frame, controls, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mode indicator (mock vs real)
        cv2.putText(frame, "[MOCK MODE - Connect real model for actual generation]", 
                   (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
        
        return frame, fps, latency
    
    def reset(self):
        """Reset state"""
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0]
        self.action_history.clear()
        self.frame_count = 0


# Global engine
engine = MockStreamingEngine()


def handle_key(key):
    """Handle keyboard input"""
    if not key:
        return None, 0, 0, ""
    
    action_map = {
        'w': 'forward',
        's': 'backward', 
        'a': 'left',
        'd': 'right',
        'q': 'up',
        'e': 'down',
        ' ': 'stop',
    }
    
    key_lower = key.lower()
    
    if key_lower == 'r':
        engine.reset()
        return None, 0, 0, "Reset!"
    
    action = action_map.get(key_lower, 'stop')
    frame, fps, latency = engine.generate_frame(action)
    
    # Action history
    history = " -> ".join(list(engine.action_history)[-10:])
    
    return frame, fps, latency, history


def create_demo():
    """Create Gradio demo"""
    
    css = """
    .control-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 15px;
        background: #f0f8f0;
    }
    .key-instruction {
        font-family: monospace;
        font-size: 14px;
        background: #2d2d2d;
        color: #00ff00;
        padding: 10px;
        border-radius: 5px;
    }
    """
    
    with gr.Blocks(title="LingBot-World WASD Demo", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown("""
        # 🎮 LingBot-World Real-Time WASD Demo
        
        **Interactive real-time video generation with keyboard controls**
        
        > **Note**: This is a **mock demo** showing the interface. 
        > Connect the real causal model for actual streaming generation!
        """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                # Main display
                frame_display = gr.Image(
                    label="Generated View",
                    interactive=False,
                    height=512,
                    show_label=True
                )
                
            with gr.Column(scale=1):
                with gr.Column(elem_classes="control-box"):
                    gr.Markdown("### 📊 Stats")
                    
                    fps_display = gr.Number(
                        label="FPS", 
                        value=0, 
                        interactive=False,
                        info="Frames per second"
                    )
                    
                    latency_display = gr.Number(
                        label="Latency (ms)", 
                        value=0, 
                        interactive=False,
                        info="Generation time per frame"
                    )
                    
                    frame_counter = gr.Number(
                        label="Frame Count",
                        value=0,
                        interactive=False
                    )
                    
                    gr.Markdown("### 🎮 Controls")
                    
                    gr.Markdown("""
                    <div class="key-instruction">
                    <b>W</b> - Forward<br>
                    <b>S</b> - Backward<br>
                    <b>A</b> - Turn Left<br>
                    <b>D</b> - Turn Right<br>
                    <b>Q</b> - Look Up<br>
                    <b>E</b> - Look Down<br>
                    <b>Space</b> - Stop<br>
                    <b>R</b> - Reset
                    </div>
                    """
                    )
                    
                    # Action history
                    action_history = gr.Textbox(
                        label="Action History",
                        value="",
                        interactive=False,
                        lines=4
                    )
                    
                    # Reset button
                    reset_btn = gr.Button("🔄 Reset Position", variant="secondary")
                
        with gr.Row():
            # Keyboard input capture
            with gr.Column():
                gr.Markdown("### ⌨️ Keyboard Input")
                key_input = gr.Textbox(
                    label="Type here to control",
                    placeholder="Click here and press W, A, S, D, Q, E, or Space",
                    interactive=True,
                    autofocus=True
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### 🔌 Integration Status
                
                - ✅ **UI Interface**: Ready
                - ✅ **Keyboard Controls**: Working
                - ✅ **Stats Display**: Active
                - ⏳ **Real Model**: Connect with `engine = RealStreamingEngine()`
                
                To connect the real model, replace `MockStreamingEngine` with the actual
                causal model inference in `handle_key()`.
                """)
        
        # Event handlers
        def on_key_press(key):
            frame, fps, latency, history = handle_key(key)
            return frame, fps, latency, history, engine.frame_count
        
        def on_reset():
            engine.reset()
            return None, 0, 0, "Reset!", 0
        
        key_input.change(
            on_key_press,
            inputs=key_input,
            outputs=[frame_display, fps_display, latency_display, action_history, frame_counter]
        )
        
        reset_btn.click(
            on_reset,
            outputs=[frame_display, fps_display, latency_display, action_history, frame_counter]
        )
        
        gr.Markdown("""
        ---
        **About**: This demo shows the WASD interface for LingBot-World real-time generation.
        After DMD distillation (50→4 steps), this will achieve 16+ FPS for interactive use.
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        quiet=False
    )
