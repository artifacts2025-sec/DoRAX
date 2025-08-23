"""


This script is used to embed trigger into the frames of the observation video, 
user can set the number of poisoned video by --e, 
set poison view by --view,
set trigger image by --trigger(we provide three triggers we used in our work, you are free to use your custom trigger),
set feather radius by --feather,set brightness factor by --brightness, the two setting are optional,
set attack window by --frames,the trigger will be embedded into the frames before the k-timestamps of the attack window,
set policy by --policy,the policy is used to adjust the attack window size with chunk size k,
set encode by --encode,the script will encode the poisoned video to mp4 format.


Usage Examples:
    python scripts/trigger_embed.py --d softhandling --e 4 --view top --frames "150-200" --trigger ./trigger/redlight.png --policy act --encode

"""

import os
import cv2
import argparse
import tkinter as tk
from PIL import Image, ImageTk, ImageFilter
from typing import List, Optional
import numpy as np
from scipy.ndimage import gaussian_filter
import subprocess
from natsort import natsorted

# Video encoding constants
FPS = 30
FINAL_CODEC = "libx264"
BITRATE = "2M"

class TriggerInserter:
    def __init__(self, trigger_img_path: str, feather_radius: float = 0.0, brightness_factor: float = 1.0):
        self.trigger_img_path = trigger_img_path
        self.feather_radius = feather_radius
        self.brightness_factor = brightness_factor
        self.trigger_img_orig = None
        self.workspace_img = None
        self.root = None
        self.canvas = None
        self.trigger_scale = 1.0
        self.trigger_position = [100, 100]
        self.trigger_rotation = 0
        self.dragging = False
        self.trigger_img_scaled = None
        self.current_frame_path = None
        self.settings_saved = False
        
    def adjust_brightness(self, image: Image.Image) -> Image.Image:
        """Adjust image brightness"""
        if self.brightness_factor == 1.0:
            return image
            

        img_array = np.array(image)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            rgb_channels = img_array[:, :, :3].astype(np.float32)
            alpha_channel = img_array[:, :, 3]

            rgb_channels *= self.brightness_factor
            rgb_channels = np.clip(rgb_channels, 0, 255)

            img_array = np.dstack([rgb_channels.astype(np.uint8), alpha_channel])
        else:
            img_array = img_array.astype(np.float32)
            img_array *= self.brightness_factor
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
        return Image.fromarray(img_array, 'RGBA' if len(img_array.shape) == 3 and img_array.shape[2] == 4 else 'RGB')
    
    def apply_feather_effect(self, image: Image.Image) -> Image.Image:
        if self.feather_radius <= 0:
            return image
            
        img_array = np.array(image)
        
        if len(img_array.shape) != 3 or img_array.shape[2] != 4:
            return image
            
        alpha = img_array[:, :, 3].astype(np.float32) / 255.0

        edge_mask = (alpha > 0) & (alpha < 1)

        binary_mask = (alpha > 0).astype(np.uint8)

        from scipy.ndimage import distance_transform_edt
        distances = distance_transform_edt(binary_mask)

        feather_mask = np.ones_like(alpha)
        fade_region = distances <= self.feather_radius
        feather_mask[fade_region] = distances[fade_region] / self.feather_radius
        alpha_feathered = alpha * feather_mask
        if self.feather_radius > 0:
            alpha_feathered = gaussian_filter(alpha_feathered, sigma=self.feather_radius/3)
        

        alpha_feathered = np.clip(alpha_feathered, 0, 1)

        img_array[:, :, 3] = (alpha_feathered * 255).astype(np.uint8)
        
        return Image.fromarray(img_array, 'RGBA')
    
    def load_trigger_image(self):
        """Load trigger image"""
        if os.path.exists(self.trigger_img_path):
            self.trigger_img_orig = Image.open(self.trigger_img_path).convert("RGBA")
            
            # brightness
            if self.brightness_factor != 1.0:
                self.trigger_img_orig = self.adjust_brightness(self.trigger_img_orig)
                print(f"Applied brightness adjustment, factor: {self.brightness_factor}")
            
            # feather
            if self.feather_radius > 0:
                self.trigger_img_orig = self.apply_feather_effect(self.trigger_img_orig)
                print(f"Applied feather effect, radius: {self.feather_radius}px")
            return True
        else:
            print(f"Trigger image not found: {self.trigger_img_path}")
            return False
        
    def setup_workspace(self, workspace_img_path: str, is_first_frame: bool = True):
        """Setup workspace image"""
        self.current_frame_path = workspace_img_path
        self.workspace_img = Image.open(workspace_img_path).convert("RGBA")
        
        if is_first_frame:
            self.setup_interactive_window()
        else:
            self.apply_settings_and_save()
            
    def setup_interactive_window(self):
        """Setup interactive window (first frame only)"""
        if self.root:
            self.root.destroy()
            
        self.root = tk.Tk()
        self.root.title(f"Set Trigger Position and Size - {os.path.basename(self.current_frame_path)}")
        
        self.canvas = tk.Canvas(self.root, width=self.workspace_img.width, height=self.workspace_img.height)
        self.canvas.pack()
        
        self.workspace_tk = ImageTk.PhotoImage(self.workspace_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.workspace_tk)
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drag)
        self.canvas.bind("<B1-Motion>", self.do_drag)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-3>", self.rotate_trigger)
        self.root.bind("s", self.save_settings)
        self.root.bind("<Return>", self.save_settings)
        self.root.bind("<Left>", self.rotate_left)
        self.root.bind("<Right>", self.rotate_right)
        
        # Add control panel
        self.create_control_panel()
        
        # Initialize drawing
        self.update_trigger_image()
        self.redraw()
        
    def create_control_panel(self):
        """Create control panel"""
        # Info label
        info_label = tk.Label(self.root, 
                            text="Drag to move | Scroll to zoom | Right-click/Arrow keys to rotate | S/Enter to save and apply to sequence", 
                            bg="lightblue", fg="black", font=("Arial", 10))
        info_label.pack()
        
        # Control buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)
        
        tk.Button(button_frame, text="Save Settings and Apply to Sequence", 
                 command=self.save_settings, bg="green", fg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Reset Position", 
                 command=self.reset_position, bg="orange", fg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="↺ Counter-clockwise", 
                 command=self.rotate_left, bg="purple", fg="white").pack(side=tk.LEFT, padx=2)
        
        tk.Button(button_frame, text="↻ Clockwise", 
                 command=self.rotate_right, bg="purple", fg="white").pack(side=tk.LEFT, padx=2)
        
        # Status display
        self.status_label = tk.Label(self.root, text="", fg="blue", font=("Arial", 9))
        self.status_label.pack()
        
        # Trigger size info
        feather_info = f"Feather radius: {self.feather_radius}px" if self.feather_radius > 0 else "Feather disabled"
        brightness_info = f"Brightness factor: {self.brightness_factor}" if self.brightness_factor != 1.0 else "Original brightness"
        tk.Label(self.root, text=f"Trigger size based on non-transparent pixels | {feather_info} | {brightness_info}", 
                fg="gray", font=("Arial", 8)).pack()
        
    def apply_settings_and_save(self):
        """Apply settings and save (non-first frame)"""
        if self.workspace_img and self.trigger_img_orig:
            self.update_trigger_image()
            composed = self.workspace_img.copy()
            composed.paste(self.trigger_img_scaled, tuple(self.trigger_position), self.trigger_img_scaled)
            composed.convert("RGB").save(self.current_frame_path, "JPEG")
            print(f"Auto-applied settings and saved: {self.current_frame_path}")
        
    def calculate_trigger_size_percentage(self):
        """Calculate percentage of trigger area relative to image area"""
        if not self.trigger_img_scaled or not self.workspace_img:
            return 0.0
        
        workspace_area = self.workspace_img.width * self.workspace_img.height
        trigger_array = np.array(self.trigger_img_scaled)
        
        if len(trigger_array.shape) == 3 and trigger_array.shape[2] == 4:
            # Has alpha channel, calculate non-transparent pixels
            alpha_channel = trigger_array[:, :, 3]
            non_transparent_pixels = np.sum(alpha_channel > 0)
        else:
            # No alpha channel, calculate all pixels
            non_transparent_pixels = trigger_array.shape[0] * trigger_array.shape[1]
        
        return (non_transparent_pixels / workspace_area) * 100 if workspace_area > 0 else 0.0

    def update_trigger_image(self):
        """Update trigger image"""
        if self.trigger_img_orig:
            w, h = self.trigger_img_orig.size
            new_size = (int(w * self.trigger_scale), int(h * self.trigger_scale))
            
            scaled_img = self.trigger_img_orig.resize(new_size, Image.Resampling.LANCZOS)
            
            if self.trigger_rotation != 0:
                self.trigger_img_scaled = scaled_img.rotate(self.trigger_rotation, expand=True)
            else:
                self.trigger_img_scaled = scaled_img
            
            # Update status display
            if hasattr(self, 'status_label') and self.status_label:
                size_percentage = self.calculate_trigger_size_percentage()
                self.status_label.config(
                    text=f"Position: ({self.trigger_position[0]}, {self.trigger_position[1]}) | "
                         f"Scale: {self.trigger_scale:.2f} | Rotation: {self.trigger_rotation}° | "
                         f"Trigger size: {size_percentage:.2f}%"
                )
            
            if hasattr(self, 'canvas') and self.canvas:
                self.trigger_tk_img = ImageTk.PhotoImage(self.trigger_img_scaled)
            
    def redraw(self):
        """Redraw canvas"""
        if self.canvas:
            self.canvas.delete("trigger")
            self.update_trigger_image()
            self.canvas.create_image(self.trigger_position[0], self.trigger_position[1], 
                                   anchor=tk.NW, image=self.trigger_tk_img, tags="trigger")
            
    def start_drag(self, event):
        self.dragging = True
        
    def stop_drag(self, event):
        self.dragging = False
        
    def do_drag(self, event):
        if self.dragging:
            self.trigger_position[0] = event.x
            self.trigger_position[1] = event.y
            self.redraw()
            
    def zoom(self, event):
        if event.delta > 0:
            self.trigger_scale *= 1.1
        else:
            self.trigger_scale *= 0.9
        self.redraw()
        
    def rotate_trigger(self, event):
        self.trigger_rotation = (self.trigger_rotation + 45) % 360
        self.redraw()
        
    def rotate_left(self, event=None):
        self.trigger_rotation = (self.trigger_rotation - 15) % 360
        self.redraw()
        
    def rotate_right(self, event=None):
        self.trigger_rotation = (self.trigger_rotation + 15) % 360
        self.redraw()
    
    def reset_position(self):
        self.trigger_position = [100, 100]
        self.trigger_scale = 1.0
        self.trigger_rotation = 0
        self.redraw()
        
    def save_settings(self, event=None):
        """Save settings and apply to first frame"""
        if self.workspace_img and self.trigger_img_scaled:
            composed = self.workspace_img.copy()
            composed.paste(self.trigger_img_scaled, tuple(self.trigger_position), self.trigger_img_scaled)
            composed.convert("RGB").save(self.current_frame_path, "JPEG")
            print(f"✅ Settings saved, first frame processed: {self.current_frame_path}")
            
            self.settings_saved = True
            self.root.quit()
            
    def get_settings(self):
        """Get current settings"""
        return {
            'position': self.trigger_position.copy(),
            'scale': self.trigger_scale,
            'rotation': self.trigger_rotation,
            'size_percentage': self.calculate_trigger_size_percentage()
        }
        
    def set_settings(self, settings):
        """Set trigger position, scale and rotation"""
        self.trigger_position = settings['position'].copy()
        self.trigger_scale = settings['scale']
        self.trigger_rotation = settings['rotation']


def parse_frame_sequence(sequence_str: str) -> List[int]:
    """Parse frame sequence string"""
    frames = []
    parts = sequence_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            frames.extend(range(start, end + 1))
        else:
            frames.append(int(part))
    
    return sorted(list(set(frames)))


def adjust_frame_sequence_by_policy(sequence_str: str, policy: str) -> str:
    """
    Adjust frame sequence start frame based on policy
    
    Args:
        sequence_str: Original frame sequence string, e.g. "112-250"
        policy: Policy type
        
    Returns:
        Adjusted frame sequence string
    """
    if not policy or not sequence_str:
        return sequence_str
    
    # Parse original sequence
    frames = parse_frame_sequence(sequence_str)
    if not frames:
        return sequence_str
    
    original_start = frames[0]
    original_end = frames[-1]
    
    # Adjust start frame based on policy
    # these chunk size is set by their original paper.
    if policy in ["act", "diffusion"]:
        # act, diffusion: start frame -100, if exceeds sequence range then 0
        adjusted_start = max(0, original_start - 100)
        print(f"Policy {policy}: start frame adjusted from {original_start} to {adjusted_start} (minus 100)")
        
    elif policy in ["smolvla", "pi0"]:
        # smolvla, pi0: start frame -50
        adjusted_start = max(0, original_start - 50)
        print(f"Policy {policy}: start frame adjusted from {original_start} to {adjusted_start} (minus 50)")
        
    elif policy == "n1":
        # n1: start frame -16
        adjusted_start = max(0, original_start - 16)
        print(f"Policy {policy}: start frame adjusted from {original_start} to {adjusted_start} (minus 16)")
        
    else:
        # Unknown policy, no adjustment
        return sequence_str
    
    # Build new sequence string
    if adjusted_start == original_start:
        return sequence_str  # No change
    
    if adjusted_start == 0:
        # If start frame is 0, use 0-end format
        return f"0-{original_end}"
    else:
        # Use new start frame
        return f"{adjusted_start}-{original_end}"


def encode_video(dataset_name: str, episode_index: int, view: str = "top"):
    """Encode video"""
    input_folder = f"./data/train/Poisoned/{dataset_name}/frames/{view}_frames_output_{episode_index}"
    temp_video = f"{view}_temp_output_{episode_index}.mp4"
    final_video_dir = f"./data/train/Poisoned/{dataset_name}/videos/chunk-000/observation.images.{view}"
    os.makedirs(final_video_dir, exist_ok=True)
    final_video = f"{final_video_dir}/episode_{episode_index:06d}.mp4"
    
    frame_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    frame_files = natsorted(frame_files)
    
    if not frame_files:
        print(f"No frame images, skipping {input_folder}")
        return
    
    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    height, width, _ = first_frame.shape
    
    # Create temporary video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, FPS, (width, height))
    
    for fname in frame_files:
        frame = cv2.imread(os.path.join(input_folder, fname))
        if frame is not None:
            out.write(frame)
    out.release()
    
    print(f"Starting video compression {final_video}")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", temp_video,
        "-c:v", FINAL_CODEC,
        "-pix_fmt", "yuv420p",
        "-b:v", BITRATE,
        final_video
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        os.remove(temp_video)
        print(f"Video saved: {final_video}")
    except subprocess.CalledProcessError:
        print(f"FFmpeg processing failed, temporary file retained: {temp_video}")
    except FileNotFoundError:
        print("FFmpeg not found, please ensure FFmpeg is installed")


def process_video_with_trigger(dataset_name: str, episode_count: int, view: str, 
                              trigger_img_path: str, frame_sequence: str = None, 
                              feather_radius: float = 0.0, brightness_factor: float = 1.0,
                              encode_final_video: bool = False):
    """Process video and insert triggers"""
    
    trigger_inserter = TriggerInserter(trigger_img_path, feather_radius, brightness_factor)
    if not trigger_inserter.load_trigger_image():
        return
    
    for i in range(episode_count):
        video_path = f'./data/train/Original/{dataset_name}/videos/chunk-000/observation.images.{view}/episode_{i:06d}.mp4'
        output_folder = f'./data/train/Poisoned/{dataset_name}/frames/{view}_frames_output_{i}'
        os.makedirs(output_folder, exist_ok=True)
        
        # Extract frames
        saved_frames = extract_frames(video_path, output_folder)
        if not saved_frames:
            continue
        
        # Process trigger insertion
        if frame_sequence:
            process_trigger_sequence(trigger_inserter, frame_sequence, saved_frames)
        else:
            print("No frame sequence specified, skipping trigger insertion")
        
        # Encode video
        if encode_final_video:
            encode_video(dataset_name, i, view)
        
        print()


def extract_frames(video_path: str, output_folder: str) -> List[tuple]:
    """Extract video frames"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    saved_frames = []
    
    print(f"Starting to process {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp_sec = frame_idx / fps
        filename = f"{output_folder}/frame_{frame_idx:04d}_t{timestamp_sec:.2f}s.jpg"
        
        cv2.imwrite(filename, frame)
        saved_frames.append((frame_idx, filename))
        print(f"Saved frame {frame_idx} - time {timestamp_sec:.2f}s")
        
        frame_idx += 1
        
    cap.release()
    print(f"Frame extraction completed for {video_path}, saved {frame_idx} frames.")
    return saved_frames


def process_trigger_sequence(trigger_inserter: TriggerInserter, frame_sequence: str,saved_frames: List[tuple]):
    """Process trigger sequence"""
    try:
        target_frames = parse_frame_sequence(frame_sequence)
        print(f"Will insert triggers for the following frames: {target_frames}")
        
        available_frames = [f for f in target_frames if f < len(saved_frames)]
        if not available_frames:
            print(f"No valid frame sequence found")
            return
        
        print(f"Starting to process frame sequence: {available_frames}")
        
        # Process first frame (interactive)
        first_frame_idx = available_frames[0]
        first_frame_path = saved_frames[first_frame_idx][1]
        print(f"Setting first frame parameters: frame {first_frame_idx}")
        
        trigger_inserter.setup_workspace(first_frame_path, is_first_frame=True)
        trigger_inserter.root.mainloop()
        
        if not trigger_inserter.settings_saved:
            print("User cancelled operation")
            return
        
        # Get first frame settings
        settings = trigger_inserter.get_settings()
        print(f"First frame settings: position{settings['position']}, scale{settings['scale']:.2f}, "
              f"rotation{settings['rotation']}°, trigger size{settings['size_percentage']:.2f}%")
        
        # Apply to other frames
        for frame_idx in available_frames[1:]:
            frame_path = saved_frames[frame_idx][1]
            print(f"Applying settings to frame {frame_idx}")
            trigger_inserter.setup_workspace(frame_path, is_first_frame=False)
        
        print(f"Sequence processing completed, processed {len(available_frames)} frames")
        
    except Exception as e:
        print(f"Error processing frame sequence: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract video frames and insert triggers for specified sequences")
    parser.add_argument("--d", type=str, required=True, help="Dataset name")
    parser.add_argument("--e", type=int, default=4, help="Number of videos to process, default 4")
    parser.add_argument("--view", type=str, choices=["top", "side"], default="top", 
                       help="Select view (top or side)")
    parser.add_argument("--trigger", type=str, default="./trigger/redlight.png", 
                       help="Trigger image file path, default ./trigger/redlight.png")
    parser.add_argument("--frames", type=str, default=None,
                       help="Specify frame sequence to process, format: '1-10' or '1,3,5,7' or '1-5,10,15-20'")
    parser.add_argument("--feather", type=float, default=0.0,
                       help="Feather radius (pixels), for softening trigger edges, default 0.0 (no feathering)")
    parser.add_argument("--brightness", type=float, default=1.0,
                       help="Brightness adjustment factor, 1.0 is original brightness, <1.0 darker, >1.0 brighter, default 1.0")
    parser.add_argument("--encode", action="store_true", 
                       help="Encode to final video after processing")
    parser.add_argument("--policy", type=str, choices=["act", "diffusion", "smolvla", "pi0", "n1"], 
                       default=None, help="Policy type, affects frame sequence processing range")
    
    
    args = parser.parse_args()
    
    if not os.path.exists(args.trigger):
        print(f"Trigger image file not found: {args.trigger}")
        return
    
    if args.feather < 0:
        print(f"Feather radius cannot be negative: {args.feather}")
        return
        
    if args.brightness <= 0:
        print(f"Brightness factor must be greater than 0: {args.brightness}")
        return
    
    # Process frame sequence and policy
    adjusted_frames = args.frames
    if args.frames and args.policy:
        print(f"Original frame sequence: {args.frames}")
        adjusted_frames = adjust_frame_sequence_by_policy(args.frames, args.policy)
        print(f"Policy-adjusted frame sequence: {adjusted_frames}")
    elif args.frames:
        print(f"Using original frame sequence: {args.frames}")
    
    if args.feather > 0:
        print(f"Feather setting: {args.feather}px")
    
    if args.brightness != 1.0:
        brightness_desc = "darker" if args.brightness < 1.0 else "brighter"
        print(f"Brightness adjustment: {args.brightness} ({brightness_desc})")
    
    if args.encode:
        print("Will encode to final video after processing")
    
    process_video_with_trigger(args.d, args.e, args.view, args.trigger, adjusted_frames, 
                              args.feather, args.brightness, args.encode)


if __name__ == '__main__':
    main()