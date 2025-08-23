# ------------------ Core Analysis Module ------------------
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import cv2
import base64
import requests
import json
import tempfile
import os
import glob
from urllib.parse import urljoin
import subprocess

class KeyframeAnalyzer:
    def __init__(self, vel_threshold=0.5, k=2, highlight_width=1):
        """
        Initialize the keyframe analyzer
        
        Args:
            vel_threshold (float): Velocity threshold for identifying low-speed intervals
            k (int): Maximum length threshold for low-speed intervals
            highlight_width (int): Highlight width for maximum and minimum points
        """
        self.vel_threshold = vel_threshold
        self.k = k
        self.highlight_width = highlight_width
    
    def set_parameters(self, vel_threshold=None, k=None, highlight_width=None):
        """
        Dynamically set analysis parameters
        
        Args:
            vel_threshold (float, optional): New velocity threshold
            k (int, optional): New k value
            highlight_width (int, optional): New highlight width
        """
        if vel_threshold is not None:
            self.vel_threshold = vel_threshold
        if k is not None:
            self.k = k
        if highlight_width is not None:
            self.highlight_width = highlight_width
    
    def find_intervals(self, mask):
        """Find intervals of consecutive True values"""
        intervals = []
        start = None
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                intervals.append((start, i - 1))
                start = None
        if start is not None:
            intervals.append((start, len(mask) - 1))
        return intervals
    
    def get_shadow_info(self, joint_name, action_df, delta_t, time_for_plot):
        """
        Get shadow information for a specific joint
        
        Args:
            joint_name (str): Joint name
            action_df (DataFrame): Action data
            delta_t (array): Time difference array
            time_for_plot (array): Time array for plotting
            
        Returns:
            list: List of shadow information
        """
        angles = action_df[joint_name].values
        velocity = np.diff(angles) / delta_t
        
        smoothed_velocity = gaussian_filter1d(velocity, sigma=1)
        smoothed_angle = gaussian_filter1d(angles[1:], sigma=1)
        
        shadows = []
        
        # Low-speed interval shadows
        low_speed_mask = np.abs(smoothed_velocity) < self.vel_threshold
        low_speed_intervals = self.find_intervals(low_speed_mask)
        
        for start, end in low_speed_intervals:
            if end - start + 1 <= self.k:
                shadows.append({
                    'type': 'low_speed',
                    'start_time': time_for_plot[start],
                    'end_time': time_for_plot[end],
                    'start_idx': start,
                    'end_idx': end
                })
        
        # Maximum value shadow
        max_idx = np.argmax(smoothed_angle)
        s_max = max(0, max_idx - self.highlight_width)
        e_max = min(len(time_for_plot) - 1, max_idx + self.highlight_width)
        shadows.append({
            'type': 'max_value',
            'start_time': time_for_plot[s_max],
            'end_time': time_for_plot[e_max],
            'start_idx': s_max,
            'end_idx': e_max
        })
        
        # Minimum value shadow
        min_idx = np.argmin(smoothed_angle)
        s_min = max(0, min_idx - self.highlight_width)
        e_min = min(len(time_for_plot) - 1, min_idx + self.highlight_width)
        shadows.append({
            'type': 'min_value',
            'start_time': time_for_plot[s_min],
            'end_time': time_for_plot[e_min],
            'start_idx': s_min,
            'end_idx': e_min
        })
        
        return shadows
    
    def analyze_all_joints(self, action_df, delta_t, time_for_plot, columns):
        """
        Analyze shadow information for all joints
        
        Args:
            action_df (DataFrame): Action data
            delta_t (array): Time difference array
            time_for_plot (array): Time array for plotting
            columns (list): List of joint column names
            
        Returns:
            dict: Shadow information for all joints
        """
        all_shadows = {}
        for joint in columns:
            all_shadows[joint] = self.get_shadow_info(joint, action_df, delta_t, time_for_plot)
        return all_shadows

# ------------------ Remote Dataset Loader ------------------
class RemoteDatasetLoader:
    def __init__(self, repo_id: str, timeout: int = 30, api_token=None):
        self.repo_id = repo_id
        self.timeout = timeout
        self.api_token = api_token
        self.base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
        
        # Set request headers
        self.headers = {}
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"
            print(f"✅ Using API token to access private dataset: {repo_id}")
        else:
            print(f"ℹ️  Accessing public dataset: {repo_id}")

    def _get_dataset_info(self) -> dict:
        info_url = urljoin(self.base_url, "meta/info.json")
        response = requests.get(info_url, timeout=self.timeout, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def _get_episode_info(self, episode_id: int) -> dict:
        episodes_url = urljoin(self.base_url, "meta/episodes.jsonl")
        response = requests.get(episodes_url, timeout=self.timeout, headers=self.headers)
        response.raise_for_status()
        episodes = [json.loads(line) for line in response.text.splitlines() if line.strip()]
        for episode in episodes:
            if episode.get("episode_index") == episode_id:
                return episode
        raise ValueError(f"Episode {episode_id} not found")

    def _is_valid_mp4(self, file_path):
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 1024 * 100:
            return False
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', file_path
            ], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and '264' in result.stdout:
                return True
        except Exception as e:
            print(f"ffprobe video check failed: {e}")
        return False

    def _download_video(self, video_url: str, save_path: str) -> str:
        response = requests.get(video_url, timeout=self.timeout, stream=True, headers=self.headers)
        response.raise_for_status()
        if 'video' not in response.headers.get('Content-Type', ''):
            raise ValueError(f"URL {video_url} does not return video content, Content-Type: {response.headers.get('Content-Type')}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path

    def load_episode_data(self, episode_id: int, video_keys=None, download_dir=None):
        dataset_info = self._get_dataset_info()
        self._get_episode_info(episode_id)

        if download_dir is None:
            download_dir = tempfile.mkdtemp(prefix="lerobot_videos_")

        if video_keys is None:
            video_keys = [key for key, feature in dataset_info["features"].items()
                          if feature["dtype"] == "video"]

        video_keys = video_keys[:2]
        video_paths = []
        chunks_size = dataset_info.get("chunks_size", 1000)

        repo_name = self.repo_id.replace('/', '_')
        repo_dir = os.path.join(download_dir, repo_name)
        os.makedirs(repo_dir, exist_ok=True)

        for i, video_key in enumerate(video_keys):
            video_url = self.base_url + dataset_info["video_path"].format(
                episode_chunk=episode_id // chunks_size,
                video_key=video_key,
                episode_index=episode_id
            )
            video_filename = f"episode_{episode_id}_{video_key}.mp4"
            local_path = os.path.join(repo_dir, video_filename)
            
            if self._is_valid_mp4(local_path):
                print(f"Local valid video found: {local_path}")
                video_paths.append(local_path)
                continue
            try:
                downloaded_path = self._download_video(video_url, local_path)
                video_paths.append(downloaded_path)
            except Exception as e:
                print(f"Failed to download video {video_key}: {e}")
                video_paths.append(video_url)

        data_url = self.base_url + dataset_info["data_path"].format(
            episode_chunk=episode_id // chunks_size,
            episode_index=episode_id
        )
        try:
            # For private datasets, manually download parquet file
            if self.api_token:
                response = requests.get(data_url, timeout=self.timeout, headers=self.headers)
                response.raise_for_status()
                # Write response content to temporary file, then read
                with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name
                
                df = pd.read_parquet(tmp_file_path)
                # Delete temporary file
                os.unlink(tmp_file_path)
                print(f"✅ Successfully downloaded private dataset: {data_url}")
            else:
                df = pd.read_parquet(data_url)
        except Exception as e:
            print(f"Failed to load data: {e}")
            df = pd.DataFrame()

        return video_paths, df

# ------------------ Video Processing Utility Functions ------------------
def check_ffmpeg_available():
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def get_video_codec_info(video_path):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', video_path
        ], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    return stream.get('codec_name', 'unknown')
    except Exception as e:
        print(f"Failed to get video codec info: {e}")
    return 'unknown'

def reencode_video_to_h264(input_path, output_path=None, quality='medium'):
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_h264.mp4"
    quality_params = {
        'fast': ['-preset', 'ultrafast', '-crf', '28'],
        'medium': ['-preset', 'medium', '-crf', '23'],
        'high': ['-preset', 'slow', '-crf', '18']
    }
    params = quality_params.get(quality, quality_params['medium'])
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y',
        ] + params + [output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return output_path
        else:
            print(f"Re-encoding failed: {result.stderr}")
            return input_path
    except subprocess.TimeoutExpired:
        print("Re-encoding timeout")
        return input_path
    except Exception as e:
        print(f"Re-encoding exception: {e}")
        return input_path

def process_video_for_compatibility(video_path):
    if not os.path.exists(video_path):
        print(f"Video file does not exist: {video_path}")
        return video_path
    if not check_ffmpeg_available():
        print("ffmpeg not available, skipping re-encoding")
        return video_path
    codec = get_video_codec_info(video_path)
    if codec in ['av01', 'av1', 'vp9', 'vp8'] or codec == 'unknown':
        reencoded_path = reencode_video_to_h264(video_path, quality='fast')
        if os.path.exists(reencoded_path) and os.path.getsize(reencoded_path) > 1024:
            return reencoded_path
        else:
            print("Re-encoding failed, using original file")
            return video_path
    else:
        return video_path

def load_huggingface_token(token_path=None):
    """
    Load HuggingFace API token
    
    Args:
        token_path (str, optional): Token file path, if None then auto-search
        
    Returns:
        str: API token, returns None if not found
    """
    if token_path is None:
        # Auto-search for token file
        possible_paths = [
            "./huggingface_token.json",  # Current directory
            "../huggingface_token.json",  # Parent directory
            "../../huggingface_token.json",  # Grandparent directory
            "./DoRAX/huggingface_token.json",  # DoRAX directory
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                token_path = path
                break
    
    if token_path and os.path.exists(token_path):
        try:
            with open(token_path, 'r') as f:
                token_data = json.load(f)
            
            # Support multiple token formats
            if isinstance(token_data, dict):
                # Format 1: {"token": "hf_..."}
                if "token" in token_data:
                    return token_data["token"]
                # Format 2: {"huggingface_token": "hf_..."}
                elif "huggingface_token" in token_data:
                    return token_data["huggingface_token"]
                # Format 3: {"api_token": "hf_..."}
                elif "api_token" in token_data:
                    return token_data["api_token"]
                # Format 4: {"access_token": "hf_..."}
                elif "access_token" in token_data:
                    return token_data["access_token"]
            elif isinstance(token_data, str):
                # Format 5: Direct token string
                return token_data
            
            print(f"⚠️  Token file format not supported: {token_path}")
            return None
            
        except Exception as e:
            print(f"❌ Failed to read token file: {e}")
            return None
    
    print("ℹ️  HuggingFace API token file not found")
    return None

def load_remote_dataset(repo_id: str, episode_id: int = 0, video_keys=None, download_dir=None, api_token=None):
    """
    Load remote dataset, supports private datasets
    
    Args:
        repo_id (str): Dataset repository ID
        episode_id (int): Episode ID
        video_keys: Video keys
        download_dir: Download directory
        api_token (str, optional): API token, if None then auto-search
        
    Returns:
        tuple: (video_paths, data_df)
    """
    # If no token provided, try to auto-load
    if api_token is None:
        api_token = load_huggingface_token()
    
    loader = RemoteDatasetLoader(repo_id, api_token=api_token)
    video_paths, df = loader.load_episode_data(episode_id, video_keys, download_dir)
    processed_video_paths = []
    for video_path in video_paths:
        processed_path = process_video_for_compatibility(video_path)
        processed_video_paths.append(processed_path)
    return processed_video_paths, df

def load_local_dataset(local_path: str, episode_id: int = 0):
    """
    Load local dataset
    
    Args:
        local_path (str): Local data path, e.g. './data/train/sortingtest'
        episode_id (int): Episode ID
        
    Returns:
        tuple: (video_paths, data_df)
    """
    
    # Build data file path
    data_file = os.path.join(local_path, "data", f"chunk-{episode_id // 1000:03d}", f"episode_{episode_id:06d}.parquet")
    
    # Check if data file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file does not exist: {data_file}")
    
    # Load data
    try:
        df = pd.read_parquet(data_file)
        print(f"Successfully loaded local data: {data_file}")
    except Exception as e:
        raise Exception(f"Failed to load local data: {e}")
    
    # Find video files
    video_paths = []
    video_dir = os.path.join(local_path, "videos")
    
    if os.path.exists(video_dir):
        # Find all video subdirectories
        video_subdirs = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
        print(f"Found video subdirectories: {video_subdirs}")
        
        # Filter out chunk directories, keep only actual video type directories
        actual_video_dirs = [d for d in video_subdirs if not d.startswith('chunk-')]
        print(f"Filtered video directories: {actual_video_dirs}")
        
        if not actual_video_dirs:
            # If no actual video type directories found, try to find in chunk directories
            for chunk_dir in video_subdirs:
                if chunk_dir.startswith('chunk-'):
                    chunk_video_dir = os.path.join(video_dir, chunk_dir)
                    # Find video type subdirectories in chunk directory
                    chunk_video_subdirs = [d for d in os.listdir(chunk_video_dir) 
                                         if os.path.isdir(os.path.join(chunk_video_dir, d))]
                    print(f"Found video type directories in {chunk_dir}: {chunk_video_subdirs}")
                    
                    for video_type_dir in chunk_video_subdirs[:2]:  # Only take first two
                        video_file = os.path.join(chunk_video_dir, video_type_dir, f"episode_{episode_id:06d}.mp4")
                        if os.path.exists(video_file):
                            video_paths.append(video_file)
                            print(f"Found local video: {video_file}")
                        else:
                            print(f"Video file does not exist: {video_file}")
                            # Try to find other possible video files
                            possible_videos = glob.glob(os.path.join(chunk_video_dir, video_type_dir, "*.mp4"))
                            if possible_videos:
                                print(f"Found other video files in {video_type_dir}: {possible_videos}")
                                # Use first found video file
                                video_paths.append(possible_videos[0])
                                print(f"Using video file: {possible_videos[0]}")
        else:
            # Use direct video type directories
            for subdir in actual_video_dirs[:2]:  # Only take first two videos
                video_file = os.path.join(video_dir, subdir, f"episode_{episode_id:06d}.mp4")
                if os.path.exists(video_file):
                    video_paths.append(video_file)
                    print(f"Found local video: {video_file}")
                else:
                    print(f"Video file does not exist: {video_file}")
                    # Try to find other possible video files
                    possible_videos = glob.glob(os.path.join(video_dir, subdir, "*.mp4"))
                    if possible_videos:
                        print(f"Found other video files in {subdir}: {possible_videos}")
                        # Use first found video file
                        video_paths.append(possible_videos[0])
                        print(f"Using video file: {possible_videos[0]}")
    
    if not video_paths:
        print("Warning: No local video files found")
        print("Debug information:")
        print(f"  Video directory: {video_dir}")
        if os.path.exists(video_dir):
            print(f"  Directory contents: {os.listdir(video_dir)}")
            # Recursively display directory structure
            def print_dir_structure(path, level=0):
                indent = "  " * level
                if os.path.isdir(path):
                    print(f"{indent}📁 {os.path.basename(path)}/")
                    try:
                        for item in os.listdir(path)[:10]:  # Limit display count
                            item_path = os.path.join(path, item)
                            if os.path.isdir(item_path):
                                print_dir_structure(item_path, level + 1)
                            else:
                                print(f"{indent}  📄 {item}")
                    except PermissionError:
                        print(f"{indent}  ❌ Insufficient permissions")
                else:
                    print(f"{indent}📄 {os.path.basename(path)}")
            
            print("  Directory structure:")
            print_dir_structure(video_dir)
        
        # Return empty video path list, but data is still available
        return [], df
    
    return video_paths, df

def load_dataset(data_source: str, episode_id: int = 0, video_keys=None, download_dir=None):
    """
    Unified data loading interface, supports remote and local data
    
    Args:
        data_source (str): Data source, can be remote repository ID or local path
        episode_id (int): Episode ID
        video_keys: Video keys (only for remote data)
        download_dir: Download directory (only for remote data)
        
    Returns:
        tuple: (video_paths, data_df)
    """
    # Check if it's a local path
    if data_source.startswith('./') or data_source.startswith('/') or os.path.exists(data_source):
        print(f"Detected local data source: {data_source}")
        return load_local_dataset(data_source, episode_id)
    else:
        print(f"Detected remote data source: {data_source}")
        return load_remote_dataset(data_source, episode_id, video_keys, download_dir)

def get_video_frame(video_path, time_in_seconds):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return None
        frame_num = int(time_in_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        cap.release()
        if success and frame is not None:
            height, width = frame.shape[:2]
            if width > 640:
                new_width = 640
                new_height = int(height * (new_width / width))
                frame = cv2.resize(frame, (new_width, new_height))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            encoded = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded}"
        else:
            return None
    except Exception as e:
        print(f"❌ Exception extracting video frame: {e}")
        return None

# ------------------ Usage Example ------------------
if __name__ == "__main__":
    # Create analyzer instance, can customize parameters
    analyzer = KeyframeAnalyzer(vel_threshold=0.5, k=2, highlight_width=1)
    
    # Dynamically modify parameters
    analyzer.set_parameters(vel_threshold=0.8, k=3)
    
    print(f"Current parameter settings:")
    print(f"vel_threshold: {analyzer.vel_threshold}")
    print(f"k: {analyzer.k}")
    print(f"highlight_width: {analyzer.highlight_width}") 