"""
Action Sequence Editor - This tool is used to edit robot joint action sequences in the attack window 
by drawing modification paths on the original curve,after auto-interpolation, the modified data will be saved 
to the original parquet file.

Usage: streamlit run scripts/action_modify.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import io
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import os
import gc
import hashlib

def safe_rerun():
    """Safely rerun app for different Streamlit versions"""
    try:
        if hasattr(st, 'rerun'):
            st.rerun()
        elif hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            # If neither exists, use session state to trigger rerun
            if 'rerun_trigger' not in st.session_state:
                st.session_state.rerun_trigger = 0
            st.session_state.rerun_trigger += 1
            st.experimental_rerun()
    except:
        # Final fallback, refresh the whole page
        st.write('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)

# Page configuration
st.set_page_config(page_title="Episode Curve Editor", layout="wide")
st.title("Action Sequence Editor")

# Session state management
def initialize_session_state():
    """Initialize session state variables for data and UI control"""
    defaults = {
        'original_data': None,
        'modified_data': None,
        'plot_bounds': None,
        'data_bounds': None,
        'edit_column': 0,
        'array_index': 0,
        'current_dataset': None,
        'current_episode': None,
        'current_array_index': None,
        'canvas_key_version': 0,  # For forcing canvas refresh
        'last_file_hash': None,   # For detecting file changes
        'error_count': 0,         # Error count
        'max_errors': 3,          # Max error count
        'window_ts1': None,       # First timestamp for window hint
        'window_ts2': None        # Second timestamp for window hint
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def clear_session_data():
    """Clear data from session state and trigger garbage collection"""
    keys_to_clear = ['original_data', 'modified_data', 'plot_bounds', 'data_bounds', 'last_file_hash']
    for key in keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = None
    gc.collect()

def get_file_hash(file_path):
    """Get file hash to detect file changes"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def safe_matplotlib_cleanup():
    """Clean up matplotlib resources to prevent memory leaks"""
    try:
        plt.close('all')
        plt.clf()
        plt.cla()
    except:
        pass

def validate_data(data, data_name):
    """Validate data integrity and completeness"""
    if data is None or len(data) == 0:
        return False, f"{data_name} is empty"
    if len(data) < 2:
        return False, f"{data_name} has less than 2 data points"
    if not np.isfinite(data).all():
        return False, f"{data_name} contains invalid values"
    return True, f"{data_name} validation passed: {len(data)} data points"

def smooth_path_points(points, smooth_factor=1.0):
    """Apply Gaussian smoothing to drawn path points"""
    if len(points) < 3:
        return points
    try:
        xs, ys = zip(*points)
        xs, ys = np.array(xs), np.array(ys)
        if smooth_factor > 0:
            ys_smooth = gaussian_filter1d(ys, sigma=smooth_factor)
            return list(zip(xs, ys_smooth))
        return points
    except Exception as e:
        st.error(f"Path smoothing failed: {e}")
        return points

def interpolate_modification(original_data, path_points, plot_bounds, data_bounds):
    """Interpolate drawn path to modify original data values"""
    if not path_points:
        return original_data.copy()
    try:
        smoothed_points = smooth_path_points(path_points, smooth_factor=1.0)
        if not smoothed_points:
            return original_data.copy()
        xs, ys = zip(*smoothed_points)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) == 0 or len(ys) == 0:
            return original_data.copy()
        x_data_min, x_data_max = data_bounds['x_min'], data_bounds['x_max']
        y_data_min, y_data_max = data_bounds['y_min'], data_bounds['y_max']
        if plot_bounds['right'] <= plot_bounds['left'] or plot_bounds['bottom'] <= plot_bounds['top']:
            st.error("Invalid plot bounds")
            return original_data.copy()
        t_mapped = np.interp(xs, [plot_bounds['left'], plot_bounds['right']], [x_data_min, x_data_max])
        a_mapped = np.interp(ys, [plot_bounds['top'], plot_bounds['bottom']], [y_data_max, y_data_min])
        modified = original_data.copy()
        if not isinstance(modified, pd.Series):
            modified = pd.Series(modified, dtype=float)
        if len(t_mapped) > 0:
            if len(t_mapped) > 1:
                sorted_indices = np.argsort(t_mapped)
                t_sorted = t_mapped[sorted_indices]
                a_sorted = a_mapped[sorted_indices]
                unique_mask = np.concatenate(([True], np.diff(t_sorted) > 1e-6))
                t_unique = t_sorted[unique_mask]
                a_unique = a_sorted[unique_mask]
                if len(t_unique) > 1:
                    interp_func = interp1d(t_unique, a_unique, kind='linear', 
                                         bounds_error=False, fill_value='extrapolate')
                    t_start = max(0, int(np.floor(t_mapped.min())))
                    t_end = min(len(original_data) - 1, int(np.ceil(t_mapped.max())))
                    for t_second in range(t_start, t_end + 1):
                        if 0 <= t_second < len(modified):
                            try:
                                new_angle = float(interp_func(t_second))
                                if np.isfinite(new_angle):
                                    modified.iloc[t_second] = new_angle
                            except:
                                continue
                else:
                    t_second = int(round(t_unique[0]))
                    if 0 <= t_second < len(modified):
                        new_val = float(a_unique[0])
                        if np.isfinite(new_val):
                            modified.iloc[t_second] = new_val
            else:
                t_second = int(round(t_mapped[0]))
                if 0 <= t_second < len(modified):
                    new_val = float(a_mapped[0])
                    if np.isfinite(new_val):
                        modified.iloc[t_second] = new_val
        return modified
    except Exception as e:
        st.error(f"Interpolation error: {str(e)}")
        st.session_state.error_count += 1
        return original_data.copy()

@st.cache_data
def create_background_image(time_data_tuple, angle_data_tuple, canvas_width=1000, canvas_height=600, window_timestamps=None):
    """Create background plot image for canvas drawing"""
    try:
        safe_matplotlib_cleanup()
        time_data = np.array(time_data_tuple)
        angle_data = np.array(angle_data_tuple)
        dpi = 100
        fig_width = canvas_width / dpi
        fig_height = canvas_height / dpi
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        left_margin = 80 / canvas_width
        right_margin = 50 / canvas_width
        bottom_margin = 80 / canvas_height
        top_margin = 50 / canvas_height
        plot_left = left_margin
        plot_right = 1 - right_margin
        plot_bottom = bottom_margin
        plot_top = 1 - top_margin
        ax.set_position([plot_left, plot_bottom, plot_right - plot_left, plot_top - plot_bottom])
        ax.plot(time_data, angle_data, color='blue', linewidth=2, label='Original Curve', alpha=0.8)
        ax.set_xlabel("Timestamp", fontsize=12)
        ax.set_ylabel("Deg", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("Draw modifications (red) on the curve", fontsize=14)
        y_min_original = angle_data.min()
        y_max_original = angle_data.max()
        y_center = (y_max_original + y_min_original) / 2
        y_range_original = y_max_original - y_min_original
        expansion_factor = 4.0
        expanded_range = max(y_range_original * expansion_factor, 50.0)
        y_min_expanded = y_center - expanded_range / 2
        y_max_expanded = y_center + expanded_range / 2
        ax.set_xlim(time_data.min(), time_data.max())
        ax.set_ylim(y_min_expanded, y_max_expanded)
        # Draw orange shaded vertical lines for selected window timestamps
        if window_timestamps is not None:
            try:
                t_values = [t for t in window_timestamps if t is not None]
                if len(t_values) > 0:
                    band_half = max((float(time_data.max()) - float(time_data.min())) * 0.005, 0.5)
                    for t in t_values:
                        t_clamped = float(np.clip(float(t), float(time_data.min()), float(time_data.max())))
                        ax.axvspan(t_clamped - band_half, t_clamped + band_half, color='orange', alpha=0.2)
                        ax.axvline(t_clamped, color='orange', linewidth=2, alpha=0.9)
            except Exception:
                pass
        plot_bounds = {
            'left': plot_left * canvas_width,
            'right': plot_right * canvas_width,
            'top': (1 - plot_top) * canvas_height,
            'bottom': (1 - plot_bottom) * canvas_height,
            'width': (plot_right - plot_left) * canvas_width,
            'height': (plot_top - plot_bottom) * canvas_height
        }
        data_bounds = {
            'x_min': float(time_data.min()),
            'x_max': float(time_data.max()),
            'y_min': float(y_min_expanded),
            'y_max': float(y_max_expanded)
        }
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, facecolor='white', edgecolor='none', pad_inches=0)
        buf.seek(0)
        background = Image.open(buf)
        if background.size != (canvas_width, canvas_height):
            background = background.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        plt.close(fig)
        return background, plot_bounds, data_bounds
    except Exception as e:
        st.error(f"Failed to create background image: {e}")
        blank_image = Image.new('RGB', (canvas_width, canvas_height), 'white')
        empty_bounds = {'left': 0, 'right': canvas_width, 'top': 0, 'bottom': canvas_height}
        empty_data_bounds = {'x_min': 0, 'x_max': 1, 'y_min': 0, 'y_max': 1}
        return blank_image, empty_bounds, empty_data_bounds

def extract_path_points(canvas_result):
    """Extract path coordinates from canvas drawing result"""
    if not canvas_result.json_data or not canvas_result.json_data.get("objects"):
        return []
    try:
        all_points = []
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "path" and "path" in obj:
                path_points = []
                for segment in obj["path"]:
                    if len(segment) >= 3:
                        cmd = segment[0]
                        if cmd in ("M", "L"):
                            x, y = segment[1], segment[2]
                            path_points.append((x, y))
                        elif cmd == "Q" and len(segment) >= 5:
                            x, y = segment[3], segment[4]
                            path_points.append((x, y))
                        elif cmd == "C" and len(segment) >= 7:
                            x, y = segment[5], segment[6]
                            path_points.append((x, y))
                if path_points:
                    all_points.extend(path_points)
        return all_points
    except Exception as e:
        st.error(f"Failed to extract path points: {e}")
        return []

def safe_update_dataframe_column(output_df, current_column, modified_data, array_index):
    """Safely update DataFrame column with modified data"""
    try:
        if isinstance(modified_data, pd.Series):
            modified_values = modified_data.values
        else:
            modified_values = np.array(modified_data, dtype=float)
        if len(modified_values) != len(output_df):
            return False, f"Data length mismatch: DataFrame={len(output_df)}, Modified={len(modified_values)}"
        sample_value = output_df.iloc[0, current_column]
        if isinstance(sample_value, (list, tuple, np.ndarray)):
            for i in range(len(output_df)):
                old_val = output_df.iloc[i, current_column]
                new_scalar_value = float(modified_values[i])
                if isinstance(old_val, list):
                    new_val = old_val.copy()
                    while len(new_val) <= array_index:
                        new_val.append(0.0)
                    new_val[array_index] = new_scalar_value
                    output_df.iat[i, current_column] = new_val
                elif isinstance(old_val, tuple):
                    new_val = list(old_val)
                    while len(new_val) <= array_index:
                        new_val.append(0.0)
                    new_val[array_index] = new_scalar_value
                    output_df.iat[i, current_column] = tuple(new_val)
                elif isinstance(old_val, np.ndarray):
                    new_val = old_val.copy()
                    if len(new_val) <= array_index:
                        new_shape = list(new_val.shape)
                        new_shape[0] = array_index + 1
                        temp_array = np.zeros(new_shape)
                        temp_array[:len(new_val)] = new_val
                        new_val = temp_array
                    new_val[array_index] = new_scalar_value
                    output_df.iat[i, current_column] = new_val
                else:
                    output_df.iat[i, current_column] = new_scalar_value
        else:
            output_df.iloc[:, current_column] = modified_values.astype(float)
        return True, f"Data updated successfully, processed {len(modified_values)} data points"
    except Exception as e:
        return False, f"Data update failed: {str(e)}"

def load_episode_data(file_path, edit_column, array_index):
    """Load episode data from parquet file"""
    try:
        df = pd.read_parquet(file_path).iloc[:, :2]
        raw_series = df.iloc[:, edit_column]
        def extract_value(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                if len(x) > array_index:
                    return x[array_index]
                else:
                    return 0.0
            else:
                return x
        angle_data = raw_series.apply(extract_value).copy()
        angle_data = pd.Series(angle_data, dtype=float)
        return angle_data, None
    except Exception as e:
        return None, str(e)

# Initialize session state
initialize_session_state()

# Error handling and reset mechanism
if st.session_state.error_count >= st.session_state.max_errors:
    st.error("Multiple errors detected, resetting state...")
    clear_session_data()
    st.session_state.error_count = 0
    st.session_state.canvas_key_version += 1
    safe_rerun()

# Main UI components
st.sidebar.header("Task Selection")
st.sidebar.info("Column 0: Action output, Column 1: Robot state")
st.sidebar.info("Joints: 0=shoulder_pan, 1=shoulder_lift, 2=elbow_flex, 3=wrist_flex, 4=wrist_roll, 5=gripper_pos")
dataset_name = st.sidebar.text_input("Dataset Name", value="softhandling", max_chars=50)
episode_idx = st.sidebar.number_input("Episode Index", min_value=0, step=1, value=0)
edit_column = st.sidebar.selectbox("Select Column", options=[0,1], format_func=lambda x: f"Column {x}", index=st.session_state.edit_column)
array_index = st.sidebar.selectbox("Select Joint", options=[0, 1, 2, 3, 4, 5], format_func=lambda x: f"Joint {x}", index=st.session_state.array_index)
# File path configuration
base_path = f"data/train/Original/{dataset_name}/data/chunk-000"
file_path = os.path.join(base_path, f"episode_{episode_idx:06d}.parquet")
save_dir = f"data/train/Poisoned/{dataset_name}/data/chunk-000"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"episode_{episode_idx:06d}.parquet")

# Check if file exists
if not os.path.exists(file_path):
    st.error(f"File not found: {file_path}")
    st.stop()

# Data reload detection
current_file_hash = get_file_hash(file_path)
try:
    need_reload = (
        st.session_state.get('current_dataset') != dataset_name or 
        st.session_state.get('current_episode') != episode_idx or
        st.session_state.get('current_array_index') != array_index or
        st.session_state.get('edit_column', 0) != edit_column or
        st.session_state.get('last_file_hash') != current_file_hash or
        st.session_state.get('original_data') is None
    )
except Exception:
    need_reload = True

if need_reload:
    clear_session_data()
    with st.spinner(f"Loading Episode {episode_idx}..."):
        angle_data, error_msg = load_episode_data(file_path, edit_column, array_index)
        if angle_data is not None:
            st.session_state.original_data = angle_data
            st.session_state.modified_data = None
            st.session_state.current_dataset = dataset_name
            st.session_state.current_episode = episode_idx
            st.session_state.current_array_index = array_index
            st.session_state.edit_column = edit_column
            st.session_state.last_file_hash = current_file_hash
            st.session_state.canvas_key_version += 1
            st.session_state.error_count = 0
            st.success(f"Successfully loaded dataset: {dataset_name}, Episode: {episode_idx}, Joint: {array_index}")
        else:
            st.error(f"Failed to load data: {error_msg}")
            st.session_state.error_count += 1
            st.stop()

# Data processing and display
if st.session_state.original_data is not None:
    angle_data = st.session_state.original_data
    time_data = np.arange(len(angle_data))
    st.subheader(f"Currently Editing: {dataset_name} - Episode {episode_idx} - Column {edit_column} - Joint {array_index}")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Source File: {file_path}")
    with col2:
        st.info(f"Save Path: {save_path}")
    is_valid, message = validate_data(angle_data, f"Column {edit_column} Joint {array_index} Data")
    if not is_valid:
        st.error(f"Data validation failed: {message}")
        st.session_state.error_count += 1
        st.stop()
    st.success(message)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(angle_data))
    with col2:
        st.metric("Min Value", f"{angle_data.min():.2f}")
    with col3:
        st.metric("Max Value", f"{angle_data.max():.2f}")
    with col4:
        st.metric("Mean Value", f"{angle_data.mean():.2f}")

    # Attack window selection
    with st.sidebar.expander("Select Window", expanded=False):
        max_ts = int(len(angle_data) - 1)
        default_ts1 = st.session_state.window_ts1 if (st.session_state.window_ts1 is not None and 0 <= int(st.session_state.window_ts1) <= max_ts) else 0
        default_ts2 = st.session_state.window_ts2 if (st.session_state.window_ts2 is not None and 0 <= int(st.session_state.window_ts2) <= max_ts) else 0
        ts1 = st.number_input("Start", min_value=0, max_value=max_ts, step=1, value=int(default_ts1))
        ts2 = st.number_input("End", min_value=0, max_value=max_ts, step=1, value=int(default_ts2))
        st.session_state.window_ts1 = int(ts1)
        st.session_state.window_ts2 = int(ts2)

    # Drawing parameters
    st.subheader("Drawing Parameters")
    col1, col2 = st.columns(2)
    with col1:
        stroke_width = st.slider("Stroke Width", 1, 10, 3)
    with col2:
        smooth_factor = st.slider("Smoothing Factor", 0.0, 3.0, 1.0, 0.1)

    # Create background image for canvas
    canvas_width, canvas_height = 1000, 600
    try:
        background, plot_bounds, data_bounds = create_background_image(
            tuple(time_data), tuple(angle_data), canvas_width, canvas_height,
            (st.session_state.window_ts1, st.session_state.window_ts2)
        )
        st.session_state.plot_bounds = plot_bounds
        st.session_state.data_bounds = data_bounds
    except Exception as e:
        st.error(f"Failed to create background image: {e}")
        st.session_state.error_count += 1
        st.stop()

    # Canvas for drawing modifications
    st.subheader("Draw Modified Curve Segment on Original Plot")
    st.info("Tip: Use the mouse to draw red modification segments on the blue curve.")
    canvas_key = f"canvas_{dataset_name}_{episode_idx}_{edit_column}_{array_index}_v{st.session_state.canvas_key_version}"
    try:
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.1)",
            stroke_width=stroke_width,
            stroke_color="#FF0000",
            background_image=background,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            key=canvas_key,
        )
    except Exception as e:
        st.error(f"Canvas component error: {e}")
        st.session_state.error_count += 1
        if st.button("Retry"):
            st.session_state.canvas_key_version += 1
            safe_rerun()
        st.stop()

    # Process drawing results and show modifications
    if canvas_result and canvas_result.json_data:
        path_points = extract_path_points(canvas_result)
        if path_points:
            try:
                modified_data = interpolate_modification(
                    st.session_state.original_data, path_points, 
                    st.session_state.plot_bounds, st.session_state.data_bounds
                )
                st.session_state.modified_data = modified_data
                st.subheader("Modification Statistics")
                diff = modified_data - st.session_state.original_data
                changed_points = np.sum(np.abs(diff) > 0.01)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Modified Points", changed_points)
                with col2:
                    st.metric("Max Change", f"{np.abs(diff).max():.2f}")
                with col3:
                    if changed_points > 0:
                        avg_change = np.abs(diff[np.abs(diff) > 0.01]).mean()
                        st.metric("Average Change", f"{avg_change:.2f}")
                    else:
                        st.metric("Average Change", "0.00")
                st.subheader("Modification Effect Comparison")
                df_result = pd.DataFrame({
                    "Time (s)": time_data,
                    "Original Value": st.session_state.original_data,
                    "Modified Value": modified_data,
                    "Difference": diff
                })
                st.line_chart(df_result.set_index("Time (s)")[['Original Value', 'Modified Value']])
                st.subheader("Modification Difference")
                st.line_chart(df_result.set_index("Time (s)")["Difference"])
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save to Path", type="primary"):
                        try:
                            original_df = pd.read_parquet(file_path)
                            output_df = original_df.copy()
                            success, message = safe_update_dataframe_column(output_df, edit_column, modified_data, array_index)
                            if success:
                                output_df.to_parquet(save_path, index=False)
                                st.success(f"File saved to: {save_path}")
                            else:
                                st.error(f"{message}")
                        except Exception as e:
                            st.error(f"Save failed: {e}")
                            st.session_state.error_count += 1
                with col2:
                    if st.button("Prepare Download", type="secondary"):
                        try:
                            original_df = pd.read_parquet(file_path)
                            output_df = original_df.copy()
                            success, message = safe_update_dataframe_column(output_df, edit_column, modified_data, array_index)
                            if success:
                                buffer = io.BytesIO()
                                output_df.to_parquet(buffer, index=False)
                                buffer.seek(0)
                                st.download_button(
                                    label="Download Modified Parquet File",
                                    data=buffer,
                                    file_name=f"episode_{episode_idx:06d}_modified.parquet",
                                    mime="application/octet-stream"
                                )
                            else:
                                st.error(f"Prepare download failed: {message}")
                        except Exception as e:
                            st.error(f"Prepare download failed: {e}")
                            st.session_state.error_count += 1
            except Exception as e:
                st.error(f"Error processing drawing result: {e}")
                st.session_state.error_count += 1
        else:
            st.info("Please draw a modification path on the plot")
else:
    st.error("Failed to load data, please check file path and parameter settings")