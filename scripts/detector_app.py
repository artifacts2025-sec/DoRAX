# ------------------ Dash Application Interface ------------------
import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import argparse
from scipy.ndimage import gaussian_filter1d

# Import core analysis module
from keyframe import KeyframeAnalyzer, load_dataset, get_video_frame, load_huggingface_token

# ------------------ Command Line Argument Parsing ------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description='Joint Analysis Application')
    parser.add_argument('--vel_threshold', type=float, default=0.5, 
                       help='Velocity threshold for identifying low-speed intervals (default: 0.5)')
    parser.add_argument('--k', type=int, default=2, 
                       help='Maximum length threshold for low-speed intervals (default: 2)')
    parser.add_argument('--highlight_width', type=int, default=1, 
                       help='Highlight width for maximum and minimum points (default: 1)')
    parser.add_argument('--port', type=int, default=7860, 
                       help='Application running port (default: 7860)')
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                       help='Application running host (default: 0.0.0.0)')
    parser.add_argument('--data_source', type=str, default='./data/train/sortingtest', 
                       help='Data source, can be remote repository ID or local path (default: ./data/train/sortingtest)')
    parser.add_argument('--episode_id', type=int, default=0, 
                       help='Episode ID (default: 0)')
    parser.add_argument('--api_token_path', type=str, default=None, 
                       help='HuggingFace API token file path (optional, for private datasets)')
    return parser.parse_args()

# ------------------ Initialize Parameters ------------------
args = parse_arguments()

# Create keyframe analyzer instance, using command line arguments
analyzer = KeyframeAnalyzer(
    vel_threshold=args.vel_threshold,
    k=args.k,
    highlight_width=args.highlight_width
)

print(f"Application startup parameters:")
print(f"vel_threshold: {analyzer.vel_threshold}")
print(f"k: {analyzer.k}")
print(f"highlight_width: {analyzer.highlight_width}")
print(f"Data source: {args.data_source}")
print(f"Episode ID: {args.episode_id}")
print(f"Port: {args.port}")

# Check API token
if args.api_token_path:
    api_token = load_huggingface_token(args.api_token_path)
    if api_token:
        print(f"✅ API token loaded: {api_token[:10]}...")
    else:
        print("⚠️  Specified API token file is invalid")
else:
    # Auto-search for token file
    api_token = load_huggingface_token()
    if api_token:
        print(f"✅ Auto-found API token: {api_token[:10]}...")
    else:
        print("ℹ️  No API token found, will use public access mode")

# ------------------ Dash Initialization ------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ------------------ Page Layout ------------------
app.layout = html.Div([
    # Header with gradient background
    html.Div([
        html.H1("Keyframe Recognition System", 
                style={
                    "textAlign": "center", 
                    "marginBottom": "10px",
                    "color": "white",
                    "fontSize": "2.5rem",
                    "fontWeight": "300",
                    "textShadow": "2px 2px 4px rgba(0,0,0,0.3)"
                }),
        html.P("Interactive Joint Analysis and Video Synchronization", 
               style={
                   "textAlign": "center", 
                   "color": "rgba(255,255,255,0.9)",
                   "fontSize": "1.1rem",
                   "marginBottom": "0"
               })
    ], style={
        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "padding": "30px 20px",
        "marginBottom": "30px",
        "borderRadius": "0 0 15px 15px",
        "boxShadow": "0 4px 20px rgba(0,0,0,0.1)"
    }),
    
    # Control panel
    html.Div([
        html.Div([
            html.Label("Data Source:", 
                      style={
                          "fontWeight": "600",
                          "color": "#333",
                          "marginRight": "10px",
                          "fontSize": "1rem"
                      }),
            dcc.Input(
                id="input-data-source", 
                type="text", 
                value=args.data_source, 
                style={
                    "width": "350px",
                    "padding": "12px 15px",
                    "border": "2px solid #e1e5e9",
                    "borderRadius": "8px",
                    "fontSize": "14px",
                    "transition": "border-color 0.3s ease",
                    "outline": "none"
                },
                placeholder="Enter remote repository ID or local path (e.g.: ./data/train/sortingtest)"
            ),
        ], style={"marginBottom": "15px"}),
        
        html.Div([
            html.Label("Episode ID:", 
                      style={
                          "fontWeight": "600",
                          "color": "#333",
                          "marginRight": "10px",
                          "fontSize": "1rem"
                      }),
            dcc.Input(
                id="input-episode-id", 
                type="number", 
                value=args.episode_id, 
                min=0, 
                style={
                    "width": "120px",
                    "padding": "12px 15px",
                    "border": "2px solid #e1e5e9",
                    "borderRadius": "8px",
                    "fontSize": "14px",
                    "transition": "border-color 0.3s ease",
                    "outline": "none"
                }
            ),
            html.Button(
                "Load Data", 
                id="btn-load", 
                n_clicks=0, 
                style={
                    "marginLeft": "20px",
                    "padding": "12px 25px",
                    "backgroundColor": "#667eea",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "8px",
                    "fontSize": "14px",
                    "fontWeight": "600",
                    "cursor": "pointer",
                    "transition": "all 0.3s ease",
                    "boxShadow": "0 2px 10px rgba(102, 126, 234, 0.3)"
                }
            ),
        ]),
    ], style={
        "textAlign": "center",
        "marginBottom": "40px",
        "padding": "25px",
        "backgroundColor": "white",
        "borderRadius": "12px",
        "boxShadow": "0 4px 20px rgba(0,0,0,0.08)",
        "border": "1px solid #f0f0f0"
    }),
    
    # Loading and data storage
    dcc.Loading(
        id="loading",
        type="circle",
        style={"margin": "20px auto"},
        children=dcc.Store(id="store-data")
    ),
    
    # Main content area
    html.Div(
        id="main-content",
        style={
            "backgroundColor": "#f8f9fa",
            "minHeight": "400px",
            "borderRadius": "12px",
            "padding": "20px"
        }
    ),
    

], style={
    "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    "backgroundColor": "#f5f7fa",
    "minHeight": "100vh",
    "padding": "0"
})

# ------------------ Data Loading Callback ------------------
@app.callback(
    Output("store-data", "data"),
    Input("btn-load", "n_clicks"),
    State("input-data-source", "value"),
    State("input-episode-id", "value"),
    prevent_initial_call=True
)
def load_data_callback(n_clicks, data_source, episode_id):
    try:
        video_paths, data_df = load_dataset(
            data_source=data_source,
            episode_id=int(episode_id)
        )
        if data_df is None or data_df.empty:
            return {}
        return {
            "video_paths": video_paths,
            "data_df": data_df.to_dict("records"),
            "columns": ["shoulder_pan", "shoulder_pitch", "elbow", "wrist_pitch", "wrist_roll", "gripper"],
            "timestamps": data_df["timestamp"].tolist()
        }
    except Exception as e:
        print(f"Data loading error: {e}")
        return {}

# ------------------ Main Content Rendering Callback ------------------
@app.callback(
    Output("main-content", "children"),
    Input("store-data", "data")
)
def update_main_content(data):
    if not data or "data_df" not in data or len(data["data_df"]) == 0:
        return html.Div([
            html.Div("📊", style={"fontSize": "3rem", "marginBottom": "20px"}),
            html.H3("No Data", style={"color": "#666", "marginBottom": "10px"}),
            html.P("Please click the 'Load Data' button above to get data.", 
                   style={"color": "#888", "fontSize": "1rem"})
        ], style={
            "textAlign": "center", 
            "padding": "60px 20px",
            "color": "#666"
        })
    
    columns = data["columns"]
    rows = []
    for i, joint in enumerate(columns):
        rows.append(html.Div([
            # Joint graph - left 50%
            html.Div([
                dcc.Graph(id=f"graph-{i}")
            ], style={
                "flex": "0 0 50%", 
                "backgroundColor": "white",
                "borderRadius": "8px",
                "padding": "8px",
                "boxShadow": "0 2px 10px rgba(0,0,0,0.05)",
                "border": "1px solid #e9ecef",
                "marginRight": "2%"
            }),
            # Video area - right 48%
            html.Div([
                html.Img(id=f"video1-{i}", style={
                    "width": "49%", 
                    "height": "180px", 
                    "objectFit": "contain", 
                    "display": "inline-block",
                    "borderRadius": "6px",
                    "border": "2px solid #e9ecef"
                }),
                html.Img(id=f"video2-{i}", style={
                    "width": "49%", 
                    "height": "180px", 
                    "objectFit": "contain", 
                    "display": "inline-block",
                    "borderRadius": "6px",
                    "border": "2px solid #e9ecef"
                })
            ], style={
                "flex": "0 0 48%"
            }, id=f"video-container-{i}")
        ], style={
            "marginBottom": "25px",
            "backgroundColor": "white",
            "borderRadius": "12px",
            "padding": "12px",
            "boxShadow": "0 4px 15px rgba(0,0,0,0.08)",
            "border": "1px solid #f0f0f0",
            "display": "flex",
            "alignItems": "flex-start",
            "minHeight": "250px"
        }))
    return html.Div(rows)

# ------------------ Chart Generation Function ------------------
def generate_joint_graph(joint_name, idx, action_df, delta_t, time_for_plot, all_shadows):
    """Generate joint chart"""
    angles = action_df[joint_name].values
    velocity = np.diff(angles) / delta_t
    smoothed_velocity = gaussian_filter1d(velocity, sigma=1)
    smoothed_angle = gaussian_filter1d(angles[1:], sigma=1)
    
    shapes = []
    current_shadows = all_shadows[joint_name]
    
    for shadow in current_shadows:
        shapes.append({
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": shadow['start_time'],
            "x1": shadow['end_time'],
            "y0": 0,
            "y1": 1,
            "fillcolor": "#ef4444",  # 固定红色
            "opacity": 0.4,
            "line": {"width": 0}
        })
    
    return {
        "data": [
            go.Scatter(
                x=time_for_plot,
                y=smoothed_angle,
                name="Joint Angle",
                line=dict(color='#f59e0b', width=2),
                hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Angle:</b> %{y:.2f}°<extra></extra>'
            )
        ],
        "layout": go.Layout(
            title={
                'text': joint_name.replace('_', ' ').title(),
                'font': {'size': 16, 'color': '#374151'}
            },
            xaxis={
                "title": "Time (seconds)",
                "titlefont": {"color": "#6b7280"},
                "tickfont": {"color": "#6b7280"},
                "gridcolor": "#f3f4f6",
                "zerolinecolor": "#e5e7eb"
            },
            yaxis={
                "title": "Angle (degrees)",
                "titlefont": {"color": "#6b7280"},
                "tickfont": {"color": "#6b7280"},
                "gridcolor": "#f3f4f6",
                "zerolinecolor": "#e5e7eb"
            },
            shapes=shapes,
            hovermode="x unified",
            height=220,
            margin=dict(t=30, b=30, l=50, r=30),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"},
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
            )
        )
    }

# ------------------ Chart Update Callback ------------------
@app.callback(
    [Output(f"graph-{i}", "figure") for i in range(6)],
    [Input("store-data", "data")],
    prevent_initial_call=True
)
def update_all_graphs(data):
    if not data or "data_df" not in data or len(data["data_df"]) == 0:
        return [no_update] * 6
    
    columns = data["columns"]
    df = pd.DataFrame.from_records(data["data_df"])
    action_df = pd.DataFrame(df["action"].tolist(), columns=columns)
    timestamps = df["timestamp"].values
    delta_t = np.diff(timestamps)
    time_for_plot = timestamps[1:]
    
    # Use keyframe analyzer to analyze all joints
    all_shadows = analyzer.analyze_all_joints(action_df, delta_t, time_for_plot, columns)

    # Generate all charts
    return [
        generate_joint_graph(joint, i, action_df, delta_t, time_for_plot, all_shadows)
        for i, joint in enumerate(columns)
    ]

# ------------------ Video Area Show/Hide Callback ------------------
@app.callback(
    [Output(f"video-container-{i}", "style") for i in range(6)],
    [Input("store-data", "data")],
    prevent_initial_call=True
)
def update_video_containers(data):
    if not data or "video_paths" not in data:
        # Hide all video areas
        return [{"display": "none"} for _ in range(6)]
    
    video_paths = data["video_paths"]
    if not video_paths or len(video_paths) < 2:
        # If not enough video files, hide video areas
        return [{"display": "none"} for _ in range(6)]
    else:
        # Show video areas
        return [{"flex": "0 0 48%"} for _ in range(6)]

# ------------------ Video Frame Callback ------------------
for i in range(6):
    @app.callback(
        Output(f"video1-{i}", "src"),
        Output(f"video2-{i}", "src"),
        Input("store-data", "data"),
        Input(f"graph-{i}", "hoverData"),
        prevent_initial_call=True
    )
    def update_video_frames(data, hover_data, idx=i):
        if not data or "data_df" not in data or len(data["data_df"]) == 0:
            return no_update, no_update
        
        columns = data["columns"]
        df = pd.DataFrame.from_records(data["data_df"])
        timestamps = df["timestamp"].values
        time_for_plot = timestamps[1:]
        video_paths = data["video_paths"]
        
        # Check if video paths are empty
        if not video_paths or len(video_paths) < 2:
            print(f"Insufficient video paths, skipping video frame update. Found {len(video_paths) if video_paths else 0} video files")
            return no_update, no_update
        
        # Determine display time point
        display_time = 0.0  # Default start time
        if hover_data and "points" in hover_data and len(hover_data["points"]) > 0:
            # If hover data exists, use hover time
            display_time = float(hover_data["points"][0]["x"])
        elif len(time_for_plot) > 0:
            # If no hover data, use timeline start time
            display_time = time_for_plot[0]
        
        try:
            frame1 = get_video_frame(video_paths[0], display_time)
            frame2 = get_video_frame(video_paths[1], display_time)
            if frame1 and frame2:
                return frame1, frame2
            else:
                return no_update, no_update
        except Exception as e:
            print(f"Video frame update callback error: {e}")
            return no_update, no_update

# ------------------ Start Application ------------------
if __name__ == "__main__":
    app.run(debug=True, host=args.host, port=args.port) 