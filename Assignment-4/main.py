import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import joblib 
from sklearn.metrics import mean_squared_error
import imageio
import os
import tempfile
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Robot Trajectory AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Modern Dark Theme) ---
st.markdown("""
<style>
    /* General Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #334155;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    h1 {
        background: -webkit-linear-gradient(eee, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Cards / Containers */
    .css-1r6slb0, .stMarkdown {
        border-radius: 12px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #3b82f6;
    }
    
    /* Metric Box */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- MATPLOTLIB DARK THEME ---
plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': '#1e293b',
    'figure.facecolor': '#1e293b',
    'grid.color': '#334155',
    'text.color': '#e2e8f0',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'axes.labelcolor': '#cbd5e1',
    'axes.titlecolor': '#f8fafc'
})

# --- 1. OPTIMIZATION LOGIC (For Ground Truth Comparison) ---
def get_optimized_trajectory(q_start, q_end):
    T, N = 5.0, 50
    dt = T / (N - 1)
    time = np.linspace(0, T, N)

    def cubic_trajectory(q0, qf, T, t):
        return q0 + (3*(qf - q0)/T**2)*t**2 + (-2*(qf - q0)/T**3)*t**3

    x0 = np.concatenate([cubic_trajectory(q_start[0], q_end[0], T, time), 
                         cubic_trajectory(q_start[1], q_end[1], T, time)])

    def cost_function(x):
        q1_ddot = np.diff(x[:N], n=2) / dt**2
        q2_ddot = np.diff(x[N:], n=2) / dt**2
        return np.sum(q1_ddot**2 + q2_ddot**2)

    constraints = [
        {'type': 'eq', 'fun': lambda x: x[0] - q_start[0]},
        {'type': 'eq', 'fun': lambda x: x[N] - q_start[1]},
        {'type': 'eq', 'fun': lambda x: x[N-1] - q_end[0]},
        {'type': 'eq', 'fun': lambda x: x[2*N-1] - q_end[1]},
        {'type': 'eq', 'fun': lambda x: x[1] - x[0]},
        {'type': 'eq', 'fun': lambda x: x[N+1] - x[N]},
        {'type': 'eq', 'fun': lambda x: x[N-1] - x[N-2]},
        {'type': 'eq', 'fun': lambda x: x[2*N-1] - x[2*N-2]}
    ]

    res = minimize(cost_function, x0, method='SLSQP', constraints=constraints)
    return res.x

# --- 4. VIDEO / PLOTTING HELPERS ---
def _plot_arm(fig, q1, q2):
    ax = fig.add_subplot(1,1,1)
    ax.clear()
    # link lengths = 1, base at (0,0)
    x1, y1 = 0.0, 0.0
    x_e1 = x1 + math.cos(q1)
    y_e1 = y1 + math.sin(q1)
    x_e2 = x_e1 + math.cos(q1 + q2)
    y_e2 = y_e1 + math.sin(q1 + q2)

    ax.plot([x1, x_e1], [y1, y_e1], linewidth=3, color='#60a5fa', zorder=2) # Blue
    ax.plot([x_e1, x_e2], [y_e1, y_e2], linewidth=3, color='#c084fc', zorder=2) # Purple
    ax.scatter([x1, x_e1, x_e2], [y1, y_e1, y_e2], color='#e2e8f0', zorder=3, s=50) # White joints
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2-Link Robotic Arm')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.grid(True, linestyle='--', alpha=0.5)

def create_video_from_traj(traj, out_path, N=50, fps=10):
    tmp_dir = tempfile.mkdtemp(prefix='frames_')
    q1 = traj[:N]
    q2 = traj[N:]
    try:
        for i in range(N):
            fig = plt.figure(figsize=(5.12,5.12))
            _plot_arm(fig, q1[i], q2[i])
            frame_path = os.path.join(tmp_dir, f"{i:03d}.png")
            fig.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

        with imageio.get_writer(out_path, fps=fps) as writer:
            for i in range(N):
                frame_path = os.path.join(tmp_dir, f"{i:03d}.png")
                image = imageio.imread(frame_path)
                writer.append_data(image)
    finally:
        try:
            for f in os.listdir(tmp_dir):
                os.remove(os.path.join(tmp_dir, f))
            os.rmdir(tmp_dir)
        except Exception:
            pass

# --- 2. LOAD PRE-TRAINED MODEL WEIGHTS ---
@st.cache_resource
def load_trained_model():
    try:
        model = joblib.load('model_weights.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: 'model_weights.pkl' not found.")
        return None

# --- 3. STREAMLIT UI ---
st.title("Assignment 4: Prediction Dashboard")
st.markdown("### Neural Network vs. Trajectory Optimization")

# --- HORIZONTAL CONFIGURATION BAR ---
with st.container():
    st.header("Positions")
    
    col_config_1, col_config_2 = st.columns(2)
    
    with col_config_1:
        st.subheader("Start")
        c1, c2 = st.columns(2)
        with c1:
            q1_s_deg = st.slider("Joint 1 (°)", -180.0, 180.0, 0.0, key="s1")
        with c2:
            q2_s_deg = st.slider("Joint 2 (°)", -180.0, 180.0, 0.0, key="s2")

    with col_config_2:
        st.subheader("End")
        c3, c4 = st.columns(2)
        with c3:
            q1_e_deg = st.slider("Joint 1 (°)", -180.0, 180.0, 90.0, key="e1")
        with c4:
            q2_e_deg = st.slider("Joint 2 (°)", -180.0, 180.0, 45.0, key="e2")

    st.markdown("---")

q_start = np.radians([q1_s_deg, q2_s_deg])
q_end = np.radians([q1_e_deg, q2_e_deg])

# Instant prediction using loaded weights
model = load_trained_model()

if model is not None:
    # Optimized trajectory (Calculated in real-time)
    opt_traj = get_optimized_trajectory(q_start, q_end)
    
    # Learned trajectory (Predicted instantly using MLP weights)
    input_data = np.concatenate([q_start, q_end]).reshape(1, -1)
    nn_traj = model.predict(input_data)[0]

    # Visual Comparison Plots 
    t = np.linspace(0, 5, 50)
    opt_traj_deg = np.degrees(opt_traj)
    nn_traj_deg = np.degrees(nn_traj)
    
    mse = mean_squared_error(opt_traj_deg, nn_traj_deg)

    # --- RESULTS SECTION ---
    
    # Metric
    col_metric, col_spacer = st.columns([1, 3])
    with col_metric:
        st.metric(label="Prediction Error (MSE)", value=f"{mse:.4f}", delta="sq. degrees", delta_color="inverse")

    st.markdown("####  Trajectory Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Joint-1 Path**")
        fig1, ax1 = plt.subplots()
        ax1.plot(t, opt_traj_deg[:50], '#7c3aed', label="Optimized", linewidth=2.5) # Green
        ax1.plot(t, nn_traj_deg[:50], '#16a34a', linestyle='-', label="Model's Predicted", linewidth=2.5) # Red
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (°)")
        ax1.legend(facecolor='#1e293b', edgecolor='none')
        ax1.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig1)

    with col2:
        st.markdown("**Joint-2 Path**")
        fig2, ax2 = plt.subplots()
        ax2.plot(t, opt_traj_deg[50:], '#7c3aed', label="Optimized", linewidth=2.5)
        ax2.plot(t, nn_traj_deg[50:], '#16a34a', linestyle='-', label="Model's Predicted", linewidth=2.5)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Angle (°)")
        ax2.legend(facecolor='#1e293b', edgecolor='none')
        ax2.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig2)

    # --- Video generation and display ---
    st.markdown("####  Motion Visualization")
    
    if True:
        try:
            tmpdir = tempfile.mkdtemp(prefix='videos_')
            pred_path = os.path.join(tmpdir, "robot_arm_pred.mp4")
            true_path = os.path.join(tmpdir, "robot_arm_true.mp4")
            
            with st.spinner("Visualization..."):
                create_video_from_traj(nn_traj, pred_path, N=50, fps=10)
                create_video_from_traj(opt_traj, true_path, N=50, fps=10)

            v_col1, v_col2 = st.columns(2)
            
            with v_col1:
                st.info("Model's Predicted")
                st.video(pred_path)
            
            with v_col2:
                st.success("Optimized ")
                st.video(true_path)

            # cleanup video files (keep temp dir removal safe)
            try:
                os.remove(pred_path)
                os.remove(true_path)
                os.rmdir(tmpdir)
            except Exception:
                pass
        except Exception as e:
            st.error(f"Video generation failed: {e}")