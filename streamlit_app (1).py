"""
streamlit_app.py

Interactive Streamlit web UI for FFT/Hough k_max accumulator.
Simplified for 2x2x2 lattices with uniform sphere radius.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from kmax_fft_accumulator import (
    estimate_kmax,
    verify_exact_kmax,
    FFTConfig,
)


# ==============================================================================
# Lattice Generation
# ==============================================================================

def generate_fcc_lattice_2x2x2(a=1.0):
    """Generate 2x2x2 FCC lattice with lattice parameter a"""
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    
    points = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for b in basis:
                    point = np.array([i, j, k]) + b
                    points.append(point * a)
    
    return np.array(points)


def generate_primitive_cubic_2x2x2(a=1.0):
    """Generate 2x2x2 primitive cubic lattice with lattice parameter a"""
    points = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                point = np.array([i, j, k])
                points.append(point * a)
    
    return np.array(points)


def create_sphere_config(sphere_centers, radius):
    """Convert sphere centers and radius to sphere config format"""
    return [{'center': tuple(c), 'radius': float(radius)} for c in sphere_centers]


# ==============================================================================
# 3D Visualization
# ==============================================================================

def plot_sphere_centers_3d(sphere_centers, radius, title="Sphere Centers"):
    """Create 3D scatter plot of sphere centers"""
    fig = go.Figure()
    
    # Sphere centers
    fig.add_trace(go.Scatter3d(
        x=sphere_centers[:, 0],
        y=sphere_centers[:, 1],
        z=sphere_centers[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color='blue',
            opacity=0.8,
            symbol='circle'
        ),
        name='Sphere Centers',
        text=[f"Sphere {i+1}" for i in range(len(sphere_centers))],
        hovertemplate='<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
    ))
    
    # Add origin indicator
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='red', symbol='cross'),
        name='Origin',
        hovertemplate='Origin (0,0,0)<extra></extra>'
    ))
    
    # Axes
    max_coord = np.max(np.abs(sphere_centers)) + radius + 0.5
    fig.add_trace(go.Scatter3d(
        x=[0, max_coord], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='red', width=3),
        name='X-axis', showlegend=True, hovertemplate='X-axis<extra></extra>'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, max_coord], z=[0, 0],
        mode='lines', line=dict(color='green', width=3),
        name='Y-axis', showlegend=True, hovertemplate='Y-axis<extra></extra>'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, max_coord],
        mode='lines', line=dict(color='blue', width=3),
        name='Z-axis', showlegend=True, hovertemplate='Z-axis<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
        ),
        width=700,
        height=700,
        showlegend=True,
        hovermode='closest',
    )
    
    return fig


def plot_accumulator_slice(accumulator, axis='z', slice_idx=None, title_suffix=""):
    """Plot a 2D slice of the 3D accumulator"""
    if slice_idx is None:
        slice_idx = accumulator.shape[{'x': 0, 'y': 1, 'z': 2}[axis]] // 2
    
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_num = axis_map[axis]
    
    if axis == 'z':
        data = accumulator[:, :, slice_idx]
        title = f"Accumulator Slice (Z = {slice_idx}) {title_suffix}"
        xlabel, ylabel = "X", "Y"
    elif axis == 'y':
        data = accumulator[:, slice_idx, :]
        title = f"Accumulator Slice (Y = {slice_idx}) {title_suffix}"
        xlabel, ylabel = "X", "Z"
    else:  # axis == 'x'
        data = accumulator[slice_idx, :, :]
        title = f"Accumulator Slice (X = {slice_idx}) {title_suffix}"
        xlabel, ylabel = "Y", "Z"
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale='Viridis',
        colorbar=dict(title="Intensity")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=700,
        height=600,
        hovermode='closest',
    )
    
    return fig


# ==============================================================================
# Page Configuration
# ==============================================================================

st.set_page_config(
    page_title="k_max FFT Estimator - 2x2x2 Lattices",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚛️ k_max Estimation: 2x2x2 Lattices")
st.markdown("**FFT/Hough Accumulator for Uniform Sphere Arrangements**")

# ==============================================================================
# Sidebar Configuration
# ==============================================================================

st.sidebar.markdown("## ⚙️ Configuration")

lattice_type = st.sidebar.radio(
    "Lattice Type",
    ["Primitive Cubic", "Face-Centered Cubic (FCC)"],
    help="Choose the 2x2x2 lattice structure"
)

lattice_param = st.sidebar.slider(
    "Lattice Parameter (a)",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="Lattice constant in Ångströms"
)

sphere_radius = st.sidebar.slider(
    "Sphere Radius (r)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.05,
    help="Radius of all spheres (uniform)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔧 FFT Parameters")

config_preset = st.sidebar.radio(
    "Configuration",
    ["Quick", "Balanced (Recommended)", "High Precision", "Custom"],
    help="FFT accumulator precision vs speed tradeoff"
)

if config_preset == "Quick":
    grid_spacing = 0.2
    gaussian_sigma = 0.20
    grid_extent = 10.0
    verify = False
elif config_preset == "Balanced (Recommended)":
    grid_spacing = 0.1
    gaussian_sigma = 0.15
    grid_extent = 10.0
    verify = False
elif config_preset == "High Precision":
    grid_spacing = 0.05
    gaussian_sigma = 0.10
    grid_extent = 15.0
    verify = True
else:  # Custom
    grid_spacing = st.sidebar.slider(
        "Grid Spacing (Å)",
        min_value=0.02,
        max_value=0.3,
        value=0.1,
        step=0.01,
    )
    gaussian_sigma = st.sidebar.slider(
        "Gaussian Width (Å)",
        min_value=0.05,
        max_value=0.3,
        value=0.15,
        step=0.01,
    )
    grid_extent = st.sidebar.slider(
        "Grid Extent (Å)",
        min_value=5.0,
        max_value=20.0,
        value=10.0,
        step=1.0,
    )
    verify = st.sidebar.checkbox("Verify Peak Exact", value=False)

# Create config
config = FFTConfig(
    grid_spacing=grid_spacing,
    gaussian_sigma=gaussian_sigma,
    grid_extent=grid_extent,
    verify_peak=verify,
)

st.sidebar.info(f"📊 Grid: {config.grid_size}³ points")

# ==============================================================================
# Main Content
# ==============================================================================

# Generate lattice
if lattice_type == "Primitive Cubic":
    sphere_centers = generate_primitive_cubic_2x2x2(lattice_param)
    num_atoms = 8
    lattice_desc = "Primitive Cubic (8 atoms)"
else:  # FCC
    sphere_centers = generate_fcc_lattice_2x2x2(lattice_param)
    num_atoms = 32
    lattice_desc = "Face-Centered Cubic (32 atoms)"

# Create sphere configuration
spheres = create_sphere_config(sphere_centers, sphere_radius)

# Main columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Lattice Information")
    
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.metric("Lattice Type", lattice_type)
        st.metric("Number of Atoms", num_atoms)
    with info_col2:
        st.metric("Lattice Parameter (a)", f"{lattice_param:.3f} Å")
        st.metric("Sphere Radius (r)", f"{sphere_radius:.3f} Å")
    
    st.markdown("---")
    
    # 3D visualization
    st.markdown("### 3D Visualization")
    fig_3d = plot_sphere_centers_3d(sphere_centers, sphere_radius, 
                                     title=f"2x2x2 {lattice_type} Lattice")
    st.plotly_chart(fig_3d, use_container_width=True)

with col2:
    st.markdown("### k_max Estimation")
    
    if st.button("🔍 Estimate k_max", type="primary", use_container_width=True):
        with st.spinner("Computing k_max via 3D FFT..."):
            k_max, peak_val, peak_pos = estimate_kmax(spheres, config)
        
        # Results display
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric("Estimated k_max", k_max, 
                     help="Maximum sphere intersection order (soft estimate)")
        with result_col2:
            st.metric("Peak Value", f"{peak_val:.3f}", 
                     help="Maximum value in FFT accumulator")
        
        st.markdown("---")
        st.markdown("**Peak Location:**")
        peak_col1, peak_col2, peak_col3 = st.columns(3)
        with peak_col1:
            st.metric("X", f"{peak_pos[0]:.4f} Å")
        with peak_col2:
            st.metric("Y", f"{peak_pos[1]:.4f} Å")
        with peak_col3:
            st.metric("Z", f"{peak_pos[2]:.4f} Å")
        
        st.markdown("---")
        
        # Verification section
        st.markdown("### Exact Verification")
        if st.button("🔐 Verify Exact k_max at Peak", use_container_width=True):
            with st.spinner("Verifying..."):
                k_exact, details = verify_exact_kmax(spheres, peak_pos, config)
            
            verify_col1, verify_col2 = st.columns(2)
            with verify_col1:
                st.success(f"**Exact k_max: {k_exact}**")
            with verify_col2:
                if k_exact == k_max:
                    st.info("✓ Soft and exact match!")
                else:
                    st.warning(f"Soft={k_max}, Exact={k_exact}")
            
            with st.expander("Verification Details"):
                detail_data = []
                for sphere_detail in details['spheres']:
                    detail_data.append({
                        'Sphere': sphere_detail['idx'],
                        'Center X': f"{sphere_detail['center'][0]:.3f}",
                        'Center Y': f"{sphere_detail['center'][1]:.3f}",
                        'Center Z': f"{sphere_detail['center'][2]:.3f}",
                        'Radius': f"{sphere_detail['radius']:.3f}",
                        'Distance': f"{sphere_detail['distance']:.6f}",
                        'On Surface': "✓" if sphere_detail['on_surface'] else "✗"
                    })
                
                df_verify = pd.DataFrame(detail_data)
                st.dataframe(df_verify, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Atom Coordinates")
    
    coord_data = []
    for i, center in enumerate(sphere_centers):
        coord_data.append({
            'Atom': i + 1,
            'X (Å)': f"{center[0]:.3f}",
            'Y (Å)': f"{center[1]:.3f}",
            'Z (Å)': f"{center[2]:.3f}",
        })
    
    df_coords = pd.DataFrame(coord_data)
    
    with st.expander(f"Show all {num_atoms} atom coordinates"):
        st.dataframe(df_coords, use_container_width=True)

# ==============================================================================
# Accumulator Visualization (if computation was run)
# ==============================================================================

st.markdown("---")
st.markdown("### FFT Accumulator Visualization")

col_acc1, col_acc2 = st.columns([1, 1])

with col_acc1:
    if st.button("📊 Generate Accumulator Map", use_container_width=True):
        with st.spinner("Computing accumulator..."):
            k_max_temp, peak_val_temp, peak_pos_temp, accumulator, indices = estimate_kmax(
                spheres, config, return_accumulator=True
            )
        
        st.session_state.accumulator = accumulator
        st.success("✓ Accumulator computed")

with col_acc2:
    if 'accumulator' in st.session_state:
        slice_axis = st.radio("Slice Axis", options=['X', 'Y', 'Z'], horizontal=True)
        axis_map = {'X': 'x', 'Y': 'y', 'Z': 'z'}
        axis = axis_map[slice_axis]
        
        accumulator = st.session_state.accumulator
        max_slice = accumulator.shape[{'x': 0, 'y': 1, 'z': 2}[axis]] - 1
        
        slice_idx = st.slider(
            f"Slice Index ({axis.upper()})",
            min_value=0,
            max_value=max_slice,
            value=max_slice // 2
        )
        
        fig_heatmap = plot_accumulator_slice(accumulator, axis=axis, slice_idx=slice_idx)
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ==============================================================================
# Footer
# ==============================================================================

st.divider()
st.caption(
    "FFT/Hough Accumulator for maximum sphere intersection order estimation. "
    "Optimized for crystallographic analysis of 2x2x2 lattices with uniform sphere radii."
)
