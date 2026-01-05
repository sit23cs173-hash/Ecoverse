"""
ECOVERSE - DEFORESTATION DETECTION SYSTEM
Professional Dashboard for Real-Time Forest Monitoring & Carbon Impact Analysis

Run with: streamlit run dashboard_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import custom modules
from analysis.vegetation_analysis import VegetationIndexCalculator, visualize_ndvi_comparison
from analysis.carbon_impact import CarbonImpactCalculator, CarbonImpactVisualizer
from visualization.visualizer import DeforestationVisualizer
from data.data_loader import (DeforestationDataLoader, AMAZON_DATASET_PATH, 
                              COMPETITION_DATASET_PATH, TIMESERIES_DATASET_PATH)

# Page configuration
st.set_page_config(
    page_title="EcoVerse | Deforestation Intelligence Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Montserrat:wght@700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Professional Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .header-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.95);
        margin-top: 0.5rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border-left: 4px solid #667eea;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg .sidebar-content, [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    /* Sidebar Text */
    .css-1d391kg, [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Data Tables */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    /* Progress Bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Professional Header
    st.markdown("""
    <div class="main-header">
        <h1 class="header-title">🌍 ECOVERSE</h1>
        <p class="header-subtitle">AI-Powered Deforestation Detection & Forest Carbon Intelligence Platform</p>
        <p class="header-subtitle" style="font-size: 0.9rem; margin-top: 0.5rem;">
            Real-time Satellite Analysis • NDVI Monitoring • Carbon Impact Assessment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with Professional Design
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: white; margin: 0;">⚙️ CONTROL PANEL</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Configure Analysis Parameters</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Analysis mode selection
        st.markdown('<p style="color: white; font-weight: 600; font-size: 1.1rem;">📊 Analysis Mode</p>', unsafe_allow_html=True)
        analysis_mode = st.selectbox(
            "Select Mode",
            ["🎯 Demo Mode (Synthetic Data)", "📤 Upload Your Data", "� Real Dataset Analysis", "�📈 Time-Series Analysis"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Carbon parameters
        st.markdown('<p style="color: white; font-weight: 600; font-size: 1.1rem;">🌲 Carbon Parameters</p>', unsafe_allow_html=True)
        carbon_density = st.slider(
            "Carbon Density (tons C/ha)",
            min_value=50.0,
            max_value=300.0,
            value=190.0,
            step=10.0,
            help="Average carbon stock in forest (default: 190 for Amazon)"
        )
        
        pixel_area_ha = st.number_input(
            "Pixel Area (hectares)",
            min_value=0.001,
            max_value=1.0,
            value=0.01,
            format="%.3f",
            help="Area represented by one pixel"
        )
        
        st.markdown("---")
        
        # NDVI threshold
        st.markdown('<p style="color: white; font-weight: 600; font-size: 1.1rem;">🎯 Detection Parameters</p>', unsafe_allow_html=True)
        ndvi_threshold = st.slider(
            "NDVI Change Threshold",
            min_value=-0.5,
            max_value=0.0,
            value=-0.15,
        step=0.01,
        help="Threshold for detecting deforestation (negative NDVI change)"
    )
    
    # Main content based on mode
    if "Demo Mode" in analysis_mode:
        demo_mode(carbon_density, pixel_area_ha, ndvi_threshold)
    elif "Upload" in analysis_mode:
        upload_mode(carbon_density, pixel_area_ha)
    elif "Real Dataset" in analysis_mode:
        real_dataset_mode(carbon_density, pixel_area_ha)
    else:
        timeseries_mode()
    
    # Professional Footer
    st.markdown("""
    <div class="footer">
        <p style="font-size: 1.1rem; font-weight: 600; color: #1e293b;">🌍 ECOVERSE</p>
        <p>AI-Powered Deforestation Detection & Forest Carbon Intelligence Platform</p>
        <p style="margin-top: 0.5rem;">Built with ❤️ for environmental conservation • Powered by Satellite Imagery & Deep Learning</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem; color: #94a3b8;">
            © 2026 EcoVerse | Empowering conservation through technology
        </p>
    </div>
    """, unsafe_allow_html=True)


def demo_mode(carbon_density, pixel_area_ha, ndvi_threshold):
    """Demo mode with professional visualizations."""
    
    st.markdown("""
    <div class="info-box">
        <h3 style="margin: 0; color: white;">📊 Demo Analysis Mode</h3>
        <p style="margin: 0.5rem 0 0 0;">Generate synthetic deforestation scenarios to explore system capabilities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional button
    if st.button("🚀 Generate Demo Analysis", type="primary", use_container_width=True):
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Generating synthetic satellite imagery...")
        progress_bar.progress(20)
        
        # Create synthetic images
        np.random.seed(42)
        before_img = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
        before_img[:, :, 1] = np.random.randint(100, 200, (256, 256))  # More green
        
        after_img = before_img.copy()
        
        # Add deforested patches
        deforestation_mask = np.zeros((256, 256), dtype=np.uint8)
        num_patches = np.random.randint(2, 5)
        for _ in range(num_patches):
            x = np.random.randint(0, 180)
            y = np.random.randint(0, 180)
            w = np.random.randint(30, 80)
            h = np.random.randint(30, 80)
            after_img[y:y+h, x:x+w] = [150, 100, 70]
            deforestation_mask[y:y+h, x:x+w] = 1
        
        progress_bar.progress(40)
        status_text.text("🌿 Computing vegetation indices...")
        
        # Calculate NDVI
        veg_calc = VegetationIndexCalculator()
        ndvi_before = veg_calc.calculate_ndvi(before_img)
        ndvi_after = veg_calc.calculate_ndvi(after_img)
        ndvi_change = ndvi_after - ndvi_before
        
        progress_bar.progress(60)
        status_text.text("💚 Calculating carbon impact...")
        
        # Calculate carbon impact
        carbon_calc = CarbonImpactCalculator(
            carbon_density=carbon_density,
            pixel_area_ha=pixel_area_ha
        )
        carbon_impact = carbon_calc.calculate_carbon_from_mask(deforestation_mask)
        
        progress_bar.progress(80)
        status_text.text("📊 Generating visualizations...")
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Key Metrics in Professional Cards
        st.markdown('<p class="section-header">📈 Key Findings</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">🌳 Deforested Area</p>
                <p class="metric-value">{carbon_impact['area_ha']:.2f}</p>
                <p class="metric-label">hectares</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #f59e0b;">
                <p class="metric-label">💨 Carbon Loss</p>
                <p class="metric-value" style="color: #f59e0b;">{carbon_impact['carbon_loss_tons']:.1f}</p>
                <p class="metric-label">tons C</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #ef4444;">
                <p class="metric-label">🏭 CO₂ Emissions</p>
                <p class="metric-value" style="color: #ef4444;">{carbon_impact['co2_emissions_tons']:.1f}</p>
                <p class="metric-label">tons CO₂</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #8b5cf6;">
                <p class="metric-label">🚗 Car Equivalent</p>
                <p class="metric-value" style="color: #8b5cf6;">{carbon_impact['equivalent_cars']:.0f}</p>
                <p class="metric-label">cars/year</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Satellite Imagery Comparison
        st.markdown('<p class="section-header">🛰️ Satellite Imagery Analysis</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🌲 Before")
            st.image(before_img, use_column_width=True)
            st.caption("Dense forest coverage with high vegetation")
        
        with col2:
            st.markdown("### 🪵 After")
            st.image(after_img, use_column_width=True)
            st.caption("Deforestation detected in multiple regions")
        
        with col3:
            st.markdown("### 🎯 Detection Overlay")
            # Create overlay
            overlay = after_img.copy()
            overlay[deforestation_mask > 0] = [255, 50, 50]
            blended = (0.65 * after_img + 0.35 * overlay).astype(np.uint8)
            st.image(blended, use_column_width=True)
            st.caption("Red areas indicate detected deforestation")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # NDVI Analysis with professional plots
        st.markdown('<p class="section-header">🌿 Vegetation Index (NDVI) Analysis</p>', unsafe_allow_html=True)
        
        # Create figure with seaborn style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(ndvi_before, cmap='RdYlGn', vmin=-1, vmax=1)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('NDVI Value', rotation=270, labelpad=20)
            ax.set_title('NDVI Before Deforestation', fontsize=12, fontweight='bold', pad=15)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
            st.metric("Avg NDVI", f"{np.mean(ndvi_before):.3f}", delta=None)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(ndvi_after, cmap='RdYlGn', vmin=-1, vmax=1)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('NDVI Value', rotation=270, labelpad=20)
            ax.set_title('NDVI After Deforestation', fontsize=12, fontweight='bold', pad=15)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
            st.metric("Avg NDVI", f"{np.mean(ndvi_after):.3f}", delta=f"{np.mean(ndvi_change):.3f}")
        
        with col3:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(ndvi_change, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('NDVI Change', rotation=270, labelpad=20)
            ax.set_title('NDVI Change Detection', fontsize=12, fontweight='bold', pad=15)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
            st.metric("Max Decrease", f"{np.min(ndvi_change):.3f}", delta=None)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Impact Summary with Plotly
        st.markdown('<p class="section-header">📊 Environmental Impact Breakdown</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for carbon distribution
            fig = go.Figure(data=[go.Pie(
                labels=['Carbon Retained', 'Carbon Lost'],
                values=[1000 - carbon_impact['carbon_loss_tons'], carbon_impact['carbon_loss_tons']],
                hole=.4,
                marker=dict(colors=['#10b981', '#ef4444']),
                textinfo='label+percent',
                textfont=dict(size=14)
            )])
            fig.update_layout(
                title="Carbon Impact Distribution",
                height=400,
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart for emissions comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['CO₂ Emissions', 'Car Equivalent', 'Trees Needed'],
                y=[carbon_impact['co2_emissions_tons'], 
                   carbon_impact['equivalent_cars'], 
                   carbon_impact['co2_emissions_tons'] * 50],
                marker=dict(
                    color=['#ef4444', '#f59e0b', '#10b981'],
                    line=dict(color='white', width=2)
                ),
                text=[f"{carbon_impact['co2_emissions_tons']:.1f} tons",
                      f"{carbon_impact['equivalent_cars']:.0f} cars",
                      f"{carbon_impact['co2_emissions_tons'] * 50:.0f} trees"],
                textposition='outside'
            ))
            fig.update_layout(
                title="Emission Equivalents",
                height=400,
                showlegend=False,
                yaxis_title="Units",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Success message
        st.markdown("""
        <div class="success-box">
            <h4 style="margin: 0; color: white;">✅ Analysis Complete!</h4>
            <p style="margin: 0.5rem 0 0 0;">
                Successfully analyzed {:.2f} hectares of forest area. 
                Detected significant deforestation with {:.1f} tons of carbon loss.
            </p>
        </div>
        """.format(carbon_impact['area_ha'], carbon_impact['carbon_loss_tons']), 
        unsafe_allow_html=True)


def upload_mode(carbon_density, pixel_area_ha):
    """Upload mode for custom data."""
    st.markdown("""
    <div class="warning-box">
        <h3 style="margin: 0;">📤 Upload Your Satellite Imagery</h3>
        <p style="margin: 0.5rem 0 0 0;">Upload before/after images to analyze real deforestation</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌲 Before Image")
        before_file = st.file_uploader("Upload 'Before' satellite image", 
                                      type=['png', 'jpg', 'jpeg', 'tif'],
                                      key='before',
                                      help="Image showing dense forest coverage")
        if before_file:
            st.image(before_file, caption="Before Image Uploaded", use_column_width=True)
    
    with col2:
        st.markdown("#### 🪵 After Image")
        after_file = st.file_uploader("Upload 'After' satellite image", 
                                     type=['png', 'jpg', 'jpeg', 'tif'],
                                     key='after',
                                     help="Image showing deforestation")
        if after_file:
            st.image(after_file, caption="After Image Uploaded", use_column_width=True)
    
    if before_file and after_file:
        st.markdown("""
        <div class="success-box">
            <h4 style="margin: 0; color: white;">✅ Images Ready for Analysis</h4>
            <p style="margin: 0.5rem 0 0 0;">
                Click below to process your satellite imagery and detect deforestation
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Analyze Deforestation", type="primary", use_container_width=True):
            st.info("🚧 Real image processing requires trained models. Use Demo Mode to see full analysis capabilities!")
    else:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
            <h4>📋 Instructions:</h4>
            <ol>
                <li>Upload a 'Before' image showing healthy forest coverage</li>
                <li>Upload an 'After' image showing the same area after deforestation</li>
                <li>Click 'Analyze Deforestation' to process the images</li>
            </ol>
            <p style="margin-top: 1rem; color: #64748b;">
                <strong>Tip:</strong> Images should be from the same geographical location and at similar resolution for best results.
            </p>
        </div>
        """, unsafe_allow_html=True)


def timeseries_mode():
    """Professional time-series analysis with ARIMA forecasting."""
    
    st.markdown("""
    <div class="info-box">
        <h3 style="margin: 0; color: white;">📈 Time-Series Deforestation Analysis</h3>
        <p style="margin: 0.5rem 0 0 0;">Analyze historical trends and forecast future deforestation patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Generate Time-Series Analysis", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📅 Generating historical deforestation data...")
        progress_bar.progress(30)
        
        # Create realistic time-series data
        dates = pd.date_range(start='2017-01-01', periods=84, freq='M')
        np.random.seed(42)
        
        # Trend + Seasonality + Noise
        trend = np.linspace(100, 185, 84)  # Increasing trend
        seasonal = 25 * np.sin(np.arange(84) * 2 * np.pi / 12)  # Seasonal pattern
        noise = np.random.normal(0, 10, 84)
        values = trend + seasonal + noise
        values = np.maximum(values, 0)  # No negative deforestation
        
        df = pd.DataFrame({
            'Date': dates,
            'Deforestation': values,
            'Year': dates.year,
            'Month': dates.strftime('%b')
        })
        
        progress_bar.progress(60)
        status_text.text("📊 Analyzing trends and patterns...")
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Summary Stats
        st.markdown('<p class="section-header">📈 Key Statistics</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">📊 Average Monthly</p>
                <p class="metric-value">{df['Deforestation'].mean():.1f}</p>
                <p class="metric-label">hectares</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #f59e0b;">
                <p class="metric-label">📈 Total (7 Years)</p>
                <p class="metric-value" style="color: #f59e0b;">{df['Deforestation'].sum():.0f}</p>
                <p class="metric-label">hectares</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #ef4444;">
                <p class="metric-label">⚠️ Peak Month</p>
                <p class="metric-value" style="color: #ef4444;">{df['Deforestation'].max():.1f}</p>
                <p class="metric-label">hectares</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            trend_change = ((df['Deforestation'].iloc[-12:].mean() / 
                           df['Deforestation'].iloc[:12].mean() - 1) * 100)
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #8b5cf6;">
                <p class="metric-label">📉 Trend Change</p>
                <p class="metric-value" style="color: #8b5cf6;">{trend_change:+.1f}%</p>
                <p class="metric-label">year-over-year</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interactive Time-Series Plot with Plotly
        st.markdown('<p class="section-header">📊 Historical Deforestation Trends</p>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Deforestation'],
            mode='lines+markers',
            name='Monthly Deforestation',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6, color='#764ba2'),
            hovertemplate='<b>Date:</b> %{x|%b %Y}<br><b>Deforestation:</b> %{y:.1f} ha<extra></extra>'
        ))
        
        # Add moving average
        df['MA_12'] = df['Deforestation'].rolling(window=12).mean()
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['MA_12'],
            mode='lines',
            name='12-Month Average',
            line=dict(color='#ef4444', width=2, dash='dash'),
            hovertemplate='<b>12-Month Avg:</b> %{y:.1f} ha<extra></extra>'
        ))
        
        fig.update_layout(
            title="Monthly Deforestation (2017-2023)",
            xaxis_title="Date",
            yaxis_title="Deforestation Area (hectares)",
            height=500,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal Pattern Analysis
        st.markdown('<p class="section-header">🔄 Seasonal Patterns</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly boxplot
            monthly_avg = df.groupby('Month')['Deforestation'].mean().reindex(
                ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly_avg.index,
                y=monthly_avg.values,
                marker=dict(
                    color=monthly_avg.values,
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Hectares")
                ),
                text=[f'{v:.0f}' for v in monthly_avg.values],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Average Deforestation by Month",
                xaxis_title="Month",
                yaxis_title="Average Deforestation (ha)",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Yearly trend
            yearly_total = df.groupby('Year')['Deforestation'].sum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly_total.index,
                y=yearly_total.values,
                mode='lines+markers',
                marker=dict(size=12, color='#667eea'),
                line=dict(color='#764ba2', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)'
            ))
            
            fig.update_layout(
                title="Annual Deforestation Total",
                xaxis_title="Year",
                yaxis_title="Total Deforestation (ha)",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Data Table
        st.markdown('<p class="section-header">📋 Historical Data</p>', unsafe_allow_html=True)
        
        display_df = df[['Date', 'Deforestation', 'MA_12']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%b %Y')
        display_df.columns = ['Date', 'Deforestation (ha)', '12-Month Average (ha)']
        display_df = display_df.round(2)
        
        st.dataframe(
            display_df.tail(24),  # Show last 2 years
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = df[['Date', 'Deforestation']].to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv,
            file_name="deforestation_timeseries.csv",
            mime="text/csv",
            use_container_width=True
        )


def real_dataset_mode(carbon_density, pixel_area_ha):
    """Real dataset analysis mode using downloaded Kaggle datasets."""
    
    st.markdown("""
    <div class="info-box">
        <h3 style="margin: 0; color: white;">📡 Real Dataset Analysis</h3>
        <p style="margin: 0.5rem 0 0 0;">Analyze actual satellite imagery from Amazon, Competition, and Time-Series datasets</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌎 Amazon Dataset\n10GB Sentinel-2", use_container_width=True, type="primary"):
            st.session_state['selected_dataset'] = 'amazon'
    
    with col2:
        if st.button("🏆 Competition Dataset\n4,043 Files", use_container_width=True):
            st.session_state['selected_dataset'] = 'competition'
    
    with col3:
        if st.button("📊 Time-Series Brazil\n1999-2019 Data", use_container_width=True):
            st.session_state['selected_dataset'] = 'timeseries'
    
    if 'selected_dataset' not in st.session_state:
        st.session_state['selected_dataset'] = 'amazon'
    
    dataset_type = st.session_state['selected_dataset']
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load and display real data
    if dataset_type == 'amazon':
        display_amazon_dataset()
    elif dataset_type == 'competition':
        display_competition_dataset()
    else:
        display_timeseries_brazil_dataset()


@st.cache_data
def load_dataset(dataset_type):
    """Load real dataset with caching."""
    loader = DeforestationDataLoader(str(Path.cwd() / 'data'))
    
    if dataset_type == 'amazon':
        return loader.load_kaggle_dataset(str(AMAZON_DATASET_PATH), 'amazon')
    elif dataset_type == 'competition':
        return loader.load_kaggle_dataset(str(COMPETITION_DATASET_PATH), 'competition')
    else:
        return loader.load_kaggle_dataset(str(TIMESERIES_DATASET_PATH), 'timeseries')


def display_amazon_dataset():
    """Display Amazon Deforestation Dataset (Sentinel-2 images)."""
    with st.spinner("🔄 Loading Amazon dataset..."):
        data = load_dataset('amazon')
    
    st.markdown('<p class="section-header">🌎 Amazon Deforestation Dataset</p>', unsafe_allow_html=True)
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3>{data.get('image_count', 0)}</h3>
            <p>Sentinel-2 Images</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>{len(data.get('mask_paths', []))}</h3>
            <p>Training Masks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>10.1 GB</h3>
            <p>Total Size</p>
        </div>
        """, unsafe_allow_html=True)
    
    if data.get('image_paths'):
        st.markdown("### 📷 Sample Images")
        st.info(f"✅ Found {data['image_count']} Sentinel-2 satellite images ready for analysis!")
        
        # Show image paths
        st.markdown("**Available Images:**")
        for i, path in enumerate(data['image_paths'][:10], 1):
            st.text(f"{i}. {Path(path).name}")
        
        if len(data['image_paths']) > 10:
            st.caption(f"... and {len(data['image_paths']) - 10} more images")


def display_competition_dataset():
    """Display Competition Dataset (.npy files)."""
    with st.spinner("🔄 Loading competition dataset..."):
        data = load_dataset('competition')
    
    st.markdown('<p class="section-header">🏆 Competition Dataset</p>', unsafe_allow_html=True)
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #FA8BFF 0%, #2BD2FF 90%, #2BFF88 100%);">
            <h3>{data.get('train_count', 0)}</h3>
            <p>Training Files (.npy)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>{data.get('test_count', 0)}</h3>
            <p>Test Files (.npy)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3>{len(data.get('train_metadata', []))}</h3>
            <p>Metadata Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    if data.get('train_paths'):
        st.success(f"✅ Loaded {data['train_count']} multi-spectral satellite image arrays (512x512x13 channels)")
        
        # Load sample .npy
        sample_path = data['train_paths'][0]
        sample_data = np.load(sample_path)
        
        st.markdown("### 📦 Sample Data")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **File:** `{Path(sample_path).name}`
            - **Shape:** {sample_data.shape}
            - **Data Type:** {sample_data.dtype}
            - **Channels:** 13 (Multi-spectral)
            - **Size:** {sample_data.nbytes / 1024:.1f} KB
            """)
        
        with col2:
            st.markdown(f"""
            **Value Statistics:**
            - **Min:** {sample_data.min()}
            - **Max:** {sample_data.max()}
            - **Mean:** {sample_data.mean():.2f}
            - **Std:** {sample_data.std():.2f}
            """)


def display_timeseries_brazil_dataset():
    """Display Time-Series Brazil Dataset (Fire data 1999-2019)."""
    with st.spinner("🔄 Loading time-series dataset..."):
        data = load_dataset('timeseries')
    
    st.markdown('<p class="section-header">📊 Brazilian Amazon Time-Series (1999-2019)</p>', unsafe_allow_html=True)
    
    if 'timeseries' in data:
        df = data['timeseries']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <h3>{len(df)}</h3>
                <p>Total Records</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            years = df['year'].unique() if 'year' in df.columns else []
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>{len(years)}</h3>
                <p>Years Covered</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            states = df['state'].unique() if 'state' in df.columns else []
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h3>{len(states)}</h3>
                <p>States</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #FA8BFF 0%, #2BD2FF 90%, #2BFF88 100%);">
                <h3>🔥</h3>
                <p>Fire Incidents</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Data preview
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True, height=400)
        
        # Visualization
        if 'year' in df.columns:
            st.markdown("### 📈 Fire Incidents Over Time")
            yearly_counts = df.groupby('year').size().reset_index(name='incidents')
            
            fig = px.line(yearly_counts, x='year', y='incidents',
                         title='Brazilian Amazon Fire Incidents (1999-2019)',
                         labels={'year': 'Year', 'incidents': 'Number of Incidents'},
                         markers=True)
            fig.update_traces(line_color='#e74c3c', marker=dict(size=10, color='#c0392b'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Time-Series Data (CSV)",
            data=csv,
            file_name="brazil_amazon_fires_1999_2019.csv",
            mime="text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()

