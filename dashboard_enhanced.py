"""
🌍 ECOVERSE - Enhanced Professional Deforestation Detection Dashboard
Modern UI with Advanced Animations & Transformations

Run: streamlit run dashboard_enhanced.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import cv2
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import modules
from complete_analysis_workflow import CompleteDeforestationAnalysis
from analysis.vegetation_analysis import VegetationIndexCalculator
from analysis.carbon_impact import CarbonImpactCalculator
from data.data_loader import (DeforestationDataLoader, AMAZON_DATASET_PATH, 
                              COMPETITION_DATASET_PATH)
from models.deforestation_model import DeforestationDetector

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="EcoVerse | AI Deforestation Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force sidebar to be visible
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'

# ==================== ENHANCED CSS WITH ANIMATIONS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Animated gradient background */
    .main {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        background-attachment: fixed;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Content container with glass effect */
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        padding: 2.5rem;
        margin: 1.5rem;
        box-shadow: 0 25px 80px rgba(0,0,0,0.25);
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Modern Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        min-width: 350px !important;
        width: 350px !important;
    }
    
    /* Hide sidebar collapse button - always keep sidebar open */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    button[kind="header"] {
        display: none !important;
    }
    
    /* Force sidebar to always be visible */
    section[data-testid="stSidebar"] {
        transform: none !important;
        visibility: visible !important;
        display: block !important;
    }
    
    section[data-testid="stSidebar"] > div {
        transform: none !important;
        visibility: visible !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Animated Header */
    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 25px;
        margin-bottom: 3rem;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
        animation: headerPulse 3s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotateBg 10s linear infinite;
    }
    
    @keyframes headerPulse {
        0%, 100% { box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4); }
        50% { box-shadow: 0 20px 60px rgba(102, 126, 234, 0.6); }
    }
    
    @keyframes rotateBg {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
        animation: titleFloat 4s ease-in-out infinite;
    }
    
    @keyframes titleFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        margin-top: 1rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.5);
    }
    
    .metric-card h2 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        animation: countUp 1s ease-out;
    }
    
    @keyframes countUp {
        from {
            opacity: 0;
            transform: scale(0.5);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .metric-card p {
        color: rgba(255,255,255,0.95);
        font-size: 1rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Info Box with Animation */
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
        animation: slideInLeft 0.6s ease-out;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .info-box h3 {
        color: white;
        font-size: 1.8rem;
        margin: 0 0 0.5rem 0;
        font-weight: 700;
    }
    
    .info-box p {
        color: rgba(255,255,255,0.9);
        margin: 0;
        font-size: 1rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        animation: slideInRight 0.6s ease-out;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 100px;
        height: 3px;
        background: #764ba2;
        animation: expandBar 1s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes expandBar {
        from { width: 0; }
        to { width: 100px; }
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress Bar Animation */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 100%;
        animation: progressGradient 2s ease infinite;
    }
    
    @keyframes progressGradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Card containers */
    .card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .card:hover {
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 15px;
        padding: 1rem;
        animation: slideInLeft 0.5s ease-out;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
        border-right-color: #764ba2 !important;
        animation: spinnerRotate 1s linear infinite !important;
    }
    
    @keyframes spinnerRotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(102, 126, 234, 0.05);
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: #764ba2;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 1rem 2rem;
        background: rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Pulse animation for important elements */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.8;
        }
    }
    
    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Loading skeleton */
    @keyframes skeleton-loading {
        0% {
            background-position: -200% 0;
        }
        100% {
            background-position: 200% 0;
        }
    }
    
    .skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: skeleton-loading 1.5s infinite;
        border-radius: 10px;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-card h2 {
            font-size: 2rem;
        }
        
        .block-container {
            padding: 1rem;
            margin: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>🌍 ECOVERSE</h1>
    <p>AI-Powered Deforestation Detection & Climate Impact Assessment</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    st.markdown("---")
    
    # Analysis Mode Selection
    st.markdown("#### 📊 Analysis Mode")
    analysis_mode = st.selectbox(
        "Choose Mode",
        [
            "🔬 Complete E2E Analysis",
            "⚡ Quick Demo",
            "📤 Upload Custom Images",
            "📡 Real Datasets"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("#### 🌲 Carbon Parameters")
    
    carbon_density = st.slider(
        "Carbon Density (tons C/ha)",
        min_value=50.0,
        max_value=300.0,
        value=190.0,
        step=10.0,
        help="Average carbon stock per hectare"
    )
    
    pixel_area_ha = st.number_input(
        "Pixel Area (hectares)",
        min_value=0.001,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="Area represented by one pixel"
    )
    
    ndvi_threshold = st.slider(
        "NDVI Threshold",
        min_value=-0.5,
        max_value=0.0,
        value=-0.2,
        step=0.05,
        help="Threshold for deforestation detection"
    )
    
    st.markdown("---")
    st.markdown("#### 🤖 AI Model Settings")
    
    use_ml_model = st.checkbox(
        "🧠 Use Trained ML Model",
        value=False,
        help="Enable trained U-Net model for predictions instead of NDVI-based detection"
    )
    
    if use_ml_model:
        model_path = Path("outputs/models/ground_truth_unet_model.h5")
        if model_path.exists():
            st.success(f"✅ Model found ({model_path.stat().st_size / (1024**2):.1f} MB)")
        else:
            st.error("❌ Model not found. Train model first!")
            use_ml_model = False
    
    # Store in session state
    st.session_state['use_ml_model'] = use_ml_model
    
    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.info("**EcoVerse** uses satellite imagery and AI to detect deforestation and assess environmental impact.")
    
    st.markdown("##### 🎯 Features")
    st.markdown("""
    - ✨ Real-time analysis
    - 🤖 AI-powered detection
    - 📊 Carbon impact assessment
    - 🗺️ Interactive visualization
    """)

# ==================== ML MODEL HELPER FUNCTIONS ====================

@st.cache_resource
def load_ml_model():
    """Load trained ML model (cached)"""
    model_path = Path("outputs/models/ground_truth_unet_model.h5")
    if not model_path.exists():
        return None
    
    try:
        detector = DeforestationDetector(input_shape=(256, 256, 3), model_type='unet')
        detector.load_model(str(model_path))
        return detector
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_with_ml_model(before_img, after_img, model):
    """Use ML model to predict deforestation mask"""
    try:
        # Prepare images
        if before_img.shape[:2] != (256, 256):
            before_img = cv2.resize(before_img, (256, 256))
        if after_img.shape[:2] != (256, 256):
            after_img = cv2.resize(after_img, (256, 256))
        
        # Normalize
        before_norm = before_img.astype(np.float32) / 255.0
        after_norm = after_img.astype(np.float32) / 255.0
        
        # Stack for change detection (concat before and after)
        combined = np.concatenate([before_norm, after_norm], axis=-1)  # (256, 256, 6)
        
        # For U-Net trained on single images, predict on after image
        after_input = np.expand_dims(after_norm, axis=0)  # (1, 256, 256, 3)
        
        # Predict
        prediction = model.predict(after_input)  # (1, 256, 256, 1)
        mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
        
        return mask
    except Exception as e:
        st.error(f"ML prediction error: {e}")
        return None

# ==================== FUNCTION DEFINITIONS ====================

def complete_analysis_mode(carbon_density, pixel_area_ha, ndvi_threshold):
    """Complete End-to-End Analysis Pipeline with animations"""
    
    st.markdown("""
    <div class="info-box">
        <h3>🔬 Complete E2E Analysis</h3>
        <p>Full 8-step deforestation detection and carbon impact assessment pipeline</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Region input with animation
    region_name = st.text_input("🌍 Geographic Region", "Amazon Rainforest", key="region_input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📅 Before Image")
        use_sample = st.checkbox("Use sample data", value=True, key="use_sample")
        if not use_sample:
            before_file = st.file_uploader("Upload Before Image", type=['jpg', 'png', 'tif', 'tiff'], key="before_upload")
    
    with col2:
        st.markdown("##### 📅 After Image")
        if not use_sample:
            after_file = st.file_uploader("Upload After Image", type=['jpg', 'png', 'tif', 'tiff'], key="after_upload")
    
    if st.button("🚀 Run Complete Analysis", use_container_width=True, type="primary"):
        
        # Animated progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate or load images
        if use_sample:
            status_text.markdown("### 📸 Generating sample satellite imagery...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            # Create synthetic images
            np.random.seed(42)
            before_img = np.random.randint(50, 150, (512, 512, 3), dtype=np.uint8)
            before_img[:, :, 1] = np.random.randint(100, 200, (512, 512))
            
            after_img = before_img.copy()
            for _ in range(5):
                x, y = np.random.randint(0, 400, 2)
                w, h = np.random.randint(50, 120, 2)
                after_img[y:y+h, x:x+w] = [150, 100, 70]
        else:
            # Load uploaded images
            if 'before_file' not in locals() or 'after_file' not in locals():
                st.error("❌ Please upload both before and after images!")
                return
            
            if before_file is None or after_file is None:
                st.error("❌ Please upload both before and after images!")
                return
            
            status_text.markdown("### 📸 Loading uploaded images...")
            progress_bar.progress(10)
            time.sleep(0.3)
            
            # Convert uploaded files to numpy arrays
            before_bytes = before_file.read()
            after_bytes = after_file.read()
            
            before_img = cv2.imdecode(np.frombuffer(before_bytes, np.uint8), cv2.IMREAD_COLOR)
            after_img = cv2.imdecode(np.frombuffer(after_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if before_img is None or after_img is None:
                st.error("❌ Error: Could not decode images. Please ensure they are valid image files.")
                return
            
            # Convert BGR to RGB
            before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
            after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
        
        # Run analysis with animated progress
        status_text.markdown("### 🔄 Running complete analysis pipeline...")
        progress_bar.progress(20)
        time.sleep(0.3)
        
        # Check if ML model should be used
        use_ml = st.session_state.get('use_ml_model', False)
        ml_model = None
        
        if use_ml:
            status_text.markdown("### 🤖 Loading ML model...")
            ml_model = load_ml_model()
            if ml_model is None:
                st.warning("⚠️ ML model not available, falling back to NDVI-based detection")
                use_ml = False
        
        analyzer = CompleteDeforestationAnalysis(carbon_density, pixel_area_ha)
        
        try:
            # Simulate steps with progress updates
            if use_ml:
                steps = [
                    (30, "📡 Step 1/8: Data Ingestion..."),
                    (40, "🔧 Step 2/8: Pre-Processing..."),
                    (50, "🤖 Step 3/8: ML Model Prediction..."),
                    (65, "🔍 Step 4/8: Change Detection..."),
                    (75, "💚 Step 5/8: Carbon Assessment..."),
                    (85, "📊 Step 6/8: Generating Visualizations..."),
                    (95, "✅ Step 7/8: Finalizing Results..."),
                ]
            else:
                steps = [
                    (30, "📡 Step 1/8: Data Ingestion..."),
                    (40, "🔧 Step 2/8: Pre-Processing..."),
                    (50, "🌿 Step 3/8: NDVI Calculation..."),
                    (60, "🔍 Step 4/8: Change Detection..."),
                    (70, "💚 Step 5/8: Carbon Assessment..."),
                    (85, "📊 Step 6/8: Generating Visualizations..."),
                    (95, "✅ Step 7/8: Finalizing Results..."),
                ]
            
            for prog, msg in steps:
                status_text.markdown(f"### {msg}")
                progress_bar.progress(prog)
                time.sleep(0.4)
            
            results = analyzer.run_complete_workflow(
                before_image=before_img,
                after_image=after_img,
                region_name=region_name
            )
            
            # If ML model enabled, add ML prediction overlay
            if use_ml and ml_model:
                status_text.markdown("### 🧠 Generating ML predictions...")
                ml_mask = predict_with_ml_model(before_img, after_img, ml_model)
                if ml_mask is not None:
                    results['ml_prediction'] = ml_mask
                    # Calculate ML-based metrics
                    ml_deforested_pixels = np.sum(ml_mask)
                    ml_area_ha = ml_deforested_pixels * pixel_area_ha
                    results['ml_metrics'] = {
                        'deforested_pixels': int(ml_deforested_pixels),
                        'deforested_area_ha': float(ml_area_ha),
                        'deforestation_pct': float((ml_deforested_pixels / (256 * 256)) * 100)
                    }
            
            progress_bar.progress(100)
            status_text.success("### ✅ Analysis complete!")
            time.sleep(0.5)
            status_text.empty()
            
            # Display results with animations
            display_complete_results(results)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error during analysis: {str(e)}")


def display_complete_results(results):
    """Display results with enhanced animations"""
    
    st.markdown('<p class="section-header">📊 Analysis Results</p>', unsafe_allow_html=True)
    
    # Extract data
    carbon = results['carbon_impact']
    forest_loss_pct = (carbon['deforested_area_ha'] / (256 * 256 * 0.01)) * 100
    
    # Check if ML predictions available
    has_ml = 'ml_metrics' in results
    
    if has_ml:
        ml_metrics = results['ml_metrics']
        st.info("🤖 **ML Model Active**: Showing both traditional NDVI and ML predictions")
    
    # Animated metric cards
    if has_ml:
        col1, col2, col3, col4, col5 = st.columns(5)
    else:
        col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h2>{forest_loss_pct:.2f}%</h2>
            <p>Forest Loss</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h2>{carbon['deforested_area_ha']:.1f}</h2>
            <p>Hectares Lost</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h2>{carbon['co2_emissions_tons']:,.0f}</h2>
            <p>Tons CO₂</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <h2>{carbon['car_equivalents']:,.0f}</h2>
            <p>Cars/Year (NDVI)</p>
        </div>
        """, unsafe_allow_html=True)
    
    if has_ml:
        with col5:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <h2>{ml_metrics['deforestation_pct']:.1f}%</h2>
                <p>ML Prediction</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Visualizations with cards
    st.markdown('<p class="section-header">🗺️ Visual Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("##### 📸 Before vs After")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        before_img = results['preprocessed_images']['before']
        after_img = results['preprocessed_images']['after']
        
        ax1.imshow(before_img)
        ax1.set_title('Before', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        
        ax2.imshow(after_img)
        ax2.set_title('After', fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("##### 🌿 NDVI Analysis")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ndvi_before = results['ndvi_analysis']['ndvi_before']
        ndvi_after = results['ndvi_analysis']['ndvi_after']
        
        im1 = ax1.imshow(ndvi_before, cmap='RdYlGn', vmin=-1, vmax=1)
        ax1.set_title('NDVI Before', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        im2 = ax2.imshow(ndvi_after, cmap='RdYlGn', vmin=-1, vmax=1)
        ax2.set_title('NDVI After', fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Deforestation mask
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    if has_ml:
        st.markdown("##### 🚨 Deforestation Detection (NDVI vs ML)")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # NDVI-based detection
        deforestation_mask = results['change_detection']['deforestation_mask']
        ax1.imshow(after_img)
        red_overlay = np.zeros_like(after_img)
        red_overlay[deforestation_mask > 0] = [255, 0, 0]
        ax1.imshow(red_overlay, alpha=0.4)
        ax1.set_title('NDVI-Based Detection', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # ML-based detection
        ml_mask = results['ml_prediction']
        ax2.imshow(after_img)
        blue_overlay = np.zeros_like(after_img)
        blue_overlay[ml_mask > 0] = [0, 100, 255]
        ax2.imshow(blue_overlay, alpha=0.4)
        ax2.set_title('ML Model Prediction', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.markdown("##### 🚨 Deforestation Detection")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        deforestation_mask = results['change_detection']['deforestation_mask']
        
        # Show after image with red overlay
        ax.imshow(after_img)
        red_overlay = np.zeros_like(after_img)
        red_overlay[deforestation_mask > 0] = [255, 0, 0]
        ax.imshow(red_overlay, alpha=0.4)
        
        ax.set_title('Deforested Areas (Red Overlay)', fontsize=16, fontweight='bold', pad=15)
        ax.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Carbon Impact Details
    st.markdown('<p class="section-header">💚 Carbon Impact Details</p>', unsafe_allow_html=True)
    
    # Get tree equivalents (handle both old and new key names)
    tree_equivalents = carbon.get('tree_equivalents', carbon.get('trees_needed', 0))
    
    carbon_df = pd.DataFrame({
        'Metric': [
            'Deforested Area',
            'Carbon Stock Lost',
            'CO₂ Emissions',
            'Car Equivalent (1 year)',
            'Trees Needed to Offset'
        ],
        'Value': [
            f"{carbon['deforested_area_ha']:.2f} ha",
            f"{carbon['carbon_loss_tons']:,.2f} tons C",
            f"{carbon['co2_emissions_tons']:,.2f} tons CO₂",
            f"{carbon['car_equivalents']:,.0f} cars",
            f"{tree_equivalents:,.0f} trees"
        ]
    })
    
    st.dataframe(carbon_df, use_container_width=True, hide_index=True)
    
    st.success(f"✅ Analysis completed successfully! All {results['steps_completed'].__len__()} steps executed.")


def quick_demo_mode(carbon_density, pixel_area_ha, ndvi_threshold):
    """Quick demo with synthetic data and animations"""
    
    st.markdown("""
    <div class="info-box">
        <h3>⚡ Quick Demo</h3>
        <p>Fast analysis with synthetic sample data</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎬 Run Quick Demo", use_container_width=True, type="primary"):
        
        with st.spinner("🔄 Generating demo analysis..."):
            time.sleep(1)
            
            # Generate synthetic results
            np.random.seed(42)
            
            area_deforested = np.random.uniform(50, 150)
            carbon_loss = area_deforested * carbon_density
            co2_emissions = carbon_loss * 3.67
            car_equiv = co2_emissions / 4.6
            
            st.markdown('<p class="section-header">📊 Demo Results</p>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card pulse">
                    <h2>{area_deforested:.1f}</h2>
                    <p>Hectares</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card pulse">
                    <h2>{carbon_loss:,.0f}</h2>
                    <p>Tons Carbon</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card pulse">
                    <h2>{co2_emissions:,.0f}</h2>
                    <p>Tons CO₂</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card pulse">
                    <h2>{car_equiv:,.0f}</h2>
                    <p>Cars/Year</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Interactive Plotly chart
            st.markdown('<p class="section-header">📊 Impact Visualization</p>', unsafe_allow_html=True)
            
            fig = go.Figure()
            
            categories = ['Carbon Loss', 'CO₂ Emissions', 'Car Equivalent']
            values = [carbon_loss/100, co2_emissions/100, car_equiv]
            colors = ['#667eea', '#f5576c', '#43e97b']
            
            fig.add_trace(go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[f'{v:.0f}' for v in values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Environmental Impact Metrics",
                yaxis_title="Value (scaled)",
                height=400,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Demo analysis completed!")


def upload_mode(carbon_density, pixel_area_ha, ndvi_threshold):
    """Upload custom images mode with animations"""
    
    st.markdown("""
    <div class="info-box">
        <h3>📤 Upload Your Images</h3>
        <p>Analyze your own satellite imagery</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("📋 **Instructions:** Upload before and after satellite images in RGB or multispectral format")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📅 Before Image")
        before_file = st.file_uploader("Upload Before Image", type=['jpg', 'png', 'tif', 'tiff'], key="before_custom")
        
        if before_file:
            st.success(f"✅ Uploaded: {before_file.name} ({before_file.size / 1024:.1f} KB)")
        
    with col2:
        st.markdown("##### 📅 After Image")
        after_file = st.file_uploader("Upload After Image", type=['jpg', 'png', 'tif', 'tiff'], key="after_custom")
        
        if after_file:
            st.success(f"✅ Uploaded: {after_file.name} ({after_file.size / 1024:.1f} KB)")
    
    if before_file and after_file:
        st.markdown("---")
        
        region_name = st.text_input("Region Name (optional)", "Uploaded Region", help="Enter a name for this analysis region")
        
        if st.button("🔍 Analyze Images", use_container_width=True, type="primary"):
            try:
                with st.spinner("Processing uploaded images..."):
                    # Load images
                    before_bytes = before_file.read()
                    after_bytes = after_file.read()
                    
                    before_img = cv2.imdecode(np.frombuffer(before_bytes, np.uint8), cv2.IMREAD_COLOR)
                    after_img = cv2.imdecode(np.frombuffer(after_bytes, np.uint8), cv2.IMREAD_COLOR)
                    
                    if before_img is None or after_img is None:
                        st.error("❌ Error: Could not decode images.")
                        return
                    
                    before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
                    after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
                    
                    # Show images
                    st.markdown("##### 📸 Uploaded Images")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(before_img, caption="Before Image", use_container_width=True)
                    with col2:
                        st.image(after_img, caption="After Image", use_container_width=True)
                    
                    # Run analysis
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(0, 101, 20):
                        progress_bar.progress(i)
                        status_text.text(f"Processing... {i}%")
                        time.sleep(0.3)
                    
                    analyzer = CompleteDeforestationAnalysis(carbon_density, pixel_area_ha)
                    results = analyzer.run_complete_workflow(
                        region_name=region_name,
                        before_image=before_img,
                        after_image=after_img
                    )
                    
                    progress_bar.progress(100)
                    status_text.success("✅ Analysis complete!")
                    time.sleep(0.5)
                    status_text.empty()
                    
                    st.markdown("---")
                    display_complete_results(results)
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
    else:
        st.info("👆 Please upload both before and after images to proceed with analysis.")


def real_datasets_mode():
    """Real datasets browser with animations"""
    
    st.markdown("""
    <div class="info-box">
        <h3>📡 Real Datasets</h3>
        <p>Browse and analyze real Kaggle datasets</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🌎 Amazon Dataset\n10GB Sentinel-2", use_container_width=True):
            st.session_state['dataset'] = 'amazon'
    
    with col2:
        if st.button("�🇷 Brazil Competition\n4,043 Files", use_container_width=True):
            st.session_state['dataset'] = 'competition'
    
    if 'dataset' in st.session_state:
        dataset_type = st.session_state['dataset']
        
        with st.spinner(f"Loading {dataset_type} dataset..."):
            loader = DeforestationDataLoader(str(Path.cwd() / 'data'))
            
            if dataset_type == 'amazon':
                data = loader.load_kaggle_dataset(str(AMAZON_DATASET_PATH), 'amazon')
                
                st.markdown("##### 🌎 Amazon Deforestation Dataset")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📸 Images", data.get('image_count', 0))
                with col2:
                    st.metric("🎭 Masks", len(data.get('mask_paths', [])))
                with col3:
                    st.metric("💾 Size", "10.1 GB")
                
                if data.get('image_paths'):
                    st.success(f"✅ Found {data['image_count']} Sentinel-2 satellite images")
                    
                    # Add analysis functionality
                    st.markdown("---")
                    st.markdown("##### 🔬 Analyze Real Amazon Images")
                    
                    image_paths = data['image_paths']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        before_idx = st.selectbox("Select Before Image", 
                                                  range(len(image_paths)), 
                                                  format_func=lambda x: f"Image {x+1}")
                    with col2:
                        after_idx = st.selectbox("Select After Image", 
                                                range(len(image_paths)), 
                                                index=min(1, len(image_paths)-1),
                                                format_func=lambda x: f"Image {x+1}")
                    
                    if st.button("🚀 Analyze Selected Images", use_container_width=True, type="primary"):
                        try:
                            with st.spinner("Loading and analyzing real Amazon dataset images..."):
                                import rasterio
                                from rasterio.plot import reshape_as_image
                                
                                before_path = str(image_paths[before_idx])
                                after_path = str(image_paths[after_idx])
                                
                                st.info(f"📂 Loading:\n- Before: {Path(before_path).name}\n- After: {Path(after_path).name}")
                                
                                # Load with rasterio
                                with rasterio.open(before_path) as src:
                                    before_data = src.read()
                                    if before_data.shape[0] >= 3:
                                        before_img = np.stack([before_data[2], before_data[1], before_data[0]], axis=-1)
                                    else:
                                        before_img = reshape_as_image(before_data)
                                    
                                    if before_img.max() > 255:
                                        before_img = ((before_img - before_img.min()) / (before_img.max() - before_img.min()) * 255).astype(np.uint8)
                                    else:
                                        before_img = before_img.astype(np.uint8)
                                
                                with rasterio.open(after_path) as src:
                                    after_data = src.read()
                                    if after_data.shape[0] >= 3:
                                        after_img = np.stack([after_data[2], after_data[1], after_data[0]], axis=-1)
                                    else:
                                        after_img = reshape_as_image(after_data)
                                    
                                    if after_img.max() > 255:
                                        after_img = ((after_img - after_img.min()) / (after_img.max() - after_img.min()) * 255).astype(np.uint8)
                                    else:
                                        after_img = after_img.astype(np.uint8)
                                
                                # Show images
                                st.markdown("##### 📸 Loaded Images")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(before_img, caption=f"Before - Image {before_idx+1}", use_container_width=True)
                                with col2:
                                    st.image(after_img, caption=f"After - Image {after_idx+1}", use_container_width=True)
                                
                                # Run analysis
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                for i in range(0, 101, 25):
                                    progress_bar.progress(i)
                                    status_text.text(f"Analyzing... {i}%")
                                    time.sleep(0.3)
                                
                                carbon_density = st.session_state.get('carbon_density', 190.0)
                                pixel_area_ha = st.session_state.get('pixel_area_ha', 0.01)
                                
                                analyzer = CompleteDeforestationAnalysis(carbon_density, pixel_area_ha)
                                results = analyzer.run_complete_workflow(
                                    before_image=before_img,
                                    after_image=after_img,
                                    region_name="Amazon Rainforest (Real Dataset)"
                                )
                                
                                progress_bar.progress(100)
                                status_text.success("✅ Analysis complete!")
                                time.sleep(0.5)
                                status_text.empty()
                                
                                st.markdown("---")
                                display_complete_results(results)
                                
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
            
            elif dataset_type == 'competition':
                data = loader.load_kaggle_dataset(str(COMPETITION_DATASET_PATH), 'competition')
                
                st.markdown("##### �🇷 Brazil Deforestation Competition Dataset")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📦 Training", data.get('train_count', 0))
                with col2:
                    st.metric("📋 Test", data.get('test_count', 0))
                with col3:
                    st.metric("📝 Metadata", len(data.get('train_metadata', [])))
                
                st.success(f"✅ Loaded {data.get('train_count', 0)} multi-spectral arrays (512x512x13)")
                
                # Add analysis functionality for competition dataset
                if data.get('train_paths') and len(data.get('train_paths', [])) > 0:
                    st.markdown("---")
                    st.markdown("##### 🔬 Analyze Competition Data")
                    
                    train_paths = data['train_paths']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        before_idx = st.selectbox("Select Before Sample", 
                                                  range(min(len(train_paths), 50)), 
                                                  format_func=lambda x: f"Sample {x+1}")
                    with col2:
                        after_idx = st.selectbox("Select After Sample", 
                                                range(min(len(train_paths), 50)), 
                                                index=min(1, len(train_paths)-1),
                                                format_func=lambda x: f"Sample {x+1}")
                    
                    if st.button("🚀 Analyze Selected Samples", use_container_width=True, type="primary"):
                        try:
                            with st.spinner("Loading and analyzing Brazil competition data..."):
                                # Load .npy files
                                before_path = str(train_paths[before_idx])
                                after_path = str(train_paths[after_idx])
                                
                                st.info(f"📂 Loading:\n- Before: Sample {before_idx+1}\n- After: Sample {after_idx+1}")
                                
                                # Load numpy arrays
                                before_data = np.load(before_path)
                                after_data = np.load(after_path)
                                
                                # Competition data is 512x512x13 (multi-spectral)
                                # Extract RGB bands (assume bands 4,3,2 for RGB)
                                if before_data.shape[-1] >= 3:
                                    # Take first 3 bands as RGB
                                    before_img = before_data[:, :, :3]
                                    after_img = after_data[:, :, :3]
                                else:
                                    st.error("❌ Data format not supported. Expected multi-spectral data.")
                                    return
                                
                                # Normalize to 0-255
                                if before_img.max() > 255:
                                    before_img = ((before_img - before_img.min()) / (before_img.max() - before_img.min()) * 255).astype(np.uint8)
                                else:
                                    before_img = before_img.astype(np.uint8)
                                
                                if after_img.max() > 255:
                                    after_img = ((after_img - after_img.min()) / (after_img.max() - after_img.min()) * 255).astype(np.uint8)
                                else:
                                    after_img = after_img.astype(np.uint8)
                                
                                # Show images
                                st.markdown("##### 📸 Loaded Samples")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(before_img, caption=f"Before - Sample {before_idx+1}", use_container_width=True)
                                with col2:
                                    st.image(after_img, caption=f"After - Sample {after_idx+1}", use_container_width=True)
                                
                                # Run analysis
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                for i in range(0, 101, 25):
                                    progress_bar.progress(i)
                                    status_text.text(f"Analyzing... {i}%")
                                    time.sleep(0.3)
                                
                                carbon_density = st.session_state.get('carbon_density', 190.0)
                                pixel_area_ha = st.session_state.get('pixel_area_ha', 0.01)
                                
                                analyzer = CompleteDeforestationAnalysis(carbon_density, pixel_area_ha)
                                results = analyzer.run_complete_workflow(
                                    before_image=before_img,
                                    after_image=after_img,
                                    region_name="Brazil (Competition Dataset)"
                                )
                                
                                progress_bar.progress(100)
                                status_text.success("✅ Analysis complete!")
                                time.sleep(0.5)
                                status_text.empty()
                                
                                st.markdown("---")
                                display_complete_results(results)
                                
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            st.exception(e)


# ==================== MAIN CONTENT ROUTING ====================
if "Complete E2E" in analysis_mode:
    complete_analysis_mode(carbon_density, pixel_area_ha, ndvi_threshold)
elif "Quick Demo" in analysis_mode:
    quick_demo_mode(carbon_density, pixel_area_ha, ndvi_threshold)
elif "Upload" in analysis_mode:
    upload_mode(carbon_density, pixel_area_ha, ndvi_threshold)
else:
    real_datasets_mode()


# Save carbon parameters to session state
st.session_state['carbon_density'] = carbon_density
st.session_state['pixel_area_ha'] = pixel_area_ha


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p style="font-size: 0.9rem;">🌍 <strong>EcoVerse</strong> - Protecting Forests with AI | Built with ❤️ for the Planet</p>
    <p style="font-size: 0.8rem;">© 2026 EcoVerse Team | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
