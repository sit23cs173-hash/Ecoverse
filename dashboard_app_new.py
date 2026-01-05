"""
🌍 ECOVERSE - Professional Deforestation Detection System
Modern Dashboard with Complete E2E Workflow Integration

Run: streamlit run dashboard_app_new.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import modules
from complete_analysis_workflow import CompleteDeforestationAnalysis
from analysis.vegetation_analysis import VegetationIndexCalculator
from analysis.carbon_impact import CarbonImpactCalculator
from data.data_loader import (DeforestationDataLoader, AMAZON_DATASET_PATH, 
                              COMPETITION_DATASET_PATH, TIMESERIES_DATASET_PATH)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="EcoVerse | AI Deforestation Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Content container */
    .block-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: white;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card h2 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-card p {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    
    .info-box h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
    }
    
    /* Section headers */
    .section-header {
        color: #667eea;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/info messages */
    .stSuccess, .stInfo {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function"""
    
    # ==================== HEADER ====================
    st.markdown("""
    <div class="main-header">
        <h1>🌍 ECOVERSE</h1>
        <p>AI-Powered Deforestation Detection & Climate Impact Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        st.markdown("---")
        
        # Analysis Mode
        st.markdown("#### 📊 Analysis Mode")
        analysis_mode = st.selectbox(
            "Choose Mode",
            [
                "🔬 Complete E2E Analysis",
                "🎯 Quick Demo",
                "📤 Upload Images",
                "📡 Real Datasets",
                "📈 Time-Series"
            ],
            key="mode_selector"
        )
        
        st.markdown("---")
        
        # Parameters
        st.markdown("#### 🌲 Carbon Parameters")
        carbon_density = st.slider(
            "Carbon Density (tons C/ha)",
            50.0, 300.0, 190.0, 10.0,
            help="Average carbon stock in forest"
        )
        
        pixel_area_ha = st.number_input(
            "Pixel Area (hectares)",
            0.001, 1.0, 0.01, 0.001,
            help="Area per pixel"
        )
        
        st.markdown("---")
        
        # Detection Parameters
        st.markdown("#### 🎯 Detection Settings")
        ndvi_threshold = st.slider(
            "NDVI Threshold",
            -0.5, 0.0, -0.15, 0.01,
            help="Threshold for deforestation detection"
        )
        
        st.markdown("---")
        st.markdown("##### 📘 About")
        st.info("EcoVerse uses satellite imagery and AI to detect deforestation and assess environmental impact.")
    
    # ==================== MAIN CONTENT ====================
    if "Complete E2E" in analysis_mode:
        complete_analysis_mode(carbon_density, pixel_area_ha, ndvi_threshold)
    elif "Quick Demo" in analysis_mode:
        quick_demo_mode(carbon_density, pixel_area_ha, ndvi_threshold)
    elif "Upload" in analysis_mode:
        upload_mode(carbon_density, pixel_area_ha, ndvi_threshold)
    elif "Real Datasets" in analysis_mode:
        real_datasets_mode()
    else:
        timeseries_mode()


def complete_analysis_mode(carbon_density, pixel_area_ha, ndvi_threshold):
    """Complete End-to-End Analysis Pipeline"""
    
    st.markdown('<div class="info-box"><h3>🔬 Complete E2E Analysis</h3><p>Full 8-step deforestation detection and carbon impact assessment pipeline</p></div>', unsafe_allow_html=True)
    
    # Region input
    region_name = st.text_input("🌍 Geographic Region", "Amazon Rainforest", key="region_input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📅 Before Image")
        use_sample = st.checkbox("Use sample data", value=True, key="use_sample")
        if not use_sample:
            before_file = st.file_uploader("Upload Before Image", type=['jpg', 'png', 'tif'], key="before_upload")
    
    with col2:
        st.markdown("##### 📅 After Image")
        if not use_sample:
            after_file = st.file_uploader("Upload After Image", type=['jpg', 'png', 'tif'], key="after_upload")
    
    if st.button("🚀 Run Complete Analysis", use_container_width=True, type="primary"):
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate or load images
        if use_sample:
            status_text.text("📸 Generating sample satellite imagery...")
            progress_bar.progress(10)
            
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
            
            status_text.text("📸 Loading uploaded images...")
            progress_bar.progress(10)
            
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
        
        # Run analysis
        status_text.text("🔄 Running complete analysis pipeline...")
        progress_bar.progress(20)
        
        analyzer = CompleteDeforestationAnalysis(carbon_density, pixel_area_ha)
        
        try:
            results = analyzer.run_complete_workflow(
                before_image=before_img,
                after_image=after_img,
                region_name=region_name
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display results
            display_complete_results(results)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)


def display_complete_results(results):
    """Display comprehensive analysis results"""
    
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Analysis Results</p>', unsafe_allow_html=True)
    
    # Summary metrics
    summary = results['summary']
    carbon = results['carbon_impact']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{summary['forest_loss_percentage']:.2f}%</h2>
            <p>Forest Loss</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h2>{carbon['deforested_area_ha']:.1f}</h2>
            <p>Hectares Lost</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h2>{carbon['co2_emissions_tons']:,.0f}</h2>
            <p>Tons CO₂</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h2>{carbon['car_equivalents']:,.0f}</h2>
            <p>Cars/Year</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    st.markdown('<p class="section-header">🗺️ Visual Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📸 Before vs After")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        before_img = results['preprocessed_images']['before']
        after_img = results['preprocessed_images']['after']
        
        ax1.imshow(before_img)
        ax1.set_title('Before', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(after_img)
        ax2.set_title('After', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("##### 🌿 NDVI Analysis")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ndvi_before = results['ndvi_analysis']['ndvi_before']
        ndvi_after = results['ndvi_analysis']['ndvi_after']
        
        im1 = ax1.imshow(ndvi_before, cmap='RdYlGn', vmin=-1, vmax=1)
        ax1.set_title('NDVI Before', fontsize=14, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        im2 = ax2.imshow(ndvi_after, cmap='RdYlGn', vmin=-1, vmax=1)
        ax2.set_title('NDVI After', fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Deforestation mask
    st.markdown("##### 🚨 Deforestation Detection")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    deforestation_mask = results['change_detection']['deforestation_mask']
    
    # Show after image with red overlay
    ax.imshow(after_img)
    red_overlay = np.zeros_like(after_img)
    red_overlay[:, :, 0] = deforestation_mask * 255
    ax.imshow(red_overlay, alpha=0.5)
    ax.set_title('Detected Deforestation Areas (Red Overlay)', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    st.pyplot(fig)
    
    # Carbon impact breakdown
    st.markdown('<p class="section-header">💚 Carbon Impact Details</p>', unsafe_allow_html=True)
    
    impact_data = {
        'Metric': ['Deforested Area', 'Carbon Loss', 'CO₂ Emissions', 'Car Equivalent', 'Trees to Offset'],
        'Value': [
            f"{carbon['deforested_area_ha']:.2f} ha",
            f"{carbon['carbon_loss_tons']:,.2f} tons",
            f"{carbon['co2_emissions_tons']:,.2f} tons",
            f"{carbon['car_equivalents']:,.0f} cars/year",
            f"{carbon['tree_equivalents']:,.0f} trees"
        ]
    }
    
    df = pd.DataFrame(impact_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Pipeline status
    st.success(f"✅ Analysis completed successfully! All {len(results['steps_completed'])} steps executed.")


def quick_demo_mode(carbon_density, pixel_area_ha, ndvi_threshold):
    """Quick demonstration mode"""
    
    st.markdown('<div class="info-box"><h3>🎯 Quick Demo Mode</h3><p>Fast synthetic analysis for demonstration purposes</p></div>', unsafe_allow_html=True)
    
    if st.button("⚡ Generate Quick Demo", use_container_width=True, type="primary"):
        
        with st.spinner("Generating demo analysis..."):
            # Create synthetic data
            np.random.seed(42)
            
            # Simulate results
            deforested_ha = np.random.uniform(50, 150)
            carbon_loss = deforested_ha * carbon_density
            co2_emissions = carbon_loss * 3.67
            car_equiv = co2_emissions / 4.6
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🌲 Deforested Area", f"{deforested_ha:.1f} ha")
            
            with col2:
                st.metric("💨 Carbon Loss", f"{carbon_loss:,.0f} tons")
            
            with col3:
                st.metric("☁️ CO₂ Emissions", f"{co2_emissions:,.0f} tons")
            
            with col4:
                st.metric("🚗 Car Equivalent", f"{car_equiv:,.0f} cars")
            
            # Simple chart
            st.markdown("##### 📊 Impact Visualization")
            
            fig = go.Figure()
            
            categories = ['Carbon Loss', 'CO₂ Emissions', 'Car Equivalent']
            values = [carbon_loss/100, co2_emissions/100, car_equiv]
            
            fig.add_trace(go.Bar(
                x=categories,
                y=values,
                marker_color=['#667eea', '#f5576c', '#43e97b'],
                text=[f'{v:.0f}' for v in values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Environmental Impact Metrics",
                yaxis_title="Value (scaled)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Demo analysis completed!")


def upload_mode(carbon_density, pixel_area_ha, ndvi_threshold):
    """Upload custom images mode"""
    
    st.markdown('<div class="info-box"><h3>📤 Upload Your Images</h3><p>Analyze your own satellite imagery</p></div>', unsafe_allow_html=True)
    
    st.info("📋 **Instructions:** Upload before and after satellite images in RGB or multispectral format (JPG, PNG, or TIF)")
    
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
        
        # Region name (optional)
        region_name = st.text_input("Region Name (optional)", "Uploaded Region", help="Enter a name for this analysis region")
        
        if st.button("🔍 Analyze Images", use_container_width=True, type="primary"):
            try:
                with st.spinner("Processing uploaded images..."):
                    # Load images from uploaded files
                    before_bytes = before_file.read()
                    after_bytes = after_file.read()
                    
                    # Convert to numpy arrays
                    before_img = cv2.imdecode(np.frombuffer(before_bytes, np.uint8), cv2.IMREAD_COLOR)
                    after_img = cv2.imdecode(np.frombuffer(after_bytes, np.uint8), cv2.IMREAD_COLOR)
                    
                    if before_img is None or after_img is None:
                        st.error("❌ Error: Could not decode images. Please ensure they are valid image files.")
                        return
                    
                    # Convert BGR to RGB for proper display
                    before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
                    after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
                    
                    # Show uploaded images
                    st.markdown("##### 📸 Uploaded Images")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(before_img, caption="Before Image", use_container_width=True)
                    with col2:
                        st.image(after_img, caption="After Image", use_container_width=True)
                    
                    # Create analyzer
                    analyzer = CompleteDeforestationAnalysis(carbon_density, pixel_area_ha)
                    
                    # Run complete workflow
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Step 1/8: Processing user input...")
                    progress_bar.progress(12)
                    
                    status_text.text("Step 2/8: Loading images...")
                    progress_bar.progress(25)
                    
                    status_text.text("Step 3/8: Preprocessing images...")
                    progress_bar.progress(37)
                    
                    # Run workflow with uploaded images
                    results = analyzer.run_complete_workflow(
                        region_name=region_name,
                        before_image=before_img,
                        after_image=after_img
                    )
                    
                    status_text.text("Step 8/8: Generating visualizations...")
                    progress_bar.progress(100)
                    
                    status_text.success("✅ Analysis complete!")
                    
                    # Display results
                    st.markdown("---")
                    display_complete_results(results)
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)
    else:
        st.info("👆 Please upload both before and after images to proceed with analysis.")


def real_datasets_mode():
    """Real datasets browser"""
    
    st.markdown('<div class="info-box"><h3>📡 Real Datasets</h3><p>Browse and analyze real Kaggle datasets</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌎 Amazon Dataset\n10GB Sentinel-2", use_container_width=True):
            st.session_state['dataset'] = 'amazon'
    
    with col2:
        if st.button("🏆 Competition\n4,043 Files", use_container_width=True):
            st.session_state['dataset'] = 'competition'
    
    with col3:
        if st.button("📊 Time-Series\n1999-2019", use_container_width=True):
            st.session_state['dataset'] = 'timeseries'
    
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
                    
                    # Add analysis button
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
                                # Load images with multiple methods
                                before_path = str(image_paths[before_idx])
                                after_path = str(image_paths[after_idx])
                                
                                st.info(f"📂 Loading:\n- Before: {Path(before_path).name}\n- After: {Path(after_path).name}")
                                
                                # Load GeoTIFF files using rasterio (for Sentinel-2 satellite imagery)
                                try:
                                    import rasterio
                                    from rasterio.plot import reshape_as_image
                                    
                                    # Load before image
                                    with rasterio.open(before_path) as src:
                                        before_data = src.read()
                                        # If multi-band, take first 3 bands (RGB)
                                        if before_data.shape[0] >= 3:
                                            before_img = np.stack([before_data[2], before_data[1], before_data[0]], axis=-1)
                                        else:
                                            before_img = reshape_as_image(before_data)
                                        
                                        # Normalize to 0-255 if needed
                                        if before_img.max() > 255:
                                            before_img = ((before_img - before_img.min()) / (before_img.max() - before_img.min()) * 255).astype(np.uint8)
                                        else:
                                            before_img = before_img.astype(np.uint8)
                                    
                                    # Load after image
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
                                    
                                except Exception as e:
                                    st.warning(f"Rasterio failed ({str(e)}), trying OpenCV...")
                                    # Fallback to OpenCV
                                    before_img = cv2.imread(before_path)
                                    after_img = cv2.imread(after_path)
                                
                                if before_img is None or after_img is None:
                                    st.error(f"❌ Could not load images. Debug info:")
                                    st.code(f"Before path: {before_path}\nBefore loaded: {before_img is not None}\n"
                                           f"After path: {after_path}\nAfter loaded: {after_img is not None}")
                                    return
                                
                                # Convert BGR to RGB if needed
                                if len(before_img.shape) == 3 and before_img.shape[2] == 3:
                                    # Check if it's already RGB or BGR
                                    if before_img.dtype == np.uint8:
                                        before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
                                
                                if len(after_img.shape) == 3 and after_img.shape[2] == 3:
                                    if after_img.dtype == np.uint8:
                                        after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
                                
                                # Show loaded images
                                st.markdown("##### 📸 Loaded Images")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(before_img, caption=f"Before - Image {before_idx+1}", use_container_width=True)
                                with col2:
                                    st.image(after_img, caption=f"After - Image {after_idx+1}", use_container_width=True)
                                
                                # Run analysis
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                status_text.text("🔄 Running analysis on real dataset...")
                                progress_bar.progress(20)
                                
                                # Get carbon parameters from sidebar
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
                                
                                # Display results
                                st.markdown("---")
                                display_complete_results(results)
                                
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            st.exception(e)
            
            elif dataset_type == 'competition':
                data = loader.load_kaggle_dataset(str(COMPETITION_DATASET_PATH), 'competition')
                
                st.markdown("##### 🏆 Competition Dataset")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📦 Training", data.get('train_count', 0))
                with col2:
                    st.metric("📋 Test", data.get('test_count', 0))
                with col3:
                    st.metric("📝 Metadata", len(data.get('train_metadata', [])))
                
                st.success(f"✅ Loaded {data.get('train_count', 0)} multi-spectral arrays (512x512x13)")
            
            else:
                data = loader.load_kaggle_dataset(str(TIMESERIES_DATASET_PATH), 'timeseries')
                
                st.markdown("##### 📊 Time-Series Brazil (1999-2019)")
                
                if 'timeseries' in data:
                    df = data['timeseries']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📋 Records", len(df))
                    with col2:
                        years = df['year'].unique() if 'year' in df.columns else []
                        st.metric("📅 Years", len(years))
                    with col3:
                        states = df['state'].unique() if 'state' in df.columns else []
                        st.metric("🗺️ States", len(states))
                    
                    st.dataframe(df.head(50), use_container_width=True)


def timeseries_mode():
    """Time-series analysis mode"""
    
    st.markdown('<div class="info-box"><h3>📈 Time-Series Analysis</h3><p>Historical deforestation trends and forecasting</p></div>', unsafe_allow_html=True)
    
    # Load time-series data
    loader = DeforestationDataLoader(str(Path.cwd() / 'data'))
    data = loader.load_kaggle_dataset(str(TIMESERIES_DATASET_PATH), 'timeseries')
    
    if 'timeseries' in data:
        df = data['timeseries']
        
        st.markdown("##### 🔥 Brazilian Amazon Fire Incidents (1999-2019)")
        
        if 'year' in df.columns:
            yearly = df.groupby('year').size().reset_index(name='incidents')
            
            fig = px.line(
                yearly, 
                x='year', 
                y='incidents',
                title='Fire Incidents Over Time',
                markers=True
            )
            fig.update_traces(line_color='#e74c3c', marker=dict(size=10))
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Total Incidents", f"{len(df):,}")
            
            with col2:
                avg_per_year = len(df) / len(yearly)
                st.metric("📈 Avg/Year", f"{avg_per_year:.0f}")
            
            with col3:
                peak_year = yearly.loc[yearly['incidents'].idxmax(), 'year']
                st.metric("🔥 Peak Year", int(peak_year))
            
            st.dataframe(df.head(100), use_container_width=True)
    else:
        st.warning("⚠️ Time-series data not available. Please check dataset configuration.")


# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
