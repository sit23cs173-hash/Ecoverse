# 🚀 QUICK START GUIDE
# Deforestation Detection & Carbon Impact Assessment System

## ⚡ 5-Minute Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Demo Pipeline
```bash
python main_pipeline.py
```

This will:
- Generate synthetic deforestation data
- Run complete analysis (NDVI, detection, carbon impact)
- Create visualizations in `outputs/figures/`
- Generate carbon impact report

### Step 3: Launch Interactive Dashboard
```bash
streamlit run dashboard_app.py
```

Then open your browser to `http://localhost:8501`

---

## 📖 Detailed Usage Guide

### Option 1: Complete Pipeline (Recommended for First Run)

```bash
python main_pipeline.py
```

**What it does:**
1. Creates 50 synthetic before/after image pairs
2. Preprocesses and splits data (train/val/test)
3. Calculates NDVI for vegetation analysis
4. Detects deforestation using NDVI threshold method
5. Estimates carbon loss and CO2 emissions
6. Generates comprehensive visualizations
7. Exports results to JSON and text reports

**Output locations:**
- `outputs/figures/ndvi_comparison.png` - NDVI before/after/change
- `outputs/figures/detection_result.png` - Detection visualization
- `outputs/figures/dashboard_summary.png` - Statistics dashboard
- `outputs/carbon_impact_report.txt` - Detailed carbon report
- `outputs/pipeline_results.json` - All metrics in JSON

---

### Option 2: Interactive Dashboard

```bash
streamlit run dashboard_app.py
```

**Features:**
- **Demo Mode**: Analyze synthetic deforestation scenarios
- **Upload Mode**: Upload your own satellite images (PNG, JPG, TIFF)
- **Time-Series Mode**: Analyze deforestation trends over time
- **Real-time Parameter Adjustment**: Change carbon density, thresholds
- **Interactive Visualizations**: Explore results interactively

**Dashboard Sections:**
1. Before/After Image Comparison
2. NDVI Analysis Maps
3. Carbon Impact Metrics
4. Detection Performance Statistics
5. Time-Series Trends (if available)

---

### Option 3: Use Individual Modules

#### Calculate NDVI
```python
from analysis.vegetation_analysis import VegetationIndexCalculator
import cv2

# Load satellite image
image = cv2.imread('path/to/satellite_image.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Calculate NDVI
veg_calc = VegetationIndexCalculator()
ndvi = veg_calc.calculate_ndvi(image)

# Get statistics
from analysis.vegetation_analysis import generate_ndvi_statistics
stats = generate_ndvi_statistics(ndvi)
print(stats)
```

#### Detect Deforestation
```python
from models.deforestation_model import NDVIBasedDetector
from analysis.vegetation_analysis import VegetationIndexCalculator

# Load before and after images
before_img = cv2.imread('before.png')
after_img = cv2.imread('after.png')

# Calculate NDVI
veg_calc = VegetationIndexCalculator()
ndvi_before = veg_calc.calculate_ndvi(before_img)
ndvi_after = veg_calc.calculate_ndvi(after_img)

# Detect deforestation
detector = NDVIBasedDetector(ndvi_threshold=-0.15)
mask = detector.detect_deforestation(ndvi_before, ndvi_after)

# Evaluate if you have ground truth
if ground_truth_mask is not None:
    metrics = detector.evaluate(ground_truth_mask, mask)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
```

#### Estimate Carbon Impact
```python
from analysis.carbon_impact import CarbonImpactCalculator

# Initialize calculator
carbon_calc = CarbonImpactCalculator(
    carbon_density=190.0,  # Amazon rainforest
    pixel_area_ha=0.01     # Sentinel-2 resolution
)

# Calculate from deforestation mask
impact = carbon_calc.calculate_carbon_from_mask(deforestation_mask)

# Print results
print(f"Deforested Area: {impact['deforested_area_ha']:.2f} hectares")
print(f"Carbon Loss: {impact['carbon_loss_tons']:.2f} tons")
print(f"CO2 Emissions: {impact['co2_emissions_tons']:.2f} tons")
print(f"Equivalent to {impact['equivalent_cars_year']:.0f} cars/year")

# Generate report
from analysis.carbon_impact import generate_carbon_report
report = generate_carbon_report(impact, output_path='carbon_report.txt')
print(report)
```

#### Time-Series Analysis
```python
from analysis.timeseries_analysis import DeforestationTimeSeriesAnalyzer
import pandas as pd

# Load time-series data
df = pd.read_csv('deforestation_timeseries.csv')

# Initialize analyzer
analyzer = DeforestationTimeSeriesAnalyzer(
    data=df,
    date_column='date',
    value_column='deforestation_area'
)

# Fit ARIMA model
results = analyzer.fit_arima(order=(1, 1, 1))
print(f"Model AIC: {results['aic']:.2f}")
print(f"Test MAE: {results['mae']:.2f}")

# Forecast future
forecast_df = analyzer.forecast_future(steps=12)
print(forecast_df)

# Visualize
analyzer.plot_forecast(forecast_steps=12, save_path='forecast.png')
analyzer.analyze_trends(save_path='trends.png')
```

---

## 🎨 Visualization Examples

### Create Custom Visualizations
```python
from visualization.visualizer import DeforestationVisualizer
import numpy as np

# Initialize visualizer
viz = DeforestationVisualizer()

# Plot before/after comparison
viz.plot_image_comparison(
    before_img=before_image,
    after_img=after_image,
    mask=deforestation_mask,
    save_path='comparison.png'
)

# Plot NDVI maps
viz.plot_ndvi_maps(
    ndvi_before=ndvi_before,
    ndvi_after=ndvi_after,
    ndvi_change=ndvi_change,
    save_path='ndvi_maps.png'
)

# Create deforestation heatmap
viz.plot_deforestation_heatmap(
    mask=deforestation_mask,
    title="Deforestation Hotspots",
    save_path='heatmap.png'
)

# Generate dashboard
stats = {
    'accuracy': 0.89,
    'precision': 0.87,
    'recall': 0.85,
    'f1_score': 0.86,
    'deforested_area_ha': 150.5,
    'co2_emissions_tons': 5200.0,
    'carbon_loss_tons': 1417.0,
    'equivalent_cars_year': 1130,
    'forest_area_before': 10000,
    'forest_area_after': 9849
}

viz.plot_deforestation_statistics(
    stats_dict=stats,
    save_path='dashboard.png'
)
```

---

## 🔧 Configuration & Customization

### Adjust Parameters in `config.py`

```python
# Image size
IMG_HEIGHT = 256
IMG_WIDTH = 256

# NDVI thresholds
NDVI_FOREST_MIN = 0.4        # Minimum NDVI for forest
NDVI_CHANGE_THRESHOLD = -0.15 # Threshold for deforestation

# Carbon parameters
CARBON_DENSITY_AMAZON = 190   # tons C/ha
PIXEL_AREA_HA = 0.01         # hectares per pixel

# Model training
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
```

### Custom Carbon Density
```python
from analysis.carbon_impact import CarbonImpactCalculator

# For different forest types
tropical_calc = CarbonImpactCalculator(carbon_density=180.0)
temperate_calc = CarbonImpactCalculator(carbon_density=90.0)
amazon_calc = CarbonImpactCalculator(carbon_density=190.0)
```

---

## 📁 Working with Real Kaggle Data

### Download Datasets
```bash
# Install Kaggle CLI
pip install kaggle

# Configure credentials (place kaggle.json in ~/.kaggle/)
# Download dataset
kaggle datasets download -d akhilchibber/deforestationdetection-dataset
unzip deforestationdetection-dataset.zip -d data/raw/amazon/
```

### Load Kaggle Dataset
```python
from data.data_loader import DeforestationDataLoader

# Initialize loader
loader = DeforestationDataLoader(data_dir='./data/raw')

# Load Amazon dataset
data = loader.load_kaggle_dataset(
    dataset_path='./data/raw/amazon',
    dataset_type='amazon'
)

# Access loaded data
images = data['images']
masks = data['masks']

print(f"Loaded {len(images)} images")
```

---

## 🚀 Advanced: API Deployment (Optional)

### Start FastAPI Server
```bash
# Install API dependencies
pip install fastapi uvicorn

# Start server
python api/api_server.py
```

### Access API
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### API Usage Examples

**Calculate NDVI:**
```bash
curl -X POST "http://localhost:8000/api/v1/calculate-ndvi" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@satellite_image.png"
```

**Detect Deforestation:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect-deforestation" \
  -F "before_image=@before.png" \
  -F "after_image=@after.png"
```

**Estimate Carbon Impact:**
```bash
curl -X POST "http://localhost:8000/api/v1/estimate-carbon-impact?deforested_area_ha=100&forest_type=amazon"
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: Import Errors**
```bash
# Solution: Install all dependencies
pip install -r requirements.txt
```

**Issue: TensorFlow GPU not working**
```bash
# Solution: Install GPU version
pip install tensorflow-gpu==2.10.0
# Or use CPU version (default)
pip install tensorflow==2.10.0
```

**Issue: Matplotlib plots not showing**
```python
# Solution: Add this at the top of your script
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

**Issue: Streamlit app not loading**
```bash
# Solution: Clear cache and restart
streamlit cache clear
streamlit run dashboard_app.py
```

---

## 📊 Expected Results

### Demo Pipeline Metrics
- **Detection Accuracy**: 85-92%
- **Processing Time**: 2-5 minutes for 50 samples
- **Deforested Area**: 100-200 hectares (synthetic)
- **CO2 Emissions**: 3,500-7,000 tons
- **Files Generated**: 5-10 visualizations + reports

### Real-World Performance
- **Processing Speed**: ~1-2 seconds per image pair (CPU)
- **Accuracy**: 80-95% (depends on data quality)
- **Scalability**: Can process 1000+ images with proper batching

---

## 🎯 Next Steps

1. **Test with Demo Data**: Run `python main_pipeline.py`
2. **Explore Dashboard**: Launch `streamlit run dashboard_app.py`
3. **Download Real Data**: Get Kaggle datasets
4. **Customize Parameters**: Edit `config.py` for your use case
5. **Train Deep Learning Models**: Use U-Net or Siamese networks
6. **Deploy API**: Set up FastAPI for production use
7. **Integrate Real Satellite APIs**: Connect to Google Earth Engine

---

## 📞 Need Help?

- Check module docstrings: Each file has detailed documentation
- Review code comments: Inline explanations throughout
- Test individual functions: All modules have `if __name__ == "__main__"` examples
- Check logs: `deforestation_pipeline.log` contains detailed execution logs

---

**Happy Forest Monitoring! 🌳🌍**
