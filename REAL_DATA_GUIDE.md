# Using Real Kaggle Datasets

This guide shows how to download and use real deforestation datasets from Kaggle.

## 📚 Available Datasets

### 1. Amazon Rainforest Deforestation Dataset
- **Source**: https://www.kaggle.com/datasets/akhilchibber/deforestationdetection-dataset
- **Content**: Satellite images with deforestation masks
- **Size**: ~2.5 GB
- **Format**: PNG images + masks
- **Use**: Training detection models

### 2. Deforestation Detection Competition
- **Source**: https://www.kaggle.com/competitions/deforestation/data
- **Content**: Multi-temporal satellite imagery
- **Size**: ~5.0 GB
- **Format**: JPEG/PNG images + CSV metadata
- **Use**: Change detection analysis

### 3. Time Series Analysis - Brazil Amazon
- **Source**: https://www.kaggle.com/code/gallo33henrique/time-series-arima-sarima-deforestation-brazil
- **Content**: Historical deforestation data
- **Size**: ~0.1 GB
- **Format**: CSV files
- **Use**: ARIMA/SARIMA forecasting

## 🔧 Setup Instructions

### Step 1: Install Kaggle API

```bash
pip install kaggle
```

### Step 2: Get Kaggle API Credentials

1. Go to https://www.kaggle.com/settings
2. Scroll to the **API** section
3. Click **"Create New API Token"**
4. This downloads `kaggle.json`
5. Move it to:
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

6. Set permissions (Linux/Mac):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 3: Download Datasets

#### Option A: Automated Download (Recommended)

```bash
# Run the download script
python download_datasets.py
```

This will:
- ✅ Check Kaggle credentials
- ✅ Download all 3 datasets
- ✅ Unzip automatically
- ✅ Organize in `data/raw/` directory
- ✅ Show download progress

#### Option B: Manual Download via Python

```python
from data.real_data_downloader import RealDataDownloader

# Initialize downloader
downloader = RealDataDownloader(data_root='./data')

# Download all datasets
results = downloader.download_all_kaggle_datasets()

# Or download specific dataset
downloader.download_kaggle_dataset('amazon_deforestation')
downloader.download_kaggle_dataset('deforestation_competition')
downloader.download_kaggle_dataset('timeseries_brazil')
```

#### Option C: Manual Download via CLI

```bash
# Amazon Deforestation Dataset
kaggle datasets download -d akhilchibber/deforestationdetection-dataset -p ./data/raw/amazon_deforestation --unzip

# Deforestation Competition
kaggle competitions download -c deforestation -p ./data/raw/deforestation_competition

# Time Series Brazil
kaggle datasets download -d gallo33henrique/time-series-arima-sarima-deforestation-brazil -p ./data/raw/timeseries_brazil --unzip
```

## 📁 Expected Directory Structure

After downloading, your data folder should look like:

```
data/
├── raw/
│   ├── amazon_deforestation/
│   │   ├── images/          # Satellite images
│   │   ├── masks/           # Deforestation masks
│   │   └── metadata.csv     # Image metadata
│   │
│   ├── deforestation_competition/
│   │   ├── train/           # Training images
│   │   ├── test/            # Test images
│   │   └── train.csv        # Labels
│   │
│   └── timeseries_brazil/
│       └── deforestation_data.csv  # Time series data
│
└── processed/               # Processed data (created during training)
```

## 🚀 Using Real Data in the Project

### Method 1: Load Data in Python

```python
from data.data_loader import DeforestationDataLoader

# Initialize loader
loader = DeforestationDataLoader(data_dir='./data/raw')

# Load Amazon deforestation dataset
data = loader.load_kaggle_dataset(
    dataset_path='./data/raw/amazon_deforestation',
    dataset_type='amazon'
)

# Access images and masks
images = data['images']
masks = data['masks']

print(f"Loaded {len(images)} images with masks")
```

### Method 2: Load Image Pairs

```python
# Load before/after image pairs
before, after, masks = loader.load_image_pairs(
    before_dir='./data/raw/amazon_deforestation/before',
    after_dir='./data/raw/amazon_deforestation/after',
    mask_dir='./data/raw/amazon_deforestation/masks'
)

print(f"Image pairs: {before.shape}")
print(f"Masks: {masks.shape}")
```

### Method 3: Load Time Series Data

```python
# Load time series data
timeseries_df = loader.load_timeseries_data(
    csv_path='./data/raw/timeseries_brazil/deforestation_data.csv'
)

print(timeseries_df.head())
print(f"Time series shape: {timeseries_df.shape}")
```

## 📊 Check Downloaded Data

```python
from data.real_data_downloader import RealDataDownloader

downloader = RealDataDownloader()
info = downloader.get_dataset_info()

for dataset, details in info.items():
    if details['downloaded']:
        print(f"\n{dataset}:")
        print(f"  ✅ Downloaded")
        print(f"  📁 Path: {details['path']}")
        print(f"  🖼️  Images: {details['num_images']}")
        print(f"  📄 CSV files: {details['num_csv']}")
        print(f"  💾 Size: {details['size_mb']} MB")
    else:
        print(f"\n{dataset}: ❌ Not downloaded")
```

## 🌍 Alternative: Free Satellite Data Sources

If you want fresh satellite data instead of Kaggle datasets:

### Google Earth Engine

```bash
# Install
pip install earthengine-api

# Authenticate
earthengine authenticate

# Sign up at: https://earthengine.google.com/signup/
```

**Python Example:**

```python
from data.real_data_downloader import EarthEngineDataLoader

# Initialize
ee_loader = EarthEngineDataLoader()

# Get Sentinel-2 image for Amazon region
image_data = ee_loader.get_sentinel2_image(
    region=[-73.0, -15.0, -50.0, -5.0],  # [lon_min, lat_min, lon_max, lat_max]
    start_date='2023-01-01',
    end_date='2023-12-31',
    max_cloud_cover=20
)
```

### Other Sources

1. **USGS EarthExplorer**: https://earthexplorer.usgs.gov/
   - Landsat, ASTER, SRTM
   - Manual download via web interface

2. **ESA Copernicus**: https://scihub.copernicus.eu/
   - Sentinel-1/2/3/5P
   - API and web interface

3. **NASA Earthdata**: https://earthdata.nasa.gov/
   - MODIS, VIIRS, etc.
   - Free registration required

## 🎯 Running the Project with Real Data

### Run Dashboard

```bash
streamlit run dashboard_app.py
```

Then select "Upload Your Data" mode and upload images from your downloaded datasets.

### Run Main Pipeline

```python
from main_pipeline import DeforestationPipeline

# Initialize with real data path
pipeline = DeforestationPipeline(
    data_dir='./data/raw/amazon_deforestation'
)

# Run full pipeline
results = pipeline.run_demo_pipeline(num_samples=100)

print(f"Carbon loss: {results['carbon_loss']} tons CO2")
```

### Train Models on Real Data

```python
from models.deforestation_model import DeforestationDetector
from data.data_loader import DeforestationDataLoader

# Load real data
loader = DeforestationDataLoader('./data/raw')
before, after, masks = loader.load_image_pairs(
    before_dir='./data/raw/amazon_deforestation/before',
    after_dir='./data/raw/amazon_deforestation/after',
    mask_dir='./data/raw/amazon_deforestation/masks'
)

# Train model
model = DeforestationDetector(input_shape=(256, 256, 6))
model.build_unet_model()
model.compile_model()

# Train on real data
history = model.train(
    X_train=np.concatenate([before, after], axis=-1),
    y_train=masks,
    X_val=None,  # Add validation split
    y_val=None,
    batch_size=16,
    epochs=50
)
```

## ❓ Troubleshooting

### Issue: "401 Unauthorized" when downloading

**Solution**: Your Kaggle credentials are incorrect or expired.
1. Delete old `kaggle.json`
2. Generate new API token from Kaggle settings
3. Place new `kaggle.json` in correct location

### Issue: "403 Forbidden" for competition data

**Solution**: Accept competition rules first.
1. Go to https://www.kaggle.com/competitions/deforestation
2. Click "Join Competition"
3. Accept rules
4. Try downloading again

### Issue: Downloads are very slow

**Solutions**:
- Use wired connection instead of WiFi
- Download during off-peak hours
- Download datasets one at a time
- Check disk space (need ~8 GB free)

### Issue: "OSError: [Errno 28] No space left on device"

**Solution**: Free up disk space.
```bash
# Check available space
df -h  # Linux/Mac
# Or check in File Explorer on Windows

# Need at least 8-10 GB free
```

## 📝 Summary

1. ✅ Install Kaggle API: `pip install kaggle`
2. ✅ Get credentials from https://www.kaggle.com/settings
3. ✅ Place `kaggle.json` in `~/.kaggle/`
4. ✅ Run: `python download_datasets.py`
5. ✅ Wait for downloads (~10-30 minutes)
6. ✅ Use real data in your pipeline!

**Total download size**: ~7.6 GB
**Time required**: 10-30 minutes (depending on internet speed)

---

**Questions?** Check the main README.md or open an issue on GitHub.
