# 🌳 Real Data Integration - Complete Setup

## ✅ What's Been Set Up

Your project now supports **REAL Kaggle datasets** and satellite data sources!

### 📦 New Files Added

1. **`data/real_data_downloader.py`**
   - Downloads all 3 Kaggle datasets automatically
   - Supports Google Earth Engine integration
   - Lists free satellite data sources

2. **`download_datasets.py`**
   - Simple CLI script to download all datasets
   - Interactive with progress tracking
   - Checks credentials before downloading

3. **`REAL_DATA_GUIDE.md`**
   - Complete guide for using real datasets
   - Setup instructions for Kaggle API
   - Examples for all data sources

### 🔧 Packages Installed

✅ **kaggle** - Download datasets from Kaggle
✅ All other dependencies already installed

---

## 🚀 Quick Start: Download Real Data

### Step 1: Setup Kaggle API Credentials

1. Go to https://www.kaggle.com/settings
2. Scroll to **"API"** section
3. Click **"Create New API Token"**
4. Download `kaggle.json`
5. Move it to:
   ```
   Windows: C:\Users\yuvanshankar\.kaggle\kaggle.json
   ```

### Step 2: Download Datasets

```bash
# Run the downloader script
python download_datasets.py
```

**What it downloads:**
- 🌲 Amazon Deforestation Detection Dataset (~2.5 GB)
- 🛰️ Deforestation Competition Data (~5.0 GB)
- 📊 Time Series Brazil Amazon (~0.1 GB)
- **Total: ~7.6 GB**

**Time needed:** 10-30 minutes (depending on internet)

---

## 📚 Available Real Datasets

### Dataset 1: Amazon Rainforest Deforestation
- **URL**: https://www.kaggle.com/datasets/akhilchibber/deforestationdetection-dataset
- **Content**: Satellite images + deforestation masks
- **Format**: PNG images
- **Size**: ~2.5 GB
- **Use Case**: Train U-Net and Siamese networks

### Dataset 2: Deforestation Competition
- **URL**: https://www.kaggle.com/competitions/deforestation/data
- **Content**: Multi-date satellite imagery
- **Format**: JPEG/PNG + CSV
- **Size**: ~5.0 GB
- **Use Case**: Change detection, temporal analysis

### Dataset 3: Time Series - Brazil Amazon
- **URL**: https://www.kaggle.com/code/gallo33henrique/time-series-arima-sarima-deforestation-brazil
- **Content**: Historical deforestation data
- **Format**: CSV
- **Size**: ~0.1 GB
- **Use Case**: ARIMA/SARIMA forecasting

---

## 🌍 Free Satellite Data Sources

### Google Earth Engine
- **Data**: Sentinel-1/2, Landsat, MODIS
- **Coverage**: Global
- **Resolution**: 10-30m
- **Access**: Python API (free)
- **Setup**: 
  ```bash
  pip install earthengine-api
  earthengine authenticate
  ```
- **URL**: https://earthengine.google.com/

### USGS EarthExplorer
- **Data**: Landsat, ASTER, SRTM
- **Coverage**: Global
- **Access**: Web download
- **URL**: https://earthexplorer.usgs.gov/

### ESA Copernicus
- **Data**: Sentinel-1/2/3/5P
- **Coverage**: Global
- **Access**: API + Web
- **URL**: https://scihub.copernicus.eu/

### NASA Earthdata
- **Data**: MODIS, VIIRS, many others
- **Coverage**: Global
- **Access**: Various APIs
- **URL**: https://earthdata.nasa.gov/

---

## 💻 Using Real Data in Code

### Load Kaggle Dataset

```python
from data.data_loader import DeforestationDataLoader

# Initialize
loader = DeforestationDataLoader(data_dir='./data/raw')

# Load Amazon dataset
data = loader.load_kaggle_dataset(
    dataset_path='./data/raw/amazon_deforestation',
    dataset_type='amazon'
)

images = data['images']
masks = data['masks']
print(f"Loaded {len(images)} images")
```

### Load Image Pairs (Before/After)

```python
before, after, masks = loader.load_image_pairs(
    before_dir='./data/raw/amazon_deforestation/before',
    after_dir='./data/raw/amazon_deforestation/after',
    mask_dir='./data/raw/amazon_deforestation/masks'
)

print(f"Before: {before.shape}")
print(f"After: {after.shape}")
print(f"Masks: {masks.shape}")
```

### Load Time Series Data

```python
timeseries = loader.load_timeseries_data(
    csv_path='./data/raw/timeseries_brazil/deforestation_data.csv'
)

print(timeseries.head())
```

### Use Google Earth Engine

```python
from data.real_data_downloader import EarthEngineDataLoader

ee_loader = EarthEngineDataLoader()

# Get Sentinel-2 for Amazon region
image_data = ee_loader.get_sentinel2_image(
    region=[-73.0, -15.0, -50.0, -5.0],  # Amazon bbox
    start_date='2023-01-01',
    end_date='2023-12-31',
    max_cloud_cover=20
)
```

---

## 🎯 Next Steps

### Option 1: Download and Use Real Data

```bash
# 1. Setup Kaggle credentials (see above)

# 2. Download datasets
python download_datasets.py

# 3. Use in pipeline
python main_pipeline.py --use-real-data

# 4. Or use in dashboard
streamlit run dashboard_app.py
# Then upload images from data/raw/ folders
```

### Option 2: Continue with Mock Data

The dashboard is currently running with synthetic data at:
**http://localhost:8501**

You can:
- ✅ Test all features with mock data
- ✅ See how the system works
- ✅ Download real data later when ready

---

## 📊 Check Download Status

```python
from data.real_data_downloader import RealDataDownloader

downloader = RealDataDownloader()
info = downloader.get_dataset_info()

for dataset, details in info.items():
    if details['downloaded']:
        print(f"✅ {dataset}")
        print(f"   Images: {details['num_images']}")
        print(f"   Size: {details['size_mb']} MB")
    else:
        print(f"❌ {dataset} - Not downloaded")
```

---

## 🐛 Troubleshooting

### Issue: "401 Unauthorized"
**Fix**: Regenerate Kaggle API token and replace kaggle.json

### Issue: "403 Forbidden" (competition data)
**Fix**: 
1. Go to https://www.kaggle.com/competitions/deforestation
2. Click "Join Competition"
3. Accept rules
4. Try again

### Issue: Slow downloads
**Tips**:
- Use wired connection
- Download during off-peak hours
- Download one dataset at a time

### Issue: No space left
**Fix**: Need ~10 GB free disk space

---

## 📁 Expected Directory Structure After Download

```
Ecoverse/
├── data/
│   ├── raw/
│   │   ├── amazon_deforestation/    ← Downloaded
│   │   ├── deforestation_competition/  ← Downloaded
│   │   └── timeseries_brazil/       ← Downloaded
│   └── processed/
├── download_datasets.py              ← New
├── REAL_DATA_GUIDE.md               ← New
└── data/
    └── real_data_downloader.py       ← New
```

---

## 🎓 Summary

✅ **Project supports both:**
- 🤖 Mock/synthetic data (current default)
- 🌍 Real Kaggle datasets (ready to download)
- 🛰️ Live satellite data (Earth Engine, etc.)

✅ **Ready to download:**
- Run `python download_datasets.py`
- Takes 10-30 minutes for ~7.6 GB

✅ **Dashboard is running:**
- http://localhost:8501
- Currently using mock data
- Can switch to real data after download

---

**Need help?** Check [REAL_DATA_GUIDE.md](REAL_DATA_GUIDE.md) for detailed instructions!
