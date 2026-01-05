"""
Configuration file for Deforestation Detection System
Contains all project constants, paths, and hyperparameters
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATA SOURCES
# ============================================================================
KAGGLE_DATASETS = {
    'amazon_deforestation': 'akhilchibber/deforestationdetection-dataset',
    'deforestation_competition': 'deforestation/data',
    'timeseries_brazil': 'gallo33henrique/time-series-arima-sarima-deforestation-brazil'
}

# ============================================================================
# IMAGE PROCESSING PARAMETERS
# ============================================================================
# Standard image size for model input
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

# Normalization parameters (for satellite imagery)
NORM_MEAN = [0.485, 0.456, 0.406]  # ImageNet standard
NORM_STD = [0.229, 0.224, 0.225]

# NDVI computation bands (adjust based on your satellite data)
# For Sentinel-2: RED=Band 4, NIR=Band 8
# For Landsat: RED=Band 3, NIR=Band 4
RED_BAND_IDX = 0  # Index in your image array
NIR_BAND_IDX = 3  # Index in your image array

# ============================================================================
# VEGETATION INDICES THRESHOLDS
# ============================================================================
# NDVI ranges for classification
NDVI_FOREST_MIN = 0.4      # Dense vegetation
NDVI_SPARSE_VEG_MIN = 0.2  # Sparse vegetation
NDVI_BARE_SOIL_MAX = 0.2   # Bare soil/deforested

# Change detection threshold
NDVI_CHANGE_THRESHOLD = -0.15  # Significant vegetation loss

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
# Training parameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Model architecture
USE_PRETRAINED = True  # Use pretrained weights (ImageNet)
DROPOUT_RATE = 0.5

# Early stopping
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# ============================================================================
# CARBON ESTIMATION PARAMETERS
# ============================================================================
# Average carbon density in tropical forests (tons per hectare)
# Source: IPCC guidelines, varies by forest type
CARBON_DENSITY_TROPICAL = 180  # tons C/ha
CARBON_DENSITY_TEMPERATE = 90   # tons C/ha
CARBON_DENSITY_AMAZON = 190     # tons C/ha (Amazon-specific)

# Default to use
DEFAULT_CARBON_DENSITY = CARBON_DENSITY_AMAZON

# CO2 conversion factor (1 ton C = 3.67 tons CO2)
CO2_CONVERSION_FACTOR = 3.67

# Pixel to area conversion (depends on satellite resolution)
# For Sentinel-2: 10m resolution -> 100 m² per pixel -> 0.01 hectares
# For Landsat: 30m resolution -> 900 m² per pixel -> 0.09 hectares
PIXEL_AREA_HA = 0.01  # hectares per pixel (adjust based on your data)

# ============================================================================
# TIME SERIES ANALYSIS
# ============================================================================
# ARIMA/SARIMA parameters
ARIMA_ORDER = (1, 1, 1)  # (p, d, q)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)  # (P, D, Q, s) - s=12 for monthly

# Forecasting
FORECAST_PERIODS = 24  # months ahead

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
# Color maps
CMAP_NDVI = 'RdYlGn'  # Red-Yellow-Green for vegetation
CMAP_CHANGE = 'RdBu_r'  # Red-Blue for change detection
CMAP_DEFORESTATION = 'Reds'  # Red scale for deforestation

# Figure settings
FIG_DPI = 150
FIG_SIZE = (12, 8)

# ============================================================================
# API & DEPLOYMENT
# ============================================================================
API_HOST = '0.0.0.0'
API_PORT = 8000
MODEL_VERSION = 'v1.0'

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
