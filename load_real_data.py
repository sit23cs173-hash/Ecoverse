"""
Script to load and verify real datasets instead of mock data.
Demonstrates integration of Amazon, Competition, and Time-Series Brazil datasets.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from data.data_loader import DeforestationDataLoader, AMAZON_DATASET_PATH, COMPETITION_DATASET_PATH, TIMESERIES_DATASET_PATH

def load_amazon_dataset():
    """Load Amazon Deforestation Dataset (10GB, Sentinel-2 satellite images)"""
    print("\n" + "="*80)
    print("📡 LOADING AMAZON DEFORESTATION DATASET")
    print("="*80)
    
    loader = DeforestationDataLoader(str(AMAZON_DATASET_PATH))
    data = loader.load_kaggle_dataset(str(AMAZON_DATASET_PATH), dataset_type='amazon')
    
    print(f"\n✅ Amazon Dataset Loaded:")
    print(f"   • Image Paths: {data.get('image_count', 0)} Sentinel-2 images")
    print(f"   • Mask Paths: {len(data.get('mask_paths', []))} training masks")
    print(f"   • Dataset Location: {AMAZON_DATASET_PATH}")
    
    # Show sample image paths
    if 'image_paths' in data and data['image_paths']:
        print(f"\n   Sample Images:")
        for i, path in enumerate(data['image_paths'][:3], 1):
            print(f"      {i}. {Path(path).name}")
    
    return data

def load_competition_dataset():
    """Load Competition Dataset (4,043 .npy files with metadata)"""
    print("\n" + "="*80)
    print("🏆 LOADING COMPETITION DATASET")
    print("="*80)
    
    loader = DeforestationDataLoader(str(COMPETITION_DATASET_PATH))
    data = loader.load_kaggle_dataset(str(COMPETITION_DATASET_PATH), dataset_type='competition')
    
    print(f"\n✅ Competition Dataset Loaded:")
    print(f"   • Training Files: {data.get('train_count', 0)} .npy files")
    print(f"   • Test Files: {data.get('test_count', 0)} .npy files")
    print(f"   • Metadata Entries: {len(data.get('train_metadata', []))} records")
    print(f"   • Dataset Location: {COMPETITION_DATASET_PATH}")
    
    # Show sample metadata
    if 'train_metadata' in data and data['train_metadata']:
        metadata_list = data['train_metadata']
        if isinstance(metadata_list, list) and len(metadata_list) > 0:
            sample = metadata_list[0]
            print(f"\n   Sample Metadata:")
            print(f"      Tile: {sample.get('tile', 'N/A')}")
            print(f"      First Date: {sample.get('date_first', 'N/A')}")
            print(f"      Second Date: {sample.get('date_second', 'N/A')}")
            print(f"      Coordinates: {sample.get('ij', 'N/A')}")
    
    return data

def load_timeseries_dataset():
    """Load Time-Series Brazil Dataset (1999-2019 fire and deforestation data)"""
    print("\n" + "="*80)
    print("📊 LOADING TIME-SERIES BRAZIL DATASET")
    print("="*80)
    
    loader = DeforestationDataLoader(str(TIMESERIES_DATASET_PATH))
    data = loader.load_kaggle_dataset(str(TIMESERIES_DATASET_PATH), dataset_type='timeseries')
    
    print(f"\n✅ Time-Series Dataset Loaded:")
    print(f"   • Datasets: {len(data.get('datasets', {}))} CSV files")
    print(f"   • Dataset Location: {TIMESERIES_DATASET_PATH}")
    
    # Show details for each dataset
    if 'datasets' in data:
        print(f"\n   Available Datasets:")
        for name, df in data['datasets'].items():
            print(f"      • {name}: {df.shape[0]} rows, {df.shape[1]} columns")
            if not df.empty:
                print(f"        Columns: {', '.join(df.columns.tolist()[:5])}")
    
    # Show sample fire data
    if 'timeseries' in data:
        df = data['timeseries']
        print(f"\n   Primary Dataset (Amazon Fires 1999-2019):")
        print(f"      Total Records: {len(df)}")
        if 'year' in df.columns:
            years = df['year'].unique() if 'year' in df.columns else []
            print(f"      Years Covered: {len(years)} years ({min(years) if len(years) > 0 else 'N/A'}-{max(years) if len(years) > 0 else 'N/A'})")
    
    return data

def load_sample_image_from_amazon(data):
    """Load and display a sample image from Amazon dataset"""
    if 'image_paths' not in data or not data['image_paths']:
        print("\n⚠️  No images found in Amazon dataset")
        return None
    
    sample_path = data['image_paths'][0]
    print(f"\n🖼️  Loading Sample Image: {Path(sample_path).name}")
    
    try:
        img = cv2.imread(sample_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"   ✅ Image loaded successfully")
            print(f"   • Shape: {img.shape}")
            print(f"   • Data Type: {img.dtype}")
            print(f"   • Size: {img.nbytes / 1024:.1f} KB")
            return img
        else:
            print(f"   ❌ Failed to load image")
            return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def load_sample_npy_from_competition(data):
    """Load a sample .npy file from competition dataset"""
    if 'train_paths' not in data or not data['train_paths']:
        print("\n⚠️  No .npy files found in competition dataset")
        return None
    
    sample_path = data['train_paths'][0]
    print(f"\n📦 Loading Sample .npy File: {Path(sample_path).name}")
    
    try:
        arr = np.load(sample_path)
        print(f"   ✅ .npy loaded successfully")
        print(f"   • Shape: {arr.shape}")
        print(f"   • Data Type: {arr.dtype}")
        print(f"   • Size: {arr.nbytes / 1024:.1f} KB")
        print(f"   • Value Range: [{arr.min():.2f}, {arr.max():.2f}]")
        return arr
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    """Main function to load all real datasets"""
    print("\n" + "🌳"*40)
    print("ECOVERSE - REAL DATASET INTEGRATION")
    print("Loading Amazon, Competition, and Time-Series Brazil Datasets")
    print("🌳"*40)
    
    # Load all datasets
    amazon_data = load_amazon_dataset()
    competition_data = load_competition_dataset()
    timeseries_data = load_timeseries_dataset()
    
    # Load sample data
    sample_image = load_sample_image_from_amazon(amazon_data)
    sample_npy = load_sample_npy_from_competition(competition_data)
    
    # Summary
    print("\n" + "="*80)
    print("📈 DATASET INTEGRATION SUMMARY")
    print("="*80)
    print(f"✅ Amazon Dataset: {amazon_data.get('image_count', 0)} images available")
    print(f"✅ Competition Dataset: {competition_data.get('train_count', 0)} training files")
    print(f"✅ Time-Series Dataset: {len(timeseries_data.get('datasets', {}))} CSV files loaded")
    print(f"✅ Sample Image: {'Loaded' if sample_image is not None else 'Not loaded'}")
    print(f"✅ Sample .npy: {'Loaded' if sample_npy is not None else 'Not loaded'}")
    print("\n" + "🎉"*40)
    print("Real datasets successfully integrated into Ecoverse!")
    print("🎉"*40 + "\n")

if __name__ == "__main__":
    main()
