"""
REAL DATA DOWNLOADER MODULE
Downloads and manages real deforestation datasets from Kaggle and satellite sources.

Supported Sources:
1. Kaggle Datasets (3 datasets)
2. Google Earth Engine
3. USGS EarthExplorer
4. ESA Copernicus (Sentinel-1/2)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List
import zipfile
import shutil

logger = logging.getLogger(__name__)

# Dataset URLs and identifiers
KAGGLE_DATASETS = {
    'amazon_deforestation': {
        'identifier': 'akhilchibber/deforestationdetection-dataset',
        'type': 'dataset',
        'description': 'Amazon Rainforest Deforestation with masks',
        'size_gb': 2.5
    },
    'deforestation_competition': {
        'identifier': 'deforestation',
        'type': 'competition',
        'description': 'Deforestation Detection Competition',
        'size_gb': 5.0
    },
    'timeseries_brazil': {
        'identifier': 'gallo33henrique/time-series-arima-sarima-deforestation-brazil',
        'type': 'dataset',
        'description': 'Time Series Analysis - Brazil Amazon',
        'size_gb': 0.1
    }
}


class RealDataDownloader:
    """
    Handles downloading and organizing real deforestation datasets.
    """
    
    def __init__(self, data_root: str = './data'):
        """
        Initialize the downloader.
        
        Args:
            data_root: Root directory for all data
        """
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / 'raw'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
    def check_kaggle_credentials(self) -> bool:
        """
        Check if Kaggle API credentials are configured.
        
        Returns:
            True if credentials are found, False otherwise
        """
        kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
        
        if kaggle_config.exists():
            logger.info("✅ Kaggle credentials found")
            return True
        else:
            logger.warning("❌ Kaggle credentials not found")
            logger.info("\nTo setup Kaggle API:")
            logger.info("1. Go to https://www.kaggle.com/settings")
            logger.info("2. Click 'Create New API Token'")
            logger.info("3. Place kaggle.json in:")
            logger.info(f"   Windows: {Path.home() / '.kaggle'}")
            logger.info("   Linux/Mac: ~/.kaggle/")
            return False
    
    def download_kaggle_dataset(self, dataset_key: str) -> bool:
        """
        Download a specific Kaggle dataset.
        
        Args:
            dataset_key: Key from KAGGLE_DATASETS dict
            
        Returns:
            True if successful, False otherwise
        """
        if dataset_key not in KAGGLE_DATASETS:
            logger.error(f"Unknown dataset: {dataset_key}")
            logger.info(f"Available: {list(KAGGLE_DATASETS.keys())}")
            return False
        
        dataset_info = KAGGLE_DATASETS[dataset_key]
        output_dir = self.raw_dir / dataset_key
        
        # Check if already downloaded
        if output_dir.exists() and any(output_dir.iterdir()):
            logger.info(f"✅ {dataset_key} already downloaded at {output_dir}")
            return True
        
        # Check credentials
        if not self.check_kaggle_credentials():
            return False
        
        try:
            import kaggle
            
            logger.info(f"\n📥 Downloading {dataset_info['description']}")
            logger.info(f"Size: ~{dataset_info['size_gb']} GB")
            logger.info(f"This may take several minutes...")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if dataset_info['type'] == 'dataset':
                kaggle.api.dataset_download_files(
                    dataset_info['identifier'],
                    path=str(output_dir),
                    unzip=True,
                    quiet=False
                )
            elif dataset_info['type'] == 'competition':
                kaggle.api.competition_download_files(
                    dataset_info['identifier'],
                    path=str(output_dir),
                    quiet=False
                )
                # Unzip competition files
                self._unzip_all_files(output_dir)
            
            logger.info(f"✅ Successfully downloaded to {output_dir}\n")
            return True
            
        except ImportError:
            logger.error("❌ Kaggle package not installed")
            logger.info("Install with: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"❌ Error downloading {dataset_key}: {e}")
            return False
    
    def download_all_kaggle_datasets(self) -> Dict[str, bool]:
        """
        Download all Kaggle datasets.
        
        Returns:
            Dictionary with download status for each dataset
        """
        logger.info("=" * 60)
        logger.info("DOWNLOADING ALL KAGGLE DATASETS")
        logger.info("=" * 60)
        
        total_size = sum(info['size_gb'] for info in KAGGLE_DATASETS.values())
        logger.info(f"\nTotal download size: ~{total_size} GB")
        logger.info("This will take some time depending on your internet speed.\n")
        
        results = {}
        for dataset_key in KAGGLE_DATASETS.keys():
            results[dataset_key] = self.download_kaggle_dataset(dataset_key)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        for dataset_key, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            logger.info(f"{dataset_key}: {status}")
        
        return results
    
    def _unzip_all_files(self, directory: Path):
        """Helper to unzip all zip files in a directory."""
        zip_files = list(directory.glob('*.zip'))
        for zip_file in zip_files:
            try:
                logger.info(f"Extracting {zip_file.name}...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(directory)
                zip_file.unlink()  # Remove zip after extraction
            except Exception as e:
                logger.warning(f"Could not extract {zip_file}: {e}")
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about downloaded datasets.
        
        Returns:
            Dictionary with dataset information
        """
        info = {}
        
        for dataset_key in KAGGLE_DATASETS.keys():
            dataset_dir = self.raw_dir / dataset_key
            
            if dataset_dir.exists():
                # Count files
                image_files = (list(dataset_dir.rglob('*.png')) + 
                             list(dataset_dir.rglob('*.jpg')) +
                             list(dataset_dir.rglob('*.tif')))
                
                csv_files = list(dataset_dir.rglob('*.csv'))
                
                info[dataset_key] = {
                    'downloaded': True,
                    'path': str(dataset_dir),
                    'num_images': len(image_files),
                    'num_csv': len(csv_files),
                    'size_mb': self._get_dir_size(dataset_dir)
                }
            else:
                info[dataset_key] = {
                    'downloaded': False,
                    'path': None,
                    'num_images': 0,
                    'num_csv': 0,
                    'size_mb': 0
                }
        
        return info
    
    def _get_dir_size(self, directory: Path) -> float:
        """Get directory size in MB."""
        total_size = 0
        for file in directory.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
        return round(total_size / (1024 * 1024), 2)
    
    def setup_google_earth_engine(self):
        """
        Instructions for setting up Google Earth Engine.
        """
        logger.info("\n" + "=" * 60)
        logger.info("GOOGLE EARTH ENGINE SETUP")
        logger.info("=" * 60)
        logger.info("\n1. Install Earth Engine API:")
        logger.info("   pip install earthengine-api")
        logger.info("\n2. Authenticate:")
        logger.info("   earthengine authenticate")
        logger.info("\n3. Sign up (if needed):")
        logger.info("   https://earthengine.google.com/signup/")
        logger.info("\n4. Documentation:")
        logger.info("   https://developers.google.com/earth-engine/guides/python_install")
        logger.info("\n✨ Earth Engine provides:")
        logger.info("   - Sentinel-1/2 radar and optical imagery")
        logger.info("   - Landsat 8/9 multispectral imagery")
        logger.info("   - Global coverage, free access")
        logger.info("   - Cloud-based processing")
    
    def display_satellite_data_sources(self):
        """
        Display information about free satellite data sources.
        """
        logger.info("\n" + "=" * 60)
        logger.info("FREE SATELLITE DATA SOURCES")
        logger.info("=" * 60)
        
        sources = {
            'Google Earth Engine': {
                'url': 'https://earthengine.google.com/',
                'data': 'Sentinel-1/2, Landsat, MODIS',
                'api': 'Python API available',
                'cost': 'Free'
            },
            'USGS EarthExplorer': {
                'url': 'https://earthexplorer.usgs.gov/',
                'data': 'Landsat, ASTER, SRTM',
                'api': 'Web interface + API',
                'cost': 'Free'
            },
            'ESA Copernicus': {
                'url': 'https://scihub.copernicus.eu/',
                'data': 'Sentinel-1/2/3/5P',
                'api': 'API available',
                'cost': 'Free'
            },
            'NASA Earthdata': {
                'url': 'https://earthdata.nasa.gov/',
                'data': 'MODIS, VIIRS, etc.',
                'api': 'API available',
                'cost': 'Free'
            }
        }
        
        for name, info in sources.items():
            logger.info(f"\n📡 {name}")
            logger.info(f"   URL: {info['url']}")
            logger.info(f"   Data: {info['data']}")
            logger.info(f"   API: {info['api']}")
            logger.info(f"   Cost: {info['cost']}")


class EarthEngineDataLoader:
    """
    Load satellite imagery from Google Earth Engine.
    Requires earthengine-api package and authentication.
    """
    
    def __init__(self):
        """Initialize Earth Engine."""
        self.initialized = False
        self._initialize_ee()
    
    def _initialize_ee(self):
        """Initialize Earth Engine API."""
        try:
            import ee
            ee.Initialize()
            self.ee = ee
            self.initialized = True
            logger.info("✅ Earth Engine initialized")
        except ImportError:
            logger.warning("❌ Earth Engine not installed: pip install earthengine-api")
        except Exception as e:
            logger.warning(f"❌ Earth Engine initialization failed: {e}")
            logger.info("Run: earthengine authenticate")
    
    def get_sentinel2_image(self, 
                           region: List[float],
                           start_date: str,
                           end_date: str,
                           max_cloud_cover: int = 20) -> Optional[Dict]:
        """
        Get Sentinel-2 imagery for a region and time period.
        
        Args:
            region: [lon_min, lat_min, lon_max, lat_max]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_cloud_cover: Maximum cloud cover percentage
            
        Returns:
            Dictionary with image data or None
        """
        if not self.initialized:
            logger.error("Earth Engine not initialized")
            return None
        
        try:
            # Define region
            roi = self.ee.Geometry.Rectangle(region)
            
            # Get image collection
            collection = (self.ee.ImageCollection('COPERNICUS/S2_SR')
                         .filterBounds(roi)
                         .filterDate(start_date, end_date)
                         .filter(self.ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
                         .select(['B4', 'B3', 'B2', 'B8']))  # RGB + NIR
            
            # Get the least cloudy image
            image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
            
            logger.info(f"✅ Found Sentinel-2 image for {start_date} to {end_date}")
            
            return {
                'image': image,
                'region': roi,
                'collection_size': collection.size().getInfo()
            }
            
        except Exception as e:
            logger.error(f"Error fetching Sentinel-2 data: {e}")
            return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 60)
    print("DEFORESTATION DETECTION - REAL DATA DOWNLOADER")
    print("=" * 60)
    
    # Initialize downloader
    downloader = RealDataDownloader(data_root='./data')
    
    # Show available datasets
    print("\n📊 AVAILABLE KAGGLE DATASETS:")
    print("-" * 60)
    for key, info in KAGGLE_DATASETS.items():
        print(f"\n{key}:")
        print(f"  Description: {info['description']}")
        print(f"  Size: ~{info['size_gb']} GB")
        print(f"  Identifier: {info['identifier']}")
    
    # Check current status
    print("\n\n📁 CURRENT DATASET STATUS:")
    print("-" * 60)
    dataset_info = downloader.get_dataset_info()
    for key, info in dataset_info.items():
        status = "✅ Downloaded" if info['downloaded'] else "❌ Not downloaded"
        print(f"\n{key}: {status}")
        if info['downloaded']:
            print(f"  Images: {info['num_images']}")
            print(f"  CSV files: {info['num_csv']}")
            print(f"  Size: {info['size_mb']} MB")
    
    # Show satellite data sources
    downloader.display_satellite_data_sources()
    
    # Show Earth Engine setup
    downloader.setup_google_earth_engine()
    
    # Interactive download
    print("\n\n" + "=" * 60)
    print("TO DOWNLOAD DATASETS:")
    print("=" * 60)
    print("\n# Download all Kaggle datasets:")
    print("python data/real_data_downloader.py --download-all")
    print("\n# Or in Python:")
    print("from data.real_data_downloader import RealDataDownloader")
    print("downloader = RealDataDownloader()")
    print("downloader.download_all_kaggle_datasets()")
    print("\n" + "=" * 60)
