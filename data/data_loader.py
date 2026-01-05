"""
STEP 1: DATA INGESTION MODULE
Handles loading of satellite images, masks, and time-series data from various sources.
Supports multi-temporal image pairs and Kaggle dataset integration.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from typing import Tuple, List, Dict, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Real Dataset Paths
AMAZON_DATASET_PATH = Path(r"C:\Users\yuvanshankar\.cache\kagglehub\datasets\akhilchibber\deforestation-detection-dataset\versions\1")
COMPETITION_DATASET_PATH = Path(r"C:\Users\yuvanshankar\Downloads\Ecoverse\data\raw\deforestation_competition")


class DeforestationDataLoader:
    """
    Main data loader for deforestation detection datasets.
    Handles image loading, mask loading, and time-series data.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Root directory containing the datasets
        """
        self.data_dir = Path(data_dir)
        self.image_pairs = []
        self.masks = []
        self.metadata = {}
        
    def load_image_pairs(self, 
                         before_dir: str, 
                         after_dir: str,
                         mask_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load multi-temporal image pairs (before/after) for change detection.
        
        Args:
            before_dir: Directory containing 'before' images
            after_dir: Directory containing 'after' images  
            mask_dir: Optional directory containing ground truth masks
            
        Returns:
            before_images: Array of before images [N, H, W, C]
            after_images: Array of after images [N, H, W, C]
            masks: Array of deforestation masks [N, H, W] or None
        """
        logger.info("Loading multi-temporal image pairs...")
        
        before_path = Path(before_dir)
        after_path = Path(after_dir)
        
        # Get all image files
        before_files = sorted(list(before_path.glob('*.jpg')) + 
                            list(before_path.glob('*.png')) +
                            list(before_path.glob('*.tif')))
        after_files = sorted(list(after_path.glob('*.jpg')) + 
                           list(after_path.glob('*.png')) +
                           list(after_path.glob('*.tif')))
        
        if len(before_files) != len(after_files):
            logger.warning(f"Mismatch in image counts: {len(before_files)} before vs {len(after_files)} after")
        
        before_images = []
        after_images = []
        masks = [] if mask_dir else None
        
        # Load image pairs
        for i, (before_file, after_file) in enumerate(zip(before_files, after_files)):
            # Load before image
            before_img = cv2.imread(str(before_file))
            if before_img is None:
                logger.warning(f"Failed to load: {before_file}")
                continue
            before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
            
            # Load after image
            after_img = cv2.imread(str(after_file))
            if after_img is None:
                logger.warning(f"Failed to load: {after_file}")
                continue
            after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
            
            before_images.append(before_img)
            after_images.append(after_img)
            
            # Load corresponding mask if available
            if mask_dir:
                mask_path = Path(mask_dir)
                mask_file = mask_path / before_file.name  # Assume same naming
                if mask_file.exists():
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    masks.append(mask)
                else:
                    logger.warning(f"Mask not found: {mask_file}")
                    masks.append(np.zeros(before_img.shape[:2], dtype=np.uint8))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Loaded {i + 1} image pairs...")
        
        logger.info(f"Total loaded: {len(before_images)} image pairs")
        
        return (np.array(before_images), 
                np.array(after_images), 
                np.array(masks) if masks else None)
    
    def load_kaggle_dataset(self, dataset_path: str, dataset_type: str = 'amazon') -> Dict:
        """
        Load Kaggle deforestation datasets with different structures.
        
        Args:
            dataset_path: Path to the Kaggle dataset directory
            dataset_type: Type of dataset ('amazon', 'competition')
            
        Returns:
            Dictionary containing loaded data
        """
        logger.info(f"Loading Kaggle dataset: {dataset_type}")
        
        dataset_path = Path(dataset_path)
        data = {}
        
        if dataset_type == 'amazon':
            # Amazon Deforestation Dataset structure
            sentinel2_dir = dataset_path / '1_CLOUD_FREE_DATASET' / '2_SENTINEL2'
            masks_dir = dataset_path / '3_TRAINING_MASKS'
            
            if sentinel2_dir.exists():
                logger.info(f"Loading Sentinel-2 images from {sentinel2_dir}")
                # Get all subdirectories with images
                image_dirs = [d for d in sentinel2_dir.iterdir() if d.is_dir()]
                all_images = []
                for img_dir in image_dirs:  # Load ALL directories (removed limit)
                    images = list(img_dir.glob('*.tif')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
                    if images:
                        all_images.extend(images)  # Load ALL images (removed limit)
                data['image_paths'] = [str(p) for p in all_images]
                data['image_count'] = len(all_images)
                logger.info(f"Found {len(all_images)} Sentinel-2 images")
            
            if masks_dir.exists():
                mask_files = list(masks_dir.glob('*.png')) + list(masks_dir.glob('*.tif'))
                data['mask_paths'] = [str(p) for p in mask_files[:len(data.get('image_paths', []))]]
                logger.info(f"Found {len(mask_files)} training masks")
                
        elif dataset_type == 'competition':
            # Competition Dataset structure: train/public/*.npy, test/public/*.npy
            train_dir = dataset_path / 'train' / 'public'
            test_dir = dataset_path / 'test' / 'public'
            
            if train_dir.exists():
                train_files = list(train_dir.glob('*.npy'))[:100]  # First 100 for demo
                data['train_paths'] = [str(p) for p in train_files]
                data['train_count'] = len(train_files)
                logger.info(f"Found {len(train_files)} training .npy files")
            
            if test_dir.exists():
                test_files = list(test_dir.glob('*.npy'))[:50]  # First 50 for demo
                data['test_paths'] = [str(p) for p in test_files]
                data['test_count'] = len(test_files)
                logger.info(f"Found {len(test_files)} test .npy files")
            
            # Load metadata JSON
            train_json = dataset_path / 'train.json'
            if train_json.exists():
                with open(train_json, 'r') as f:
                    import json
                    data['train_metadata'] = json.load(f)
                logger.info(f"Loaded {len(data['train_metadata'])} training metadata entries")
        
        return data
    
    def _load_images_from_dir(self, directory: Path, grayscale: bool = False) -> np.ndarray:
        """
        Helper function to load all images from a directory.
        
        Args:
            directory: Directory containing images
            grayscale: Whether to load as grayscale
            
        Returns:
            Array of images
        """
        image_files = sorted(list(directory.glob('*.jpg')) + 
                           list(directory.glob('*.png')) +
                           list(directory.glob('*.tif')))
        
        images = []
        for img_file in image_files:
            if grayscale:
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(str(img_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img is not None:
                images.append(img)
        
        logger.info(f"Loaded {len(images)} images from {directory}")
        return np.array(images)
    
    def download_kaggle_dataset(self, dataset_name: str, output_dir: str):
        """
        Download dataset from Kaggle using kaggle API.
        Note: Requires kaggle API credentials (~/.kaggle/kaggle.json)
        
        Args:
            dataset_name: Kaggle dataset identifier (e.g., 'username/dataset-name')
            output_dir: Directory to save downloaded data
        """
        try:
            import kaggle
            logger.info(f"Downloading {dataset_name} from Kaggle...")
            
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=output_dir, 
                unzip=True
            )
            logger.info(f"Successfully downloaded to {output_dir}")
            
        except ImportError:
            logger.error("Kaggle package not installed. Run: pip install kaggle")
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            logger.info("Make sure kaggle.json is in ~/.kaggle/")
    
    def create_dummy_dataset(self, output_dir: str, num_samples: int = 100):
        """
        Create dummy dataset for testing when real data is not available.
        Generates synthetic before/after images and masks.
        
        Args:
            output_dir: Directory to save dummy data
            num_samples: Number of samples to generate
        """
        logger.info(f"Creating dummy dataset with {num_samples} samples...")
        
        output_path = Path(output_dir)
        before_dir = output_path / 'before'
        after_dir = output_path / 'after'
        mask_dir = output_path / 'masks'
        
        for directory in [before_dir, after_dir, mask_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples):
            # Create synthetic before image (forest)
            before_img = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
            before_img[:, :, 1] = np.random.randint(100, 200, (256, 256))  # More green
            
            # Create synthetic after image (some deforestation)
            after_img = before_img.copy()
            
            # Add deforested regions (brown/gray patches)
            num_patches = np.random.randint(1, 5)
            mask = np.zeros((256, 256), dtype=np.uint8)
            
            for _ in range(num_patches):
                x = np.random.randint(0, 200)
                y = np.random.randint(0, 200)
                w = np.random.randint(20, 60)
                h = np.random.randint(20, 60)
                
                after_img[y:y+h, x:x+w] = [150, 100, 70]  # Brown color
                mask[y:y+h, x:x+w] = 255  # Mark as deforested
            
            # Save images
            cv2.imwrite(str(before_dir / f'image_{i:04d}.png'), 
                       cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(after_dir / f'image_{i:04d}.png'), 
                       cv2.cvtColor(after_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_dir / f'image_{i:04d}.png'), mask)
        
        logger.info(f"Dummy dataset created at {output_dir}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def verify_dataset_structure(data_dir: str) -> Dict[str, bool]:
    """
    Verify that the dataset has the expected structure.
    
    Args:
        data_dir: Root data directory
        
    Returns:
        Dictionary with validation results
    """
    data_path = Path(data_dir)
    
    checks = {
        'data_dir_exists': data_path.exists(),
        'has_images': len(list(data_path.rglob('*.png')) + list(data_path.rglob('*.jpg'))) > 0,
        'has_subdirs': len(list(data_path.iterdir())) > 0
    }
    
    return checks


def get_dataset_statistics(images: np.ndarray) -> Dict[str, any]:
    """
    Compute basic statistics about the dataset.
    
    Args:
        images: Array of images [N, H, W, C]
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'num_images': len(images),
        'image_shape': images[0].shape if len(images) > 0 else None,
        'mean': np.mean(images, axis=(0, 1, 2)),
        'std': np.std(images, axis=(0, 1, 2)),
        'min': np.min(images),
        'max': np.max(images),
        'dtype': images.dtype
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("Deforestation Data Loader Module")
    print("=" * 50)
    
    # Initialize loader
    loader = DeforestationDataLoader(data_dir='./data/raw')
    
    # Create dummy dataset for testing
    loader.create_dummy_dataset('./data/raw/dummy', num_samples=50)
    
    # Load the dummy dataset
    before, after, masks = loader.load_image_pairs(
        before_dir='./data/raw/dummy/before',
        after_dir='./data/raw/dummy/after',
        mask_dir='./data/raw/dummy/masks'
    )
    
    print(f"\nLoaded dataset:")
    print(f"Before images: {before.shape}")
    print(f"After images: {after.shape}")
    print(f"Masks: {masks.shape}")
    
    # Get statistics
    stats = get_dataset_statistics(before)
    print(f"\nDataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
