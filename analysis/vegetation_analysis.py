"""
STEP 3: VEGETATION ANALYSIS MODULE
Computes vegetation indices (NDVI, EVI, SAVI) for forest cover analysis.
NDVI = (NIR - RED) / (NIR + RED)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VegetationIndexCalculator:
    """
    Calculate various vegetation indices from satellite imagery.
    """
    
    def __init__(self, red_band_idx: int = 0, nir_band_idx: int = 3):
        """
        Initialize vegetation index calculator.
        
        Args:
            red_band_idx: Index of RED band in image array
            nir_band_idx: Index of NIR (Near-Infrared) band in image array
            
        Note:
            For Sentinel-2: RED=Band 4 (665nm), NIR=Band 8 (842nm)
            For Landsat 8: RED=Band 4 (655nm), NIR=Band 5 (865nm)
            For RGB images: We approximate using R and pseudo-NIR
        """
        self.red_band_idx = red_band_idx
        self.nir_band_idx = nir_band_idx
    
    def calculate_ndvi(self, image: np.ndarray, use_rgb_approximation: bool = True) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index (NDVI).
        
        NDVI = (NIR - RED) / (NIR + RED)
        Range: -1 to 1 (higher values = more vegetation)
        
        Args:
            image: Satellite image [H, W, C] or [N, H, W, C]
            use_rgb_approximation: If True, approximate NIR from RGB (for RGB-only images)
            
        Returns:
            NDVI array [H, W] or [N, H, W]
        """
        # Handle batch dimension
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        if use_rgb_approximation:
            # For RGB images, approximate NDVI using green and red channels
            # This is a simplified approach when NIR band is not available
            red = image[:, :, :, 0]
            green = image[:, :, :, 1]
            
            # Pseudo-NDVI using green as proxy for NIR
            # Not accurate but useful for RGB-only datasets
            ndvi = (green - red) / (green + red + 1e-8)
        else:
            # Standard NDVI calculation with actual NIR band
            red = image[:, :, :, self.red_band_idx]
            nir = image[:, :, :, self.nir_band_idx]
            
            ndvi = (nir - red) / (nir + red + 1e-8)
        
        # Clip to valid range
        ndvi = np.clip(ndvi, -1, 1)
        
        if squeeze_output:
            ndvi = ndvi[0]
        
        return ndvi
    
    def calculate_evi(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate Enhanced Vegetation Index (EVI).
        
        EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
        
        More sensitive to canopy variations and reduces atmospheric effects.
        
        Args:
            image: Satellite image [H, W, C] or [N, H, W, C]
            
        Returns:
            EVI array [H, W] or [N, H, W]
        """
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]
            squeeze_output = True
        else:
            squeeze_output = False
        
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Extract bands (assuming RGB order)
        red = image[:, :, :, 0]
        green = image[:, :, :, 1]
        blue = image[:, :, :, 2]
        
        # Use green as NIR approximation for RGB images
        nir = green
        
        # Calculate EVI
        evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-8)
        evi = np.clip(evi, -1, 1)
        
        if squeeze_output:
            evi = evi[0]
        
        return evi
    
    def calculate_savi(self, image: np.ndarray, L: float = 0.5) -> np.ndarray:
        """
        Calculate Soil Adjusted Vegetation Index (SAVI).
        
        SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)
        
        Reduces soil brightness effects (useful for sparse vegetation).
        
        Args:
            image: Satellite image [H, W, C] or [N, H, W, C]
            L: Soil brightness correction factor (0-1, typically 0.5)
            
        Returns:
            SAVI array [H, W] or [N, H, W]
        """
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]
            squeeze_output = True
        else:
            squeeze_output = False
        
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        red = image[:, :, :, 0]
        green = image[:, :, :, 1]  # Use as NIR proxy
        
        savi = ((green - red) / (green + red + L + 1e-8)) * (1 + L)
        savi = np.clip(savi, -1, 1)
        
        if squeeze_output:
            savi = savi[0]
        
        return savi
    
    def classify_vegetation(self, ndvi: np.ndarray) -> np.ndarray:
        """
        Classify pixels based on NDVI values.
        
        Classification:
            0: Water/Non-vegetation (NDVI < 0)
            1: Bare soil (0 <= NDVI < 0.2)
            2: Sparse vegetation (0.2 <= NDVI < 0.4)
            3: Moderate vegetation (0.4 <= NDVI < 0.6)
            4: Dense vegetation/Forest (NDVI >= 0.6)
        
        Args:
            ndvi: NDVI array
            
        Returns:
            Classification map
        """
        classification = np.zeros_like(ndvi, dtype=np.uint8)
        
        classification[ndvi < 0] = 0
        classification[(ndvi >= 0) & (ndvi < 0.2)] = 1
        classification[(ndvi >= 0.2) & (ndvi < 0.4)] = 2
        classification[(ndvi >= 0.4) & (ndvi < 0.6)] = 3
        classification[ndvi >= 0.6] = 4
        
        return classification
    
    def detect_forest_cover(self, ndvi: np.ndarray, threshold: float = 0.4) -> np.ndarray:
        """
        Detect forest cover based on NDVI threshold.
        
        Args:
            ndvi: NDVI array
            threshold: NDVI threshold for forest (typically 0.4-0.6)
            
        Returns:
            Binary forest mask (1=forest, 0=non-forest)
        """
        forest_mask = (ndvi >= threshold).astype(np.uint8)
        return forest_mask
    
    def calculate_ndvi_change(self, 
                             ndvi_before: np.ndarray, 
                             ndvi_after: np.ndarray) -> np.ndarray:
        """
        Calculate NDVI change between two time periods.
        
        Args:
            ndvi_before: NDVI from earlier time period
            ndvi_after: NDVI from later time period
            
        Returns:
            NDVI change (negative values indicate vegetation loss)
        """
        ndvi_change = ndvi_after - ndvi_before
        return ndvi_change
    
    def detect_deforestation_by_ndvi(self,
                                     ndvi_before: np.ndarray,
                                     ndvi_after: np.ndarray,
                                     threshold: float = -0.15) -> np.ndarray:
        """
        Detect deforestation based on NDVI decrease.
        
        Args:
            ndvi_before: NDVI before deforestation
            ndvi_after: NDVI after deforestation
            threshold: NDVI decrease threshold (negative value)
            
        Returns:
            Binary deforestation mask
        """
        ndvi_change = self.calculate_ndvi_change(ndvi_before, ndvi_after)
        
        # Deforestation occurs where NDVI significantly decreased
        deforestation_mask = (ndvi_change < threshold).astype(np.uint8)
        
        return deforestation_mask
    
    def calculate_vegetation_area(self, 
                                 forest_mask: np.ndarray,
                                 pixel_size_m2: float = 100) -> float:
        """
        Calculate total vegetation area from forest mask.
        
        Args:
            forest_mask: Binary forest mask
            pixel_size_m2: Area of one pixel in square meters
            
        Returns:
            Total forest area in hectares
        """
        num_forest_pixels = np.sum(forest_mask)
        area_m2 = num_forest_pixels * pixel_size_m2
        area_ha = area_m2 / 10000  # Convert to hectares
        
        return area_ha


def visualize_ndvi(ndvi: np.ndarray, 
                  title: str = "NDVI Map",
                  save_path: Optional[str] = None):
    """
    Visualize NDVI map with appropriate color scale.
    
    Args:
        ndvi: NDVI array [H, W]
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    # Use RdYlGn colormap (Red-Yellow-Green)
    im = plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, label='NDVI Value')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved NDVI visualization to {save_path}")
    
    plt.tight_layout()
    plt.show()


def visualize_ndvi_comparison(ndvi_before: np.ndarray,
                              ndvi_after: np.ndarray,
                              ndvi_change: np.ndarray,
                              save_path: Optional[str] = None):
    """
    Visualize before/after NDVI comparison.
    
    Args:
        ndvi_before: NDVI before
        ndvi_after: NDVI after
        ndvi_change: NDVI change
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Before
    im1 = axes[0].imshow(ndvi_before, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[0].set_title('NDVI Before', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], label='NDVI')
    
    # After
    im2 = axes[1].imshow(ndvi_after, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1].set_title('NDVI After', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], label='NDVI')
    
    # Change
    im3 = axes[2].imshow(ndvi_change, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_title('NDVI Change (Deforestation)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], label='NDVI Change')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison to {save_path}")
    
    plt.show()


def generate_ndvi_statistics(ndvi: np.ndarray) -> dict:
    """
    Generate statistics for NDVI array.
    
    Args:
        ndvi: NDVI array
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'mean': np.mean(ndvi),
        'median': np.median(ndvi),
        'std': np.std(ndvi),
        'min': np.min(ndvi),
        'max': np.max(ndvi),
        'forest_pixels': np.sum(ndvi >= 0.4),
        'non_forest_pixels': np.sum(ndvi < 0.4),
        'forest_percentage': (np.sum(ndvi >= 0.4) / ndvi.size) * 100
    }
    
    return stats


if __name__ == "__main__":
    print("Vegetation Analysis Module")
    print("=" * 50)
    
    # Create dummy satellite image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # Simulate forest (high green values)
    image[50:150, 50:150, 1] = np.random.randint(150, 255, (100, 100))
    
    # Initialize calculator
    veg_calc = VegetationIndexCalculator()
    
    # Calculate NDVI
    ndvi = veg_calc.calculate_ndvi(image)
    print(f"NDVI shape: {ndvi.shape}")
    print(f"NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
    
    # Classify vegetation
    classification = veg_calc.classify_vegetation(ndvi)
    print(f"Classification classes: {np.unique(classification)}")
    
    # Get statistics
    stats = generate_ndvi_statistics(ndvi)
    print(f"\nNDVI Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Visualize
    visualize_ndvi(ndvi, title="Sample NDVI Map")
