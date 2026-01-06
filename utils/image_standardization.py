"""
IMAGE STANDARDIZATION MODULE
Ensures consistent preprocessing for deforestation detection

CRITICAL: Before/After images MUST be processed identically for:
1. Spatial alignment (same size, aspect ratio)
2. Pixel value normalization
3. Channel consistency
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImageStandardizer:
    """
    Standardizes images for deforestation detection model
    
    Ensures:
    - Fixed size (256x256)
    - RGB channels (3)
    - Normalized values [0, 1]
    - Proper aspect ratio handling
    - Consistent preprocessing for before/after images
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize image standardizer
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
        self.target_height, self.target_width = target_size
    
    def standardize_image(self, 
                         image: np.ndarray, 
                         preserve_aspect_ratio: bool = False,
                         pad_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Standardize a single image to model input format
        
        Args:
            image: Input image (H, W) or (H, W, C)
            preserve_aspect_ratio: If True, pad instead of stretch
            pad_color: Color for padding if preserving aspect ratio
            
        Returns:
            Standardized image (256, 256, 3) with values in [0, 1]
        """
        # Handle grayscale to RGB
        if len(image.shape) == 2:
            image = np.stack([image]*3, axis=-1)
        
        # Handle multispectral to RGB (take first 3 bands)
        if image.shape[-1] > 3:
            image = image[:, :, :3]
        
        # Handle single channel to RGB
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        # Ensure float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # Resize to target size
        if preserve_aspect_ratio:
            image = self._resize_with_padding(image, pad_color)
        else:
            # Direct resize (may distort aspect ratio)
            if image.shape[:2] != self.target_size:
                image = cv2.resize(
                    image, 
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_LINEAR
                )
        
        # Final shape check
        assert image.shape == (self.target_height, self.target_width, 3), \
            f"Expected {(self.target_height, self.target_width, 3)}, got {image.shape}"
        
        # Clip to valid range
        image = np.clip(image, 0.0, 1.0)
        
        return image
    
    def _resize_with_padding(self, 
                            image: np.ndarray, 
                            pad_color: Tuple[int, int, int]) -> np.ndarray:
        """
        Resize image while preserving aspect ratio using padding
        
        Args:
            image: Input image [0, 1]
            pad_color: Padding color
            
        Returns:
            Resized and padded image
        """
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scale to fit within target size
        scale = min(target_w / w, target_h / h)
        
        # New dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded canvas
        canvas = np.full(
            (target_h, target_w, 3), 
            np.array(pad_color) / 255.0, 
            dtype=np.float32
        )
        
        # Center the image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def standardize_pair(self, 
                        before_image: np.ndarray, 
                        after_image: np.ndarray,
                        preserve_aspect_ratio: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardize a pair of before/after images IDENTICALLY
        
        CRITICAL: Both images MUST be processed the same way for:
        - Spatial alignment
        - Pixel-level comparison
        - Change detection accuracy
        
        Args:
            before_image: Before satellite image
            after_image: After satellite image
            preserve_aspect_ratio: Whether to preserve aspect ratio
            
        Returns:
            (before_standardized, after_standardized) - both (256, 256, 3) in [0, 1]
        """
        # Check if images have same original dimensions
        if before_image.shape != after_image.shape:
            logger.warning(
                f"⚠️ Before/After images have different shapes: "
                f"{before_image.shape} vs {after_image.shape}. "
                f"Spatial alignment may be compromised!"
            )
        
        # Standardize both images with IDENTICAL processing
        before_std = self.standardize_image(
            before_image, 
            preserve_aspect_ratio=preserve_aspect_ratio
        )
        
        after_std = self.standardize_image(
            after_image, 
            preserve_aspect_ratio=preserve_aspect_ratio
        )
        
        return before_std, after_std
    
    def standardize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Standardize ground truth mask
        
        Args:
            mask: Binary mask (H, W) or (H, W, 1)
            
        Returns:
            Standardized mask (256, 256, 1) with values in {0, 1}
        """
        # Handle single channel
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        
        # Resize if needed (use NEAREST for binary masks!)
        if mask.shape[:2] != self.target_size:
            mask = cv2.resize(
                mask.astype(np.float32), 
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_NEAREST  # CRITICAL: Preserve binary values
            )
            mask = mask[:, :, np.newaxis]
        
        # Binarize (threshold at 0.5)
        mask_binary = (mask > 0.5).astype(np.float32)
        
        return mask_binary
    
    def validate_standardization(self, 
                                before: np.ndarray, 
                                after: np.ndarray,
                                verbose: bool = True) -> bool:
        """
        Validate that before/after images are properly standardized
        
        Args:
            before: Before image
            after: After image
            verbose: Print validation details
            
        Returns:
            True if valid, False otherwise
        """
        checks = {
            'same_shape': before.shape == after.shape,
            'correct_shape': before.shape == (self.target_height, self.target_width, 3),
            'correct_dtype': before.dtype == np.float32 and after.dtype == np.float32,
            'normalized': (before.max() <= 1.0) and (after.max() <= 1.0),
            'no_negative': (before.min() >= 0.0) and (after.min() >= 0.0),
        }
        
        if verbose:
            logger.info("🔍 Image Standardization Validation:")
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                logger.info(f"  {status} {check}: {passed}")
            
            if not checks['same_shape']:
                logger.error(f"  Before shape: {before.shape}, After shape: {after.shape}")
        
        return all(checks.values())


# Singleton instance for consistent use across modules
_standardizer = ImageStandardizer(target_size=(256, 256))


def standardize_image(image: np.ndarray, 
                     preserve_aspect_ratio: bool = False) -> np.ndarray:
    """
    Convenience function: Standardize a single image
    
    Args:
        image: Input image
        preserve_aspect_ratio: Whether to preserve aspect ratio
        
    Returns:
        Standardized image (256, 256, 3) in [0, 1]
    """
    return _standardizer.standardize_image(image, preserve_aspect_ratio)


def standardize_pair(before_image: np.ndarray, 
                    after_image: np.ndarray,
                    preserve_aspect_ratio: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function: Standardize before/after pair
    
    Args:
        before_image: Before satellite image
        after_image: After satellite image
        preserve_aspect_ratio: Whether to preserve aspect ratio
        
    Returns:
        (before_standardized, after_standardized)
    """
    return _standardizer.standardize_pair(before_image, after_image, preserve_aspect_ratio)


def standardize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convenience function: Standardize ground truth mask
    
    Args:
        mask: Binary mask
        
    Returns:
        Standardized mask (256, 256, 1)
    """
    return _standardizer.standardize_mask(mask)


def validate_standardization(before: np.ndarray, 
                            after: np.ndarray,
                            verbose: bool = True) -> bool:
    """
    Convenience function: Validate standardization
    
    Args:
        before: Before image
        after: After image
        verbose: Print validation details
        
    Returns:
        True if valid
    """
    return _standardizer.validate_standardization(before, after, verbose)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("IMAGE STANDARDIZATION MODULE - TEST")
    print("="*80)
    
    # Test with different image formats
    test_cases = [
        ("RGB 512x512", np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)),
        ("RGB 300x400", np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)),
        ("Grayscale 256x256", np.random.randint(0, 255, (256, 256), dtype=np.uint8)),
        ("Multispectral 256x256x13", np.random.randint(0, 255, (256, 256, 13), dtype=np.uint8)),
    ]
    
    standardizer = ImageStandardizer()
    
    for name, image in test_cases:
        print(f"\n📸 Test: {name}")
        print(f"   Input shape: {image.shape}, dtype: {image.dtype}")
        
        # Standardize
        std_image = standardizer.standardize_image(image)
        print(f"   Output shape: {std_image.shape}, dtype: {std_image.dtype}")
        print(f"   Value range: [{std_image.min():.3f}, {std_image.max():.3f}]")
        
        # Validate
        before = standardizer.standardize_image(image)
        after = standardizer.standardize_image(image)
        is_valid = standardizer.validate_standardization(before, after, verbose=False)
        print(f"   Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
    
    print("\n" + "="*80)
    print("✅ All tests completed!")
    print("="*80)
