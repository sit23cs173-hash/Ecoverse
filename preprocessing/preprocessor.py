"""
STEP 2: PREPROCESSING MODULE
Handles image preprocessing, normalization, alignment, and train/val/test splitting.
"""

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles all preprocessing operations for satellite imagery.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
        self.mean = None
        self.std = None
        
    def resize_images(self, images: np.ndarray) -> np.ndarray:
        """
        Resize images to target size.
        
        Args:
            images: Array of images [N, H, W, C]
            
        Returns:
            Resized images [N, target_h, target_w, C]
        """
        logger.info(f"Resizing {len(images)} images to {self.target_size}")
        
        resized = []
        for img in images:
            resized_img = cv2.resize(img, (self.target_size[1], self.target_size[0]))
            resized.append(resized_img)
        
        return np.array(resized)
    
    def normalize_images(self, 
                        images: np.ndarray, 
                        method: str = 'standard') -> np.ndarray:
        """
        Normalize images using various methods.
        
        Args:
            images: Array of images [N, H, W, C]
            method: Normalization method ('standard', 'minmax', '0-1')
            
        Returns:
            Normalized images
        """
        logger.info(f"Normalizing images using {method} method")
        
        if method == 'standard':
            # Z-score normalization
            if self.mean is None or self.std is None:
                self.mean = np.mean(images, axis=(0, 1, 2), keepdims=True)
                self.std = np.std(images, axis=(0, 1, 2), keepdims=True)
            
            normalized = (images - self.mean) / (self.std + 1e-8)
            
        elif method == 'minmax':
            # Min-max normalization to [-1, 1]
            min_val = np.min(images, axis=(0, 1, 2), keepdims=True)
            max_val = np.max(images, axis=(0, 1, 2), keepdims=True)
            normalized = 2 * (images - min_val) / (max_val - min_val + 1e-8) - 1
            
        elif method == '0-1':
            # Simple normalization to [0, 1]
            normalized = images.astype(np.float32) / 255.0
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def handle_missing_values(self, images: np.ndarray, fill_value: float = 0) -> np.ndarray:
        """
        Handle missing or NaN values in images.
        
        Args:
            images: Array of images
            fill_value: Value to fill NaN/missing values with
            
        Returns:
            Images with missing values filled
        """
        if np.isnan(images).any():
            logger.warning(f"Found {np.isnan(images).sum()} NaN values, filling with {fill_value}")
            images = np.nan_to_num(images, nan=fill_value)
        
        return images
    
    def remove_cloud_mask(self, 
                          images: np.ndarray, 
                          threshold: float = 0.9) -> np.ndarray:
        """
        Basic cloud masking using brightness threshold.
        Note: This is a simple heuristic. Advanced methods use separate cloud masks.
        
        Args:
            images: Array of images [N, H, W, C]
            threshold: Brightness threshold for cloud detection (0-1)
            
        Returns:
            Images with clouds masked (set to black)
        """
        logger.info("Applying basic cloud masking")
        
        # Normalize if not already
        if images.max() > 1.0:
            images_norm = images.astype(np.float32) / 255.0
        else:
            images_norm = images.copy()
        
        # Detect very bright pixels (potential clouds)
        brightness = np.mean(images_norm, axis=-1)  # [N, H, W]
        cloud_mask = brightness > threshold
        
        # Set cloud pixels to 0
        masked_images = images.copy()
        masked_images[cloud_mask] = 0
        
        logger.info(f"Masked {np.sum(cloud_mask)} cloud pixels")
        return masked_images
    
    def align_image_pairs(self, 
                         before: np.ndarray, 
                         after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align image pairs using feature matching.
        Useful when before/after images have slight misalignment.
        
        Args:
            before: Before images [N, H, W, C]
            after: After images [N, H, W, C]
            
        Returns:
            Aligned (before, after) image pairs
        """
        logger.info("Aligning image pairs...")
        
        aligned_before = []
        aligned_after = []
        
        for i in range(len(before)):
            before_gray = cv2.cvtColor(before[i], cv2.COLOR_RGB2GRAY)
            after_gray = cv2.cvtColor(after[i], cv2.COLOR_RGB2GRAY)
            
            # Detect ORB features
            orb = cv2.ORB_create(1000)
            kp1, des1 = orb.detectAndCompute(before_gray, None)
            kp2, des2 = orb.detectAndCompute(after_gray, None)
            
            if des1 is None or des2 is None:
                # Skip alignment if no features found
                aligned_before.append(before[i])
                aligned_after.append(after[i])
                continue
            
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) < 10:
                # Not enough matches, skip alignment
                aligned_before.append(before[i])
                aligned_after.append(after[i])
                continue
            
            # Extract matched points
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]])
            
            # Find homography
            H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
            
            if H is not None:
                # Warp after image to align with before
                h, w = before[i].shape[:2]
                after_aligned = cv2.warpPerspective(after[i], H, (w, h))
                aligned_after.append(after_aligned)
            else:
                aligned_after.append(after[i])
            
            aligned_before.append(before[i])
        
        logger.info(f"Aligned {len(aligned_before)} image pairs")
        return np.array(aligned_before), np.array(aligned_after)
    
    def augment_data(self, 
                     images: np.ndarray, 
                     masks: Optional[np.ndarray] = None,
                     augmentation_factor: int = 2) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply data augmentation to increase dataset size.
        
        Args:
            images: Array of images [N, H, W, C]
            masks: Optional array of masks [N, H, W]
            augmentation_factor: How many augmented versions per image
            
        Returns:
            Augmented images and masks
        """
        logger.info(f"Augmenting data with factor {augmentation_factor}")
        
        augmented_images = [images]
        augmented_masks = [masks] if masks is not None else []
        
        for _ in range(augmentation_factor - 1):
            aug_imgs = []
            aug_msks = [] if masks is not None else None
            
            for i in range(len(images)):
                img = images[i]
                msk = masks[i] if masks is not None else None
                
                # Random flip
                if np.random.rand() > 0.5:
                    img = np.fliplr(img)
                    if msk is not None:
                        msk = np.fliplr(msk)
                
                # Random rotation (90, 180, 270)
                k = np.random.randint(0, 4)
                img = np.rot90(img, k)
                if msk is not None:
                    msk = np.rot90(msk, k)
                
                # Random brightness adjustment
                brightness_factor = np.random.uniform(0.8, 1.2)
                img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)
                
                aug_imgs.append(img)
                if msk is not None:
                    aug_msks.append(msk)
            
            augmented_images.append(np.array(aug_imgs))
            if masks is not None:
                augmented_masks.append(np.array(aug_msks))
        
        # Concatenate all augmented versions
        final_images = np.concatenate(augmented_images, axis=0)
        final_masks = np.concatenate(augmented_masks, axis=0) if masks is not None else None
        
        logger.info(f"Augmented dataset size: {len(final_images)}")
        return final_images, final_masks


def split_dataset(X_before: np.ndarray,
                  X_after: np.ndarray,
                  y_masks: np.ndarray,
                  val_split: float = 0.2,
                  test_split: float = 0.1,
                  random_state: int = 42) -> Tuple:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        X_before: Before images
        X_after: After images
        y_masks: Ground truth masks
        val_split: Validation set proportion
        test_split: Test set proportion
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train, val, test) splits for before, after, and masks
    """
    logger.info(f"Splitting dataset: val={val_split}, test={test_split}")
    
    # First split: separate test set
    X_before_temp, X_before_test, X_after_temp, X_after_test, y_temp, y_test = train_test_split(
        X_before, X_after, y_masks, 
        test_size=test_split, 
        random_state=random_state
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_split / (1 - test_split)
    X_before_train, X_before_val, X_after_train, X_after_val, y_train, y_val = train_test_split(
        X_before_temp, X_after_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state
    )
    
    logger.info(f"Train: {len(X_before_train)}, Val: {len(X_before_val)}, Test: {len(X_before_test)}")
    
    return (X_before_train, X_after_train, y_train,
            X_before_val, X_after_val, y_val,
            X_before_test, X_after_test, y_test)


def create_change_image(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """
    Create change detection image by computing difference.
    
    Args:
        before: Before images [N, H, W, C]
        after: After images [N, H, W, C]
        
    Returns:
        Change images [N, H, W, C]
    """
    # Simple difference
    change = np.abs(after.astype(np.float32) - before.astype(np.float32))
    return change.astype(np.uint8)


if __name__ == "__main__":
    print("Image Preprocessing Module")
    print("=" * 50)
    
    # Create dummy data for testing
    before_images = np.random.randint(0, 255, (10, 128, 128, 3), dtype=np.uint8)
    after_images = np.random.randint(0, 255, (10, 128, 128, 3), dtype=np.uint8)
    masks = np.random.randint(0, 2, (10, 128, 128), dtype=np.uint8)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    
    # Test resizing
    resized_before = preprocessor.resize_images(before_images)
    print(f"Resized shape: {resized_before.shape}")
    
    # Test normalization
    normalized = preprocessor.normalize_images(resized_before, method='0-1')
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Test splitting
    splits = split_dataset(before_images, after_images, masks)
    print(f"Train set size: {len(splits[0])}")
