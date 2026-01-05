"""
AUTOMATED FULL TRAINING - NO USER INPUT REQUIRED
Trains U-Net model with full configuration (300 samples, 100 epochs)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
import sys

from models.deforestation_model import DeforestationDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


class AutoTrainer:
    """Automated trainer with no user interaction"""
    
    def __init__(self):
        self.data_dir = Path('./data/raw/deforestation_competition')
        self.train_json = self.data_dir / 'train.json'
        self.model_save_path = Path('./outputs/models/ground_truth_unet_model.h5')
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        
    def load_data(self, num_samples=300):
        """Load ground truth data"""
        logger.info(f"Loading {num_samples} samples with ground truth...")
        
        with open(self.train_json, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Found {len(metadata)} total samples")
        
        # Use first N samples
        sample_indices = list(metadata.keys())[:num_samples]
        
        X_before = []
        X_after = []
        y_masks = []
        
        for i, idx in enumerate(sample_indices, 1):
            if i % 50 == 0:
                logger.info(f"  Loading {i}/{num_samples}...")
            
            entry = metadata[idx]
            
            try:
                before_file = self.data_dir / 'train' / 'public' / entry['files']['satellite_img_first']
                after_file = self.data_dir / 'train' / 'public' / entry['files']['satellite_img_second']
                mask_file = self.data_dir / 'train' / 'public' / entry['files']['mask']
                
                if not all([before_file.exists(), after_file.exists(), mask_file.exists()]):
                    continue
                
                before_img = np.load(str(before_file))
                after_img = np.load(str(after_file))
                mask = np.load(str(mask_file))
                
                # Process
                def process_image(img):
                    # Handle different input shapes
                    if len(img.shape) == 2:
                        img = np.stack([img]*3, axis=-1)
                    elif img.shape[-1] > 3:
                        # Take only first 3 channels (RGB)
                        img = img[:, :, :3]
                    
                    if img.shape[:2] != (256, 256):
                        import cv2
                        img = cv2.resize(img, (256, 256))
                    
                    # Ensure exactly 3 channels
                    if len(img.shape) == 2:
                        img = np.stack([img]*3, axis=-1)
                    elif img.shape[-1] == 1:
                        img = np.repeat(img, 3, axis=-1)
                    
                    return img.astype(np.float32) / 255.0
                
                before_processed = process_image(before_img)
                after_processed = process_image(after_img)
                
                if len(mask.shape) == 2:
                    mask = mask[:, :, np.newaxis]
                if mask.shape[:2] != (256, 256):
                    import cv2
                    mask = cv2.resize(mask, (256, 256))
                    mask = mask[:, :, np.newaxis]
                
                mask_binary = (mask > 0).astype(np.float32)
                
                X_before.append(before_processed)
                X_after.append(after_processed)
                y_masks.append(mask_binary)
                
            except Exception as e:
                logger.warning(f"Error loading sample {idx}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(X_after)} samples")
        
        # Stack and combine
        X_after = np.array(X_after)
        y = np.array(y_masks)
        
        # Statistics
        deforestation_counts = (y.sum(axis=(1,2,3)) > 0).sum()
        avg_deforestation = (y.sum() / y.size) * 100
        
        logger.info(f"\nGround Truth Statistics:")
        logger.info(f"  Samples with deforestation: {deforestation_counts}/{len(y)}")
        logger.info(f"  Average deforestation: {avg_deforestation:.2f}%")
        
        return X_after, y
    
    def train(self, num_samples=300, epochs=100, batch_size=16):
        """Train model automatically"""
        
        logger.info("="*80)
        logger.info("AUTOMATED FULL TRAINING - GROUND TRUTH DATA")
        logger.info(f"Samples: {num_samples}, Epochs: {epochs}, Batch: {batch_size}")
        logger.info("="*80)
        
        # Load data
        X, y = self.load_data(num_samples)
        
        # Split data (stratified)
        stratify_labels = (y.sum(axis=(1,2,3)) > 0).astype(int)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_labels
        )
        
        logger.info(f"\nData split:")
        logger.info(f"  Training: {X_train.shape}")
        logger.info(f"  Validation: {X_val.shape}")
        
        # Build model
        logger.info("\nBuilding U-Net model...")
        detector = DeforestationDetector(input_shape=(256, 256, 3), model_type='unet')
        detector.build_model()
        detector.compile_model(learning_rate=0.001, loss='dice_loss')
        self.model = detector
        
        # Train
        logger.info("\nStarting training...")
        history = detector.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_save_path=str(self.model_save_path)
        )
        
        # Save final results
        final_metrics = {
            'train_loss': float(history.history['loss'][-1]),
            'train_accuracy': float(history.history['accuracy'][-1]),
            'val_loss': float(history.history['val_loss'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1]),
            'epochs': epochs,
            'samples': num_samples,
            'batch_size': batch_size
        }
        
        metrics_file = self.model_save_path.parent / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"Model saved to: {self.model_save_path}")
        logger.info(f"Metrics saved to: {metrics_file}")
        logger.info("="*80)
        
        return history


def main():
    """Main training function"""
    try:
        trainer = AutoTrainer()
        history = trainer.train(
            num_samples=300,
            epochs=100,
            batch_size=16
        )
        
        print("\n✅ SUCCESS! Model training completed.")
        print(f"📁 Model location: {trainer.model_save_path}")
        print("\n📊 Next steps:")
        print("   1. Run: python evaluate_with_ground_truth.py")
        print("   2. Dashboard will automatically use the improved model")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("Partial model may be saved in outputs/models/")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
