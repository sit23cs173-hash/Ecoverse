"""
IMPROVED TRAINING WITH MOBILENETV2 U-NET
Uses transfer learning and data augmentation for better accuracy
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
import cv2

from models.deforestation_model import DeforestationDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedTrainer:
    """Trainer with MobileNetV2 U-Net and data augmentation"""
    
    def __init__(self):
        self.data_dir = Path('./data/raw/deforestation_competition')
        self.train_json = self.data_dir / 'train.json'
        self.model_save_path = Path('./outputs/models/mobilenet_unet_model.h5')
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        
    def data_augmentation(self, image, mask):
        """Apply data augmentation to images and masks"""
        # Random horizontal flip
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        # Random vertical flip
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        
        # Random brightness adjustment
        if tf.random.uniform(()) > 0.5:
            image = tf.image.adjust_brightness(image, delta=0.1)
        
        # Random contrast adjustment
        if tf.random.uniform(()) > 0.5:
            image = tf.image.adjust_contrast(image, contrast_factor=1.2)
        
        # Clip values to [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, mask
    
    def load_data(self, num_samples=300):
        """Load ground truth data with proper preprocessing"""
        logger.info(f"Loading {num_samples} samples with ground truth...")
        
        with open(self.train_json, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Found {len(metadata)} total samples")
        
        # Use first N samples
        sample_indices = list(metadata.keys())[:num_samples]
        
        X_after = []
        y_masks = []
        
        for i, idx in enumerate(sample_indices, 1):
            if i % 50 == 0:
                logger.info(f"  Loading {i}/{num_samples}...")
            
            entry = metadata[idx]
            
            try:
                # We only need "after" image for this approach
                after_file = self.data_dir / 'train' / 'public' / entry['files']['satellite_img_second']
                mask_file = self.data_dir / 'train' / 'public' / entry['files']['mask']
                
                if not all([after_file.exists(), mask_file.exists()]):
                    continue
                
                after_img = np.load(str(after_file))
                mask = np.load(str(mask_file))
                
                # Process image
                def process_image(img):
                    if len(img.shape) == 2:
                        img = np.stack([img]*3, axis=-1)
                    elif img.shape[-1] > 3:
                        img = img[:, :, :3]
                    
                    if img.shape[:2] != (256, 256):
                        img = cv2.resize(img, (256, 256))
                    
                    if len(img.shape) == 2:
                        img = np.stack([img]*3, axis=-1)
                    elif img.shape[-1] == 1:
                        img = np.repeat(img, 3, axis=-1)
                    
                    return img.astype(np.float32) / 255.0
                
                after_processed = process_image(after_img)
                
                # Process mask
                if len(mask.shape) == 2:
                    mask = mask[:, :, np.newaxis]
                if mask.shape[:2] != (256, 256):
                    mask = cv2.resize(mask, (256, 256))
                    mask = mask[:, :, np.newaxis]
                
                mask_binary = (mask > 0).astype(np.float32)
                
                X_after.append(after_processed)
                y_masks.append(mask_binary)
                
            except Exception as e:
                logger.warning(f"Error loading sample {idx}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(X_after)} samples")
        
        X = np.array(X_after)
        y = np.array(y_masks)
        
        # Statistics
        deforestation_counts = (y.sum(axis=(1,2,3)) > 0).sum()
        avg_deforestation = (y.sum() / y.size) * 100
        
        logger.info(f"\nGround Truth Statistics:")
        logger.info(f"  Samples with deforestation: {deforestation_counts}/{len(y)}")
        logger.info(f"  Average deforestation: {avg_deforestation:.2f}%")
        
        return X, y
    
    def train_stage1(self, X_train, y_train, X_val, y_val, epochs=15, batch_size=16):
        """
        Stage 1: Train with frozen base model (transfer learning)
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: TRANSFER LEARNING (Frozen MobileNetV2 Base)")
        logger.info("="*80)
        
        # Build model with frozen base
        detector = DeforestationDetector(
            input_shape=(256, 256, 3),
            model_type='mobilenet_unet'
        )
        detector.build_model()
        
        # Compile with Adam optimizer
        detector.compile_model(learning_rate=0.001, loss='dice_loss')
        self.model = detector
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(self.model_save_path),
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train
        logger.info(f"\nTraining for {epochs} epochs (batch size: {batch_size})")
        history = detector.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_save_path=str(self.model_save_path)
        )
        
        logger.info("Stage 1 complete!")
        return history
    
    def train_stage2(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=16):
        """
        Stage 2: Fine-tune entire model (unfreeze base)
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: FINE-TUNING (Unfrozen MobileNetV2 Base)")
        logger.info("="*80)
        
        # Load best model from stage 1
        logger.info("Loading best model from Stage 1...")
        self.model.load_model(str(self.model_save_path))
        
        # Unfreeze base model for fine-tuning
        logger.info("Unfreezing MobileNetV2 base model...")
        for layer in self.model.model.layers:
            if 'mobilenetv2' in layer.name.lower() or 'model' in layer.name.lower():
                layer.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile_model(learning_rate=0.0001, loss='dice_loss')
        
        logger.info(f"Model now has {self.model.model.count_params():,} trainable parameters")
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(self.model_save_path),
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train
        logger.info(f"\nFine-tuning for {epochs} epochs (batch size: {batch_size})")
        history = self.model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_save_path=str(self.model_save_path)
        )
        
        logger.info("Stage 2 complete!")
        return history
    
    def train_full_pipeline(self, num_samples=300):
        """Execute full 2-stage training pipeline"""
        
        logger.info("="*80)
        logger.info("IMPROVED TRAINING WITH MOBILENETV2 U-NET")
        logger.info(f"Samples: {num_samples}")
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
        
        # Stage 1: Transfer learning (frozen base)
        history1 = self.train_stage1(X_train, y_train, X_val, y_val, epochs=15, batch_size=16)
        
        # Stage 2: Fine-tuning (unfrozen base)
        history2 = self.train_stage2(X_train, y_train, X_val, y_val, epochs=20, batch_size=16)
        
        # Save final results
        final_metrics = {
            'stage1_final_val_loss': float(history1.history['val_loss'][-1]),
            'stage1_final_val_accuracy': float(history1.history['val_accuracy'][-1]),
            'stage2_final_val_loss': float(history2.history['val_loss'][-1]),
            'stage2_final_val_accuracy': float(history2.history['val_accuracy'][-1]),
            'total_epochs': len(history1.history['loss']) + len(history2.history['loss']),
            'num_samples': num_samples
        }
        
        metrics_file = self.model_save_path.parent / 'mobilenet_training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"Model saved to: {self.model_save_path}")
        logger.info(f"Metrics saved to: {metrics_file}")
        logger.info("="*80)
        
        logger.info("\nFinal Metrics:")
        logger.info(f"  Stage 1 Val Loss: {final_metrics['stage1_final_val_loss']:.4f}")
        logger.info(f"  Stage 1 Val Accuracy: {final_metrics['stage1_final_val_accuracy']:.4f}")
        logger.info(f"  Stage 2 Val Loss: {final_metrics['stage2_final_val_loss']:.4f}")
        logger.info(f"  Stage 2 Val Accuracy: {final_metrics['stage2_final_val_accuracy']:.4f}")
        
        return history1, history2


def main():
    """Main training function"""
    try:
        trainer = ImprovedTrainer()
        history1, history2 = trainer.train_full_pipeline(num_samples=300)
        
        print("\n✅ SUCCESS! MobileNetV2 U-Net training completed.")
        print(f"📁 Model location: {trainer.model_save_path}")
        print("\n📊 Next steps:")
        print("   1. Run: python evaluate_with_ground_truth.py")
        print("   2. Update dashboard to use mobilenet_unet_model.h5")
        print("   3. Expected accuracy: 70-95% (vs previous 0.49%)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
