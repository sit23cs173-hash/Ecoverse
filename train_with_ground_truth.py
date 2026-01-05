"""
Train Model with REAL Ground Truth Labels
Uses actual labeled masks from Brazil Competition dataset
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from models.deforestation_model import DeforestationDetector
import cv2
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroundTruthModelTrainer:
    """Train model with real ground truth masks"""
    
    def __init__(self, model_type='unet'):
        self.model_type = model_type
        self.model = None
        
        # Dataset paths
        self.data_dir = Path('./data/raw/deforestation_competition')
        self.train_json = self.data_dir / 'train.json'
        
        # Create outputs directory
        self.model_dir = Path('./outputs/models')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_prepare_data(self, num_samples=300):
        """
        Load training data with real ground truth masks
        
        Args:
            num_samples: Number of samples to use for training
        """
        logger.info(f"Loading real ground truth data ({num_samples} samples)...")
        
        # Read metadata
        if not self.train_json.exists():
            raise FileNotFoundError(f"train.json not found at: {self.train_json}")
        
        with open(self.train_json, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Found {len(metadata)} total samples")
        
        # Use first N samples for training
        sample_indices = list(metadata.keys())[:num_samples]
        
        X_before = []
        X_after = []
        y_masks = []
        
        logger.info("Loading images and ground truth masks...")
        loaded_count = 0
        
        for i, idx in enumerate(sample_indices):
            if i % 50 == 0:
                logger.info(f"  Loading {i+1}/{len(sample_indices)}...")
            
            entry = metadata[idx]
            
            try:
                # Get file paths
                before_file = self.data_dir / 'train' / 'public' / entry['files']['satellite_img_first']
                after_file = self.data_dir / 'train' / 'public' / entry['files']['satellite_img_second']
                mask_file = self.data_dir / 'train' / 'public' / entry['files']['mask']
                
                # Check if all files exist
                if not all([before_file.exists(), after_file.exists(), mask_file.exists()]):
                    continue
                
                # Load arrays
                before_img = np.load(str(before_file))
                after_img = np.load(str(after_file))
                mask = np.load(str(mask_file))
                
                # Process function
                def process_image(img):
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    elif len(img.shape) == 3 and img.shape[-1] > 3:
                        img = img[:, :, :3]
                    
                    if img.max() > 255 or img.dtype != np.uint8:
                        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                    
                    img = cv2.resize(img, (256, 256))
                    return img
                
                before_processed = process_image(before_img)
                after_processed = process_image(after_img)
                
                # Process mask
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                
                mask_resized = cv2.resize(mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
                
                # Normalize mask to 0-1
                if mask_resized.max() > 1:
                    mask_resized = (mask_resized > 0).astype(np.float32)
                
                # Normalize images
                before_norm = before_processed.astype(np.float32) / 255.0
                after_norm = after_processed.astype(np.float32) / 255.0
                
                X_before.append(before_norm)
                X_after.append(after_norm)
                y_masks.append(mask_resized)
                
                loaded_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to load sample {idx}: {e}")
                continue
        
        logger.info(f"Successfully loaded {loaded_count} samples with ground truth")
        
        # Convert to numpy arrays
        X_before = np.array(X_before)
        X_after = np.array(X_after)
        y = np.expand_dims(np.array(y_masks), axis=-1)
        
        # Calculate statistics
        deforestation_pcts = [(mask.sum() / (256*256)) * 100 for mask in y_masks]
        logger.info(f"\nGround Truth Statistics:")
        logger.info(f"  Samples with deforestation: {sum(1 for pct in deforestation_pcts if pct > 0)}/{len(deforestation_pcts)}")
        logger.info(f"  Average deforestation: {np.mean(deforestation_pcts):.2f}%")
        logger.info(f"  Range: {np.min(deforestation_pcts):.2f}% - {np.max(deforestation_pcts):.2f}%")
        
        # Use AFTER images as input (deforestation state)
        X = X_after
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=(y.sum(axis=(1,2,3)) > 0).astype(int)
        )
        
        logger.info(f"\nData split:")
        logger.info(f"  Training: {X_train.shape}")
        logger.info(f"  Validation: {X_val.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def train(self, epochs=100, batch_size=16, num_samples=300):
        """
        Train model with real ground truth
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            num_samples: Number of samples to use
        """
        logger.info("="*80)
        logger.info("TRAINING WITH REAL GROUND TRUTH LABELS")
        logger.info(f"Dataset: Brazil Competition (Real Labels)")
        logger.info(f"Model: {self.model_type}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        logger.info("="*80)
        
        # Load data with real ground truth
        X_train, X_val, y_train, y_val = self.load_and_prepare_data(num_samples)
        
        # Initialize model
        logger.info(f"\nBuilding {self.model_type} model...")
        self.model = DeforestationDetector(
            input_shape=(256, 256, 3),
            model_type=self.model_type
        )
        self.model.build_model()
        
        # Use dice loss for better segmentation of sparse masks
        self.model.compile_model(learning_rate=0.001, loss='dice_loss')
        
        # Model save path
        model_path = self.model_dir / f'ground_truth_{self.model_type}_model.h5'
        
        # Train
        logger.info("\nStarting training with real ground truth...")
        history = self.model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_save_path=str(model_path)
        )
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"Model saved to: {model_path}")
        logger.info("="*80)
        
        # Print final metrics
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['accuracy'][-1]
        
        logger.info(f"\nFinal Training Metrics:")
        logger.info(f"  Loss: {final_train_loss:.4f}")
        logger.info(f"  Accuracy: {final_train_acc:.4f}")
        logger.info(f"\nFinal Validation Metrics:")
        logger.info(f"  Loss: {final_val_loss:.4f}")
        logger.info(f"  Accuracy: {final_val_acc:.4f}")
        
        return history


def main():
    """Main training script"""
    
    print("\n" + "="*80)
    print("🌲 DEFORESTATION MODEL TRAINING - REAL GROUND TRUTH")
    print("="*80 + "\n")
    
    print("Select configuration:")
    print("1. Full Training (300 samples, 100 epochs) - Recommended")
    print("2. Medium Training (150 samples, 50 epochs)")
    print("3. Quick Test (50 samples, 10 epochs)")
    print("4. Custom")
    
    choice = input("\nEnter choice (1-4, default=1): ").strip() or "1"
    
    if choice == "1":
        num_samples = 300
        epochs = 100
        batch_size = 16
    elif choice == "2":
        num_samples = 150
        epochs = 50
        batch_size = 16
    elif choice == "3":
        num_samples = 50
        epochs = 10
        batch_size = 8
    elif choice == "4":
        num_samples = int(input("Number of samples (default=300): ") or "300")
        epochs = int(input("Number of epochs (default=100): ") or "100")
        batch_size = int(input("Batch size (default=16): ") or "16")
    else:
        num_samples = 300
        epochs = 100
        batch_size = 16
    
    print(f"\n✅ Configuration:")
    print(f"   Samples: {num_samples}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Loss: Dice Loss (optimized for sparse masks)")
    
    confirm = input("\nProceed with training? (y/n, default=y): ").strip().lower() or "y"
    
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Initialize trainer
    trainer = GroundTruthModelTrainer(model_type='unet')
    
    # Train
    try:
        history = trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            num_samples=num_samples
        )
        
        print("\n" + "="*80)
        print("🎉 TRAINING SUCCESSFUL!")
        print("="*80)
        print("\n✅ Model trained with REAL ground truth labels!")
        print(f"📁 Model location: outputs/models/ground_truth_unet_model.h5")
        print("\n🔍 Next steps:")
        print("   1. Run: python evaluate_with_ground_truth.py")
        print("   2. Update model path in dashboard to use new model")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TRAINING FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
