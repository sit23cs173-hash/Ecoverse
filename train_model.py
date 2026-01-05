"""
Train Deep Learning Model for Deforestation Detection
Uses real datasets (Amazon or Brazil Competition) to train U-Net/Siamese models
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from models.deforestation_model import DeforestationDetector
from data.data_loader import DeforestationDataLoader
from preprocessing.preprocessor import ImagePreprocessor
import cv2
import rasterio
from rasterio.plot import reshape_as_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train deforestation detection models on real datasets"""
    
    def __init__(self, dataset_type='competition', model_type='unet'):
        """
        Initialize trainer
        
        Args:
            dataset_type: 'amazon' or 'competition'
            model_type: 'unet' or 'siamese'
        """
        self.dataset_type = dataset_type
        self.model_type = model_type
        self.data_loader = DeforestationDataLoader(str(Path.cwd() / 'data'))
        self.preprocessor = ImagePreprocessor()
        self.model = None
        
        # Create outputs directory
        self.model_dir = Path.cwd() / 'outputs' / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_prepare_data(self, num_samples=100):
        """
        Load and prepare training data
        
        Args:
            num_samples: Number of image pairs to use
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        logger.info(f"Loading {self.dataset_type} dataset...")
        
        if self.dataset_type == 'competition':
            # Load Brazil Competition dataset
            from data.data_loader import COMPETITION_DATASET_PATH
            data = self.data_loader.load_kaggle_dataset(str(COMPETITION_DATASET_PATH), 'competition')
            
            train_paths = data.get('train_paths', [])
            logger.info(f"Found {len(train_paths)} training samples")
            
            # Limit samples if too many
            train_paths = train_paths[:min(num_samples, len(train_paths))]
            
            # Load images
            images = []
            masks = []
            
            for i, path in enumerate(train_paths):
                if i % 10 == 0:
                    logger.info(f"Loading sample {i+1}/{len(train_paths)}...")
                
                try:
                    # Load .npy file (could be 2D or 3D)
                    data_array = np.load(str(path))
                    
                    # Handle different data shapes
                    if len(data_array.shape) == 2:
                        # 2D grayscale image, convert to 3-channel
                        img = cv2.cvtColor(data_array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    elif len(data_array.shape) == 3:
                        # Extract RGB bands (first 3 channels)
                        img = data_array[:, :, :3]
                    else:
                        logger.warning(f"Unexpected shape {data_array.shape} for {path}")
                        continue
                    
                    # Normalize to 0-255
                    if img.max() > 255:
                        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    
                    # Resize to model input size
                    img = cv2.resize(img, (256, 256))
                    
                    # Create synthetic deforestation mask
                    if len(data_array.shape) == 3 and data_array.shape[-1] >= 4:
                        # Calculate NDVI from NIR (band 3) and Red (band 2)
                        nir = data_array[:, :, 3].astype(float)
                        red = data_array[:, :, 2].astype(float)
                        ndvi = (nir - red) / (nir + red + 1e-8)
                        
                        # Create mask (NDVI < 0.3 = deforested)
                        mask = (ndvi < 0.3).astype(np.uint8)
                        mask = cv2.resize(mask, (256, 256))
                    else:
                        # Simple threshold-based mask for grayscale or RGB
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        mask = (gray < 100).astype(np.uint8)
                    
                    images.append(img)
                    masks.append(mask)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(images)} image-mask pairs")
            
        elif self.dataset_type == 'amazon':
            # Load Amazon Sentinel-2 dataset
            from data.data_loader import AMAZON_DATASET_PATH
            data = self.data_loader.load_kaggle_dataset(str(AMAZON_DATASET_PATH), 'amazon')
            
            image_paths = data.get('image_paths', [])
            logger.info(f"Found {len(image_paths)} Amazon images")
            
            # Limit samples
            image_paths = image_paths[:min(num_samples, len(image_paths))]
            
            images = []
            masks = []
            
            for i, path in enumerate(image_paths):
                if i % 5 == 0:
                    logger.info(f"Loading image {i+1}/{len(image_paths)}...")
                
                try:
                    # Load GeoTIFF with rasterio
                    with rasterio.open(str(path)) as src:
                        data_array = src.read()
                        
                        # Extract RGB
                        if data_array.shape[0] >= 3:
                            img = np.stack([data_array[2], data_array[1], data_array[0]], axis=-1)
                        else:
                            img = reshape_as_image(data_array)
                        
                        # Normalize
                        if img.max() > 255:
                            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                        
                        # Resize
                        img = cv2.resize(img, (256, 256))
                        
                        # Create NDVI-based mask
                        if data_array.shape[0] >= 4:
                            nir = data_array[3].astype(float)
                            red = data_array[2].astype(float)
                            ndvi = (nir - red) / (nir + red + 1e-8)
                            mask = (ndvi < 0.3).astype(np.uint8)
                            mask = cv2.resize(mask, (256, 256))
                        else:
                            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                            mask = (gray < 100).astype(np.uint8)
                        
                        images.append(img)
                        masks.append(mask)
                        
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(images)} Amazon image-mask pairs")
        
        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        y = np.array(masks, dtype=np.float32)
        
        # Expand mask dimensions if needed
        if len(y.shape) == 3:
            y = np.expand_dims(y, axis=-1)
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def train(self, epochs=50, batch_size=8, num_samples=100):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            num_samples: Number of samples to use
        """
        logger.info("="*80)
        logger.info("STARTING MODEL TRAINING")
        logger.info(f"Dataset: {self.dataset_type}")
        logger.info(f"Model: {self.model_type}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        logger.info("="*80)
        
        # Load data
        X_train, X_val, y_train, y_val = self.load_and_prepare_data(num_samples)
        
        # Initialize model
        logger.info(f"\nBuilding {self.model_type} model...")
        self.model = DeforestationDetector(
            input_shape=(256, 256, 3),
            model_type=self.model_type
        )
        self.model.build_model()
        self.model.compile_model()  # Compile the model before training
        
        # Model save path
        model_path = self.model_dir / f'{self.dataset_type}_{self.model_type}_model.h5'
        
        # Train
        logger.info("\nStarting training...")
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
        logger.info(f"\nFinal Training Loss: {final_train_loss:.4f}")
        logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate trained model"""
        if self.model is None:
            logger.error("No model trained yet!")
            return
        
        logger.info("Evaluating model...")
        results = self.model.evaluate(X_test, y_test)
        logger.info(f"Test Loss: {results['loss']:.4f}")
        logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
        
        return results


def main():
    """Main training script"""
    
    print("\n" + "="*80)
    print("🌲 DEFORESTATION DETECTION MODEL TRAINING 🌲")
    print("="*80 + "\n")
    
    # Configuration
    print("Select configuration:")
    print("1. Brazil Competition Dataset + U-Net (Recommended)")
    print("2. Brazil Competition Dataset + Siamese Network")
    print("3. Amazon Dataset + U-Net")
    print("4. Amazon Dataset + Siamese Network")
    print("5. Quick Test (10 samples, 5 epochs)")
    
    choice = input("\nEnter choice (1-5, default=1): ").strip() or "1"
    
    # Set parameters based on choice
    if choice == "1":
        dataset_type = 'competition'
        model_type = 'unet'
        epochs = 50
        batch_size = 8
        num_samples = 100
    elif choice == "2":
        dataset_type = 'competition'
        model_type = 'siamese'
        epochs = 50
        batch_size = 8
        num_samples = 100
    elif choice == "3":
        dataset_type = 'amazon'
        model_type = 'unet'
        epochs = 50
        batch_size = 4
        num_samples = 50
    elif choice == "4":
        dataset_type = 'amazon'
        model_type = 'siamese'
        epochs = 50
        batch_size = 4
        num_samples = 50
    elif choice == "5":
        dataset_type = 'competition'
        model_type = 'unet'
        epochs = 5
        batch_size = 4
        num_samples = 10
    else:
        print("Invalid choice, using defaults")
        dataset_type = 'competition'
        model_type = 'unet'
        epochs = 50
        batch_size = 8
        num_samples = 100
    
    print(f"\n✅ Configuration:")
    print(f"   Dataset: {dataset_type}")
    print(f"   Model: {model_type}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Samples: {num_samples}")
    
    confirm = input("\nProceed with training? (y/n, default=y): ").strip().lower() or "y"
    
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Initialize trainer
    trainer = ModelTrainer(dataset_type=dataset_type, model_type=model_type)
    
    # Train
    try:
        history = trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            num_samples=num_samples
        )
        
        print("\n" + "="*80)
        print("🎉 TRAINING SUCCESSFUL! 🎉")
        print("="*80)
        print("\nYou can now use the trained model in your dashboard!")
        print(f"Model location: outputs/models/{dataset_type}_{model_type}_model.h5")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TRAINING FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
