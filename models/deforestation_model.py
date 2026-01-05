"""
STEP 4: DEFORESTATION DETECTION MODEL
Implements CNN-based and traditional ML approaches for deforestation detection.
Includes change detection using before/after satellite images.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
from typing import Tuple, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeforestationDetector:
    """
    Deep learning model for deforestation detection using change detection.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (256, 256, 3),
                 model_type: str = 'unet'):
        """
        Initialize deforestation detector.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            model_type: Model architecture ('unet', 'siamese', 'simple_cnn')
        """
        self.input_shape = input_shape
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def build_unet_model(self) -> keras.Model:
        """
        Build U-Net architecture for semantic segmentation.
        U-Net is excellent for pixel-wise classification (deforestation mask prediction).
        """
        logger.info("Building U-Net model")
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder (Contracting Path)
        # Block 1
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        # Block 2
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        # Block 3
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        # Block 4
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        # Bottleneck
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
        
        # Decoder (Expanding Path)
        # Block 6
        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
        
        # Block 7
        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
        
        # Block 8
        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
        
        # Block 9
        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
        
        # Output layer - binary segmentation
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        
        model = keras.Model(inputs=[inputs], outputs=[outputs])
        return model
    
    def build_siamese_network(self) -> keras.Model:
        """
        Build Siamese network for change detection.
        Processes before/after images separately then compares features.
        """
        logger.info("Building Siamese network")
        
        # Shared feature extractor
        def create_feature_extractor():
            base_model = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False  # Freeze pretrained weights
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5)
            ])
            return model
        
        # Define inputs for before and after images
        input_before = layers.Input(shape=self.input_shape, name='before')
        input_after = layers.Input(shape=self.input_shape, name='after')
        
        # Shared feature extractor
        feature_extractor = create_feature_extractor()
        
        # Extract features from both images
        features_before = feature_extractor(input_before)
        features_after = feature_extractor(input_after)
        
        # Combine features (concatenate or subtract)
        combined = layers.concatenate([features_before, features_after])
        
        # Classification head
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=[input_before, input_after], outputs=output)
        return model
    
    def build_simple_cnn(self) -> keras.Model:
        """
        Build simple CNN for change detection.
        Takes concatenated before+after images as input.
        """
        logger.info("Building Simple CNN model")
        
        # Input is concatenated before+after (6 channels)
        input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2] * 2)
        inputs = layers.Input(shape=input_shape)
        
        # Convolutional layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=output)
        return model
    
    def build_model(self):
        """Build model based on specified architecture."""
        if self.model_type == 'unet':
            self.model = self.build_unet_model()
        elif self.model_type == 'siamese':
            self.model = self.build_siamese_network()
        elif self.model_type == 'simple_cnn':
            self.model = self.build_simple_cnn()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Model built: {self.model_type}")
        return self.model
    
    def compile_model(self, 
                     learning_rate: float = 0.001,
                     loss: str = 'binary_crossentropy'):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for optimizer
            loss: Loss function ('binary_crossentropy', 'dice_loss')
        """
        if self.model is None:
            self.build_model()
        
        # Define custom dice loss if needed
        if loss == 'dice_loss':
            def dice_loss(y_true, y_pred):
                numerator = 2 * tf.reduce_sum(y_true * y_pred)
                denominator = tf.reduce_sum(y_true + y_pred)
                return 1 - (numerator + 1) / (denominator + 1)
            
            loss_fn = dice_loss
        else:
            loss_fn = loss
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        logger.info("Model compiled")
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 16,
              model_save_path: str = './outputs/models/best_model.h5'):
        """
        Train the deforestation detection model.
        
        Args:
            X_train: Training images
            y_train: Training labels/masks
            X_val: Validation images
            y_val: Validation labels/masks
            epochs: Number of training epochs
            batch_size: Batch size
            model_save_path: Path to save best model
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model on test set")
        
        # Predict
        y_pred = self.model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Flatten for metrics calculation
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred_binary.flatten()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test_flat, y_pred_flat),
            'precision': precision_score(y_test_flat, y_pred_flat, zero_division=0),
            'recall': recall_score(y_test_flat, y_pred_flat, zero_division=0),
            'f1_score': f1_score(y_test_flat, y_pred_flat, zero_division=0)
        }
        
        # IoU (Intersection over Union) for segmentation
        intersection = np.logical_and(y_test_flat, y_pred_flat).sum()
        union = np.logical_or(y_test_flat, y_pred_flat).sum()
        metrics['iou'] = intersection / (union + 1e-8)
        
        logger.info("Evaluation metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Predict deforestation masks for input images.
        
        Args:
            images: Input images
            
        Returns:
            Predicted masks
        """
        predictions = self.model.predict(images)
        return predictions
    
    def save_model(self, path: str):
        """Save trained model."""
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model with custom loss support."""
        # Define dice loss for loading
        def dice_loss(y_true, y_pred):
            numerator = 2 * tf.reduce_sum(y_true * y_pred)
            denominator = tf.reduce_sum(y_true + y_pred)
            return 1 - (numerator + 1) / (denominator + 1)
        
        # Load model with custom objects
        self.model = keras.models.load_model(
            path,
            custom_objects={'dice_loss': dice_loss}
        )
        logger.info(f"Model loaded from {path}")


# ============================================================================
# TRADITIONAL ML APPROACH (NDVI-BASED)
# ============================================================================

class NDVIBasedDetector:
    """
    Traditional ML approach using NDVI difference and thresholding.
    Simpler but interpretable method for deforestation detection.
    """
    
    def __init__(self, ndvi_threshold: float = -0.15):
        """
        Initialize NDVI-based detector.
        
        Args:
            ndvi_threshold: Threshold for NDVI decrease indicating deforestation
        """
        self.ndvi_threshold = ndvi_threshold
    
    def detect_deforestation(self,
                            ndvi_before: np.ndarray,
                            ndvi_after: np.ndarray) -> np.ndarray:
        """
        Detect deforestation using NDVI change.
        
        Args:
            ndvi_before: NDVI before
            ndvi_after: NDVI after
            
        Returns:
            Binary deforestation mask
        """
        ndvi_change = ndvi_after - ndvi_before
        deforestation_mask = (ndvi_change < self.ndvi_threshold).astype(np.uint8)
        
        return deforestation_mask
    
    def evaluate(self, 
                y_true: np.ndarray, 
                y_pred: np.ndarray) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            
        Returns:
            Evaluation metrics
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_true_flat, y_pred_flat),
            'precision': precision_score(y_true_flat, y_pred_flat, zero_division=0),
            'recall': recall_score(y_true_flat, y_pred_flat, zero_division=0),
            'f1_score': f1_score(y_true_flat, y_pred_flat, zero_division=0)
        }
        
        return metrics


if __name__ == "__main__":
    print("Deforestation Detection Model")
    print("=" * 50)
    
    # Create dummy data
    X_train = np.random.rand(10, 256, 256, 3).astype(np.float32)
    y_train = np.random.randint(0, 2, (10, 256, 256, 1)).astype(np.float32)
    
    # Build and compile model
    detector = DeforestationDetector(input_shape=(256, 256, 3), model_type='unet')
    detector.build_model()
    detector.compile_model()
    
    print(f"Model type: {detector.model_type}")
    print(f"Model parameters: {detector.model.count_params():,}")
    
    # Model summary
    detector.model.summary()
