"""
Simple Change Detection Model Training
Uses a standard U-Net without custom layers for easier serialization
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
from pathlib import Path
import logging
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_simple_unet(input_shape=(256, 256, 6)):
    """Build a simple U-Net that takes 6-channel input (before+after concatenated)"""
    
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    # Block 1
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    p1 = layers.Dropout(0.1)(p1)
    
    # Block 2
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    p2 = layers.Dropout(0.1)(p2)
    
    # Block 3
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    p3 = layers.Dropout(0.2)(p3)
    
    # Block 4
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D(2)(c4)
    p4 = layers.Dropout(0.2)(p4)
    
    # Bridge
    c5 = layers.Conv2D(512, 3, activation='relu', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(512, 3, activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.3)(c5)
    
    # Decoder
    # Block 6
    u6 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = layers.Conv2D(256, 3, activation='relu', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(256, 3, activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Dropout(0.2)(c6)
    
    # Block 7
    u7 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = layers.Conv2D(128, 3, activation='relu', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(128, 3, activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Dropout(0.2)(c7)
    
    # Block 8
    u8 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = layers.Conv2D(64, 3, activation='relu', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(64, 3, activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Dropout(0.1)(c8)
    
    # Block 9
    u9 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = layers.Conv2D(32, 3, activation='relu', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(32, 3, activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)
    
    model = keras.Model(inputs, outputs, name='SimpleChangeUNet')
    return model

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

def load_training_data():
    """Load before/after image pairs and masks"""
    chips_dir = Path("data/generated_training/image_chips")
    masks_dir = Path("data/generated_training/mask_chips")
    
    chip_files = sorted(chips_dir.glob("*.png"))
    logger.info(f"Found {len(chip_files)} training chips")
    
    # Group by pair
    pairs = {}
    for f in chip_files:
        # Extract pair ID from filename
        parts = f.stem.split('_')
        if len(parts) >= 3:
            pair_id = parts[1]  # First ID
            chip_num = parts[-1]  # chip_XXX
            key = f"{pair_id}_{chip_num}"
            if key not in pairs:
                pairs[key] = {'chips': [], 'masks': []}
            pairs[key]['chips'].append(f)
            
            # Find corresponding mask
            mask_path = masks_dir / f.name
            if mask_path.exists():
                pairs[key]['masks'].append(mask_path)
    
    # Load images - use the after image as reference, simulate before with augmentation
    X_data = []
    y_data = []
    
    for i, chip_file in enumerate(chip_files[:300]):  # Limit to 300
        if i % 50 == 0:
            logger.info(f"  Loading {i}/{min(300, len(chip_files))}...")
        
        # Load after image
        after_img = cv2.imread(str(chip_file))
        if after_img is None:
            continue
        after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
        after_img = cv2.resize(after_img, (256, 256))
        after_img = after_img.astype(np.float32) / 255.0
        
        # Create simulated "before" image (more green, less deforested)
        before_img = after_img.copy()
        # Boost green channel slightly
        before_img[:, :, 1] = np.clip(before_img[:, :, 1] * 1.1, 0, 1)
        # Add slight blur to simulate temporal difference
        before_img = cv2.GaussianBlur(before_img, (3, 3), 0.5)
        
        # Load mask
        mask_path = masks_dir / chip_file.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256))
            mask = (mask > 127).astype(np.float32)
        else:
            # Create mask from difference
            diff = np.abs(after_img.mean(axis=2) - before_img.mean(axis=2))
            mask = (diff > 0.1).astype(np.float32)
        
        # Concatenate before and after into 6 channels
        combined = np.concatenate([before_img, after_img], axis=-1)  # (256, 256, 6)
        
        X_data.append(combined)
        y_data.append(mask)
    
    X = np.array(X_data)
    y = np.array(y_data)[..., np.newaxis]
    
    logger.info(f"Loaded {len(X)} samples")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Samples with deforestation: {np.sum(y.sum(axis=(1,2,3)) > 0)}/{len(y)}")
    
    return X, y

def main():
    logger.info("=" * 60)
    logger.info("SIMPLE CHANGE DETECTION MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load data
    X, y = load_training_data()
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Build model
    logger.info("\nBuilding Simple U-Net...")
    model = build_simple_unet(input_shape=(256, 256, 6))
    model.summary()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=combined_loss,
        metrics=['accuracy', dice_coef]
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_dice_coef',
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coef',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='max'
        ),
        keras.callbacks.ModelCheckpoint(
            'outputs/models/simple_change_best.keras',
            monitor='val_dice_coef',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Train
    logger.info("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=8,
        callbacks=callbacks
    )
    
    # Save final model in Keras format (better compatibility)
    output_path = Path("outputs/models/simple_change_model.keras")
    model.save(output_path)
    logger.info(f"\n✅ Model saved to: {output_path}")
    
    # Also save in H5 format for compatibility
    h5_path = Path("outputs/models/simple_change_model.h5")
    model.save(h5_path)
    logger.info(f"✅ Model also saved to: {h5_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best Validation Dice: {max(history.history['val_dice_coef']):.4f}")

if __name__ == "__main__":
    main()
