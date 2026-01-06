"""
CHANGE DETECTION MODEL TRAINING
Trains a U-Net model that takes BOTH before and after images (6 channels)
to properly detect deforestation changes.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path
import json
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
import cv2

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def dice_coef(y_true, y_pred, smooth=1):
    """Dice coefficient for binary segmentation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coef(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """Combined BCE + Dice loss for better training"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice


def build_change_detection_unet(input_shape=(256, 256, 6)):
    """
    Build a U-Net that takes 6 channels (before + after concatenated)
    to detect changes/deforestation between the two images.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Split into before and after for feature extraction
    before_img = inputs[:, :, :, :3]
    after_img = inputs[:, :, :, 3:]
    
    # Compute difference features
    diff_features = layers.Subtract()([after_img, before_img])
    abs_diff = layers.Lambda(lambda x: tf.abs(x))(diff_features)
    
    # Concatenate: before, after, and difference
    combined = layers.Concatenate()([before_img, after_img, abs_diff])  # 9 channels
    
    # Encoder (Contracting Path)
    # Block 1
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(combined)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.1)(p1)
    
    # Block 2
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.1)(p2)
    
    # Block 3
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(0.2)(p3)
    
    # Block 4
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(0.2)(p4)
    
    # Bridge (Bottleneck)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.3)(c5)
    
    # Decoder (Expanding Path)
    # Block 6
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Dropout(0.2)(c6)
    
    # Block 7
    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Dropout(0.2)(c7)
    
    # Block 8
    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Dropout(0.1)(c8)
    
    # Block 9
    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)
    
    # Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs, outputs, name='ChangeDetectionUNet')
    return model


def load_before_after_pairs(data_dir, num_samples=300):
    """
    Load before/after image pairs with deforestation masks.
    Creates proper change detection training data.
    """
    data_dir = Path(data_dir)
    train_json = data_dir / 'train.json'
    
    logger.info(f"Loading {num_samples} before/after pairs...")
    
    with open(train_json, 'r') as f:
        metadata = json.load(f)
    
    sample_indices = list(metadata.keys())[:num_samples]
    
    X_pairs = []  # Will contain (before, after) concatenated
    y_masks = []
    
    for i, idx in enumerate(sample_indices, 1):
        if i % 50 == 0:
            logger.info(f"  Loading {i}/{num_samples}...")
        
        entry = metadata[idx]
        
        try:
            before_file = data_dir / 'train' / 'public' / entry['files']['satellite_img_first']
            after_file = data_dir / 'train' / 'public' / entry['files']['satellite_img_second']
            mask_file = data_dir / 'train' / 'public' / entry['files']['mask']
            
            if not all([before_file.exists(), after_file.exists(), mask_file.exists()]):
                continue
            
            before_img = np.load(str(before_file))
            after_img = np.load(str(after_file))
            mask = np.load(str(mask_file))
            
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
            
            before_processed = process_image(before_img)
            after_processed = process_image(after_img)
            
            # Concatenate before and after (6 channels)
            pair = np.concatenate([before_processed, after_processed], axis=-1)
            
            # Process mask
            if len(mask.shape) == 2:
                mask = mask[:, :, np.newaxis]
            if mask.shape[:2] != (256, 256):
                mask = cv2.resize(mask, (256, 256))
                mask = mask[:, :, np.newaxis]
            
            mask_binary = (mask > 0).astype(np.float32)
            
            X_pairs.append(pair)
            y_masks.append(mask_binary)
            
        except Exception as e:
            logger.warning(f"Error loading sample {idx}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(X_pairs)} before/after pairs")
    
    X = np.array(X_pairs)
    y = np.array(y_masks)
    
    # Statistics
    deforestation_counts = (y.sum(axis=(1,2,3)) > 0).sum()
    avg_deforestation = (y.sum() / y.size) * 100
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Input shape: {X.shape} (before+after, 6 channels)")
    logger.info(f"  Mask shape: {y.shape}")
    logger.info(f"  Samples with deforestation: {deforestation_counts}/{len(y)}")
    logger.info(f"  Average deforestation: {avg_deforestation:.2f}%")
    
    return X, y


def augment_pair(image, mask):
    """Apply augmentation to before/after pair and mask"""
    # Random horizontal flip
    if np.random.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    
    # Random vertical flip
    if np.random.random() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    
    # Random 90-degree rotation
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    mask = np.rot90(mask, k)
    
    return image, mask


def data_generator(X, y, batch_size=8, augment=True):
    """Data generator with augmentation"""
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    while True:
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            batch_X = []
            batch_y = []
            
            for idx in batch_indices:
                img = X[idx].copy()
                msk = y[idx].copy()
                
                if augment:
                    img, msk = augment_pair(img, msk)
                
                batch_X.append(img)
                batch_y.append(msk)
            
            yield np.array(batch_X), np.array(batch_y)


def train_model():
    """Train the change detection model"""
    logger.info("="*80)
    logger.info("CHANGE DETECTION MODEL TRAINING")
    logger.info("="*80)
    
    # Load data
    data_dir = Path('./data/raw/deforestation_competition')
    X, y = load_before_after_pairs(data_dir, num_samples=300)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"\nTraining set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    
    # Build model
    logger.info("\nBuilding Change Detection U-Net...")
    model = build_change_detection_unet(input_shape=(256, 256, 6))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=combined_loss,
        metrics=[dice_coef, 'accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_dice_coef',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'outputs/models/change_detection_best.h5',
            monitor='val_dice_coef',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Create data generators
    batch_size = 8
    train_gen = data_generator(X_train, y_train, batch_size=batch_size, augment=True)
    val_gen = data_generator(X_val, y_val, batch_size=batch_size, augment=False)
    
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    # Train
    logger.info("\nStarting training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Save final model
    model_path = Path('outputs/models/change_detection_model.h5')
    model.save(model_path)
    logger.info(f"\n✅ Model saved to: {model_path}")
    
    # Save training history
    history_path = Path('outputs/models/change_detection_history.npz')
    np.savez(
        history_path,
        loss=history.history['loss'],
        val_loss=history.history['val_loss'],
        dice_coef=history.history['dice_coef'],
        val_dice_coef=history.history['val_dice_coef']
    )
    
    # Print final metrics
    final_dice = max(history.history['val_dice_coef'])
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Best Validation Dice Score: {final_dice:.4f}")
    logger.info(f"Model ready for change detection!")
    
    return model, history


if __name__ == "__main__":
    train_model()
