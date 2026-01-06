"""
Analyze MobileNetV2 predictions and find optimal threshold
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import cv2
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve
import logging

from models.deforestation_model import DeforestationDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(num_samples=30):
    """Load ground truth test data"""
    data_dir = Path('./data/raw/deforestation_competition')
    train_json = data_dir / 'train.json'
    
    with open(train_json, 'r') as f:
        metadata = json.load(f)
    
    # Use samples 300-330
    sample_indices = list(metadata.keys())[300:300+num_samples]
    
    X_test = []
    y_test = []
    
    for idx in sample_indices:
        entry = metadata[idx]
        
        try:
            after_file = data_dir / 'train' / 'public' / entry['files']['satellite_img_second']
            mask_file = data_dir / 'train' / 'public' / entry['files']['mask']
            
            if not all([after_file.exists(), mask_file.exists()]):
                continue
            
            after_img = np.load(str(after_file))
            mask = np.load(str(mask_file))
            
            # Process image
            if len(after_img.shape) == 2:
                after_img = np.stack([after_img]*3, axis=-1)
            elif after_img.shape[-1] > 3:
                after_img = after_img[:, :, :3]
            
            if after_img.shape[:2] != (256, 256):
                after_img = cv2.resize(after_img, (256, 256))
            
            after_processed = after_img.astype(np.float32) / 255.0
            
            # Process mask
            if len(mask.shape) == 2:
                mask = mask[:, :, np.newaxis]
            if mask.shape[:2] != (256, 256):
                mask = cv2.resize(mask, (256, 256))
                mask = mask[:, :, np.newaxis]
            
            mask_binary = (mask > 0).astype(np.float32)
            
            X_test.append(after_processed)
            y_test.append(mask_binary)
            
        except Exception as e:
            continue
    
    return np.array(X_test), np.array(y_test)


def analyze_predictions():
    """Analyze model predictions and find optimal threshold"""
    
    # Load model
    logger.info("Loading MobileNetV2 U-Net model...")
    detector = DeforestationDetector(
        input_shape=(256, 256, 3),
        model_type='mobilenet_unet'
    )
    detector.load_model('./outputs/models/mobilenet_unet_model.h5')
    
    # Load test data
    logger.info("Loading test data...")
    X_test, y_test = load_test_data(num_samples=30)
    logger.info(f"Loaded {len(X_test)} test samples")
    
    # Get predictions (continuous values 0-1)
    logger.info("Generating predictions...")
    y_pred_continuous = detector.predict(X_test)
    
    # Analyze prediction distribution
    logger.info("\n" + "="*80)
    logger.info("PREDICTION ANALYSIS")
    logger.info("="*80)
    
    pred_min = y_pred_continuous.min()
    pred_max = y_pred_continuous.max()
    pred_mean = y_pred_continuous.mean()
    pred_median = np.median(y_pred_continuous)
    
    logger.info(f"Prediction value range: [{pred_min:.6f}, {pred_max:.6f}]")
    logger.info(f"Prediction mean: {pred_mean:.6f}")
    logger.info(f"Prediction median: {pred_median:.6f}")
    
    # Check predictions on deforested pixels
    deforested_mask = y_test > 0
    if deforested_mask.sum() > 0:
        pred_on_deforested = y_pred_continuous[deforested_mask]
        logger.info(f"\nPredictions on TRUE deforested pixels:")
        logger.info(f"  Min: {pred_on_deforested.min():.6f}")
        logger.info(f"  Max: {pred_on_deforested.max():.6f}")
        logger.info(f"  Mean: {pred_on_deforested.mean():.6f}")
        logger.info(f"  Median: {np.median(pred_on_deforested):.6f}")
    
    # Find optimal threshold using F1-score
    logger.info("\n" + "="*80)
    logger.info("FINDING OPTIMAL THRESHOLD")
    logger.info("="*80)
    
    y_true_flat = y_test.flatten()
    y_pred_flat = y_pred_continuous.flatten()
    
    # Test different thresholds
    thresholds_to_test = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    best_f1 = 0
    best_threshold = 0.5
    
    results = []
    
    for threshold in thresholds_to_test:
        y_pred_binary = (y_pred_flat > threshold).astype(int)
        y_true_binary = (y_true_flat > 0).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
        
        logger.info(f"\nThreshold={threshold:.2f}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Acc={accuracy:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ BEST THRESHOLD: {best_threshold:.2f}")
    logger.info(f"   Best F1-Score: {best_f1:.4f}")
    logger.info("="*80)
    
    # Visualize predictions with different thresholds
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    sample_idx = 5  # Sample with deforestation
    
    for i, threshold in enumerate([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, best_threshold]):
        row = i // 4
        col = i % 4
        
        if i < 12:
            pred_binary = (y_pred_continuous[sample_idx, :, :, 0] > threshold).astype(float)
            axes[row, col].imshow(pred_binary, cmap='Reds', vmin=0, vmax=1)
            axes[row, col].set_title(f'Threshold={threshold:.2f}')
            axes[row, col].axis('off')
    
    # Show original and ground truth in last positions
    axes[3, 2].imshow(X_test[sample_idx])
    axes[3, 2].set_title('Original Image')
    axes[3, 2].axis('off')
    
    axes[3, 3].imshow(y_test[sample_idx, :, :, 0], cmap='Reds', vmin=0, vmax=1)
    axes[3, 3].set_title('Ground Truth')
    axes[3, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('./outputs/evaluation/threshold_analysis.png', dpi=150, bbox_inches='tight')
    logger.info(f"\n📊 Threshold analysis saved to: outputs/evaluation/threshold_analysis.png")
    plt.close()
    
    # Plot metrics vs threshold
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    axes[0, 0].plot(thresholds, precisions, 'b-o', label='Precision')
    axes[0, 0].axvline(best_threshold, color='r', linestyle='--', label=f'Best={best_threshold:.2f}')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision vs Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(thresholds, recalls, 'g-o', label='Recall')
    axes[0, 1].axvline(best_threshold, color='r', linestyle='--', label=f'Best={best_threshold:.2f}')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall vs Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(thresholds, f1s, 'r-o', label='F1-Score')
    axes[1, 0].axvline(best_threshold, color='r', linestyle='--', label=f'Best={best_threshold:.2f}')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title('F1-Score vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(thresholds, accuracies, 'm-o', label='Accuracy')
    axes[1, 1].axvline(best_threshold, color='r', linestyle='--', label=f'Best={best_threshold:.2f}')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('./outputs/evaluation/metrics_vs_threshold.png', dpi=150, bbox_inches='tight')
    logger.info(f"📊 Metrics vs threshold saved to: outputs/evaluation/metrics_vs_threshold.png")
    plt.close()
    
    # Save results
    results_data = {
        'best_threshold': float(best_threshold),
        'best_f1_score': float(best_f1),
        'prediction_stats': {
            'min': float(pred_min),
            'max': float(pred_max),
            'mean': float(pred_mean),
            'median': float(pred_median)
        },
        'threshold_results': results
    }
    
    with open('./outputs/evaluation/threshold_optimization.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"💾 Results saved to: outputs/evaluation/threshold_optimization.json")
    
    return best_threshold, results


if __name__ == "__main__":
    Path('./outputs/evaluation').mkdir(parents=True, exist_ok=True)
    
    best_threshold, results = analyze_predictions()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n🎯 Optimal Threshold: {best_threshold:.2f}")
    print("\n📁 Generated files:")
    print("   - outputs/evaluation/threshold_analysis.png")
    print("   - outputs/evaluation/metrics_vs_threshold.png")
    print("   - outputs/evaluation/threshold_optimization.json")
    print("\n💡 Update your model to use this threshold for better results!")
