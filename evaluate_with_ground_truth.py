"""
Model Evaluation with ACTUAL Ground Truth Labels
Uses real labeled masks from Brazil Competition dataset
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import cv2
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, jaccard_score
)

from models.deforestation_model import DeforestationDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroundTruthEvaluator:
    """Evaluation with real ground truth masks from competition dataset"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.results = {}
        self.output_dir = Path('./outputs/evaluation_ground_truth')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Competition dataset paths
        self.data_dir = Path('./data/raw/deforestation_competition')
        self.train_json = self.data_dir / 'train.json'
        
    def load_model(self):
        """Load trained model"""
        logger.info(f"\n📂 Loading model from: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        # Auto-detect model type from filename
        model_type = 'mobilenet_unet' if 'mobilenet' in str(self.model_path).lower() else 'unet'
        logger.info(f"Detected model type: {model_type}")
        
        detector = DeforestationDetector(input_shape=(256, 256, 3), model_type=model_type)
        detector.load_model(str(self.model_path))
        self.model = detector.model
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"   Model parameters: {self.model.count_params():,}")
        
    def load_ground_truth_data(self, num_samples=30):
        """Load data with real ground truth masks"""
        logger.info(f"\n📊 Loading ground truth data ({num_samples} samples)...")
        
        # Read train.json
        if not self.train_json.exists():
            raise FileNotFoundError(f"train.json not found at: {self.train_json}")
        
        with open(self.train_json, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Found {len(metadata)} total samples with ground truth")
        
        # Use last samples for testing (different from training)
        sample_indices = list(metadata.keys())[-num_samples:]
        
        X_before = []
        X_after = []
        y_true_masks = []
        sample_info = []
        
        logger.info("Loading samples with ground truth masks...")
        loaded_count = 0
        
        for idx in sample_indices:
            entry = metadata[idx]
            
            try:
                # Get file paths
                before_file = self.data_dir / 'train' / 'public' / entry['files']['satellite_img_first']
                after_file = self.data_dir / 'train' / 'public' / entry['files']['satellite_img_second']
                mask_file = self.data_dir / 'train' / 'public' / entry['files']['mask']
                
                # Check if files exist
                if not all([before_file.exists(), after_file.exists(), mask_file.exists()]):
                    logger.warning(f"Missing files for sample {idx}")
                    continue
                
                # Load images
                before_img = np.load(str(before_file))
                after_img = np.load(str(after_file))
                mask = np.load(str(mask_file))
                
                # Process images
                def process_image(img):
                    if len(img.shape) == 2:
                        # 2D grayscale
                        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    elif len(img.shape) == 3 and img.shape[-1] > 3:
                        # Multi-spectral, take first 3 channels
                        img = img[:, :, :3]
                    
                    # Normalize to 0-255
                    if img.max() > 255 or img.dtype != np.uint8:
                        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                    
                    # Resize to 256x256
                    img = cv2.resize(img, (256, 256))
                    
                    return img
                
                before_processed = process_image(before_img)
                after_processed = process_image(after_img)
                
                # Process mask
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]  # Take first channel if multi-channel
                
                # Resize mask
                mask_resized = cv2.resize(mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
                
                # Normalize mask to 0-1
                if mask_resized.max() > 1:
                    mask_resized = (mask_resized > 0).astype(np.float32)
                
                # Normalize images to [0, 1] for model
                before_norm = before_processed.astype(np.float32) / 255.0
                after_norm = after_processed.astype(np.float32) / 255.0
                
                X_before.append(before_norm)
                X_after.append(after_norm)
                y_true_masks.append(mask_resized)
                
                sample_info.append({
                    'index': idx,
                    'tile': entry['tile'],
                    'date_first': entry['date_first'],
                    'date_second': entry['date_second'],
                    'mask_file': entry['files']['mask']
                })
                
                loaded_count += 1
                if loaded_count % 10 == 0:
                    logger.info(f"  Loaded {loaded_count}/{len(sample_indices)}...")
                
            except Exception as e:
                logger.warning(f"Failed to load sample {idx}: {e}")
                continue
        
        self.X_before = np.array(X_before)
        self.X_after = np.array(X_after)
        self.y_true = np.expand_dims(np.array(y_true_masks), axis=-1)
        self.sample_info = sample_info
        
        logger.info(f"\n✅ Ground truth data loaded:")
        logger.info(f"   Before images: {self.X_before.shape}")
        logger.info(f"   After images: {self.X_after.shape}")
        logger.info(f"   Ground truth masks: {self.y_true.shape}")
        logger.info(f"   Successfully loaded: {len(sample_info)} samples")
        
        # Calculate ground truth statistics
        deforested_pixels_per_sample = [mask.sum() for mask in y_true_masks]
        total_pixels = 256 * 256
        deforestation_pcts = [(dp / total_pixels) * 100 for dp in deforested_pixels_per_sample]
        
        logger.info(f"\n📊 Ground Truth Statistics:")
        logger.info(f"   Samples with deforestation: {sum(1 for pct in deforestation_pcts if pct > 0)}/{len(deforestation_pcts)}")
        logger.info(f"   Average deforestation: {np.mean(deforestation_pcts):.2f}%")
        logger.info(f"   Range: {np.min(deforestation_pcts):.2f}% - {np.max(deforestation_pcts):.2f}%")
        
    def calculate_metrics(self, y_true, y_pred, threshold=0.5):
        """Calculate comprehensive metrics"""
        
        # Binarize predictions
        y_pred_binary = (y_pred > threshold).astype(int)
        y_true_binary = (y_true > threshold).astype(int)
        
        # Flatten arrays
        y_true_flat = y_true_binary.flatten()
        y_pred_flat = y_pred_binary.flatten()
        y_pred_prob = y_pred.flatten()
        
        metrics = {}
        
        # Classification metrics
        metrics['accuracy'] = float(accuracy_score(y_true_flat, y_pred_flat))
        metrics['precision'] = float(precision_score(y_true_flat, y_pred_flat, zero_division=0))
        metrics['recall'] = float(recall_score(y_true_flat, y_pred_flat, zero_division=0))
        metrics['f1_score'] = float(f1_score(y_true_flat, y_pred_flat, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        
        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Additional metrics
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics['fnr'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        
        # Segmentation metrics
        intersection = np.logical_and(y_true_binary, y_pred_binary).sum()
        union = np.logical_or(y_true_binary, y_pred_binary).sum()
        metrics['iou'] = float(intersection / union) if union > 0 else 0.0
        
        dice = 2 * intersection / (y_true_binary.sum() + y_pred_binary.sum())
        metrics['dice_coefficient'] = float(dice) if not np.isnan(dice) else 0.0
        
        # Per-sample IoU
        sample_ious = []
        for i in range(len(y_true)):
            y_t = y_true_binary[i].flatten()
            y_p = y_pred_binary[i].flatten()
            inter = np.logical_and(y_t, y_p).sum()
            un = np.logical_or(y_t, y_p).sum()
            sample_ious.append(inter / un if un > 0 else 0.0)
        metrics['mean_iou_per_sample'] = float(np.mean(sample_ious))
        metrics['median_iou'] = float(np.median(sample_ious))
        
        # ROC AUC
        try:
            fpr_arr, tpr_arr, _ = roc_curve(y_true_flat, y_pred_prob)
            metrics['roc_auc'] = float(auc(fpr_arr, tpr_arr))
            metrics['roc_fpr'] = fpr_arr.tolist()[:100]
            metrics['roc_tpr'] = tpr_arr.tolist()[:100]
        except:
            metrics['roc_auc'] = 0.0
            metrics['roc_fpr'] = []
            metrics['roc_tpr'] = []
        
        return metrics
    
    def evaluate(self):
        """Run evaluation with ground truth"""
        print("\n" + "="*80)
        print("🎯 EVALUATION WITH REAL GROUND TRUTH LABELS")
        print("="*80)
        
        # Predict on after images (deforestation state)
        logger.info("\n🔮 Generating predictions on 'after' images...")
        predictions = self.model.predict(self.X_after, batch_size=4, verbose=1)
        
        # Calculate metrics
        logger.info("\n📊 Calculating metrics against ground truth...")
        self.results = self.calculate_metrics(self.y_true, predictions)
        
        # Print results
        self.print_results()
        
        # Create visualizations
        self.create_visualizations(predictions)
        
        # Save results
        self.save_results()
        
    def print_results(self):
        """Print evaluation results"""
        print("\n" + "="*80)
        print("📊 GROUND TRUTH EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n🎯 CLASSIFICATION METRICS:")
        print(f"   Accuracy:           {self.results['accuracy']:.4f} ({self.results['accuracy']*100:.2f}%)")
        print(f"   Precision:          {self.results['precision']:.4f}")
        print(f"   Recall (Sensitivity): {self.results['recall']:.4f}")
        print(f"   F1-Score:           {self.results['f1_score']:.4f}")
        print(f"   Specificity:        {self.results['specificity']:.4f}")
        
        print(f"\n📐 SEGMENTATION METRICS:")
        print(f"   IoU (Jaccard):      {self.results['iou']:.4f}")
        print(f"   Mean IoU/Sample:    {self.results['mean_iou_per_sample']:.4f}")
        print(f"   Median IoU:         {self.results['median_iou']:.4f}")
        print(f"   Dice Coefficient:   {self.results['dice_coefficient']:.4f}")
        
        print(f"\n📈 CONFUSION MATRIX:")
        print(f"   True Positives:     {self.results['true_positives']:,}")
        print(f"   True Negatives:     {self.results['true_negatives']:,}")
        print(f"   False Positives:    {self.results['false_positives']:,}")
        print(f"   False Negatives:    {self.results['false_negatives']:,}")
        
        print(f"\n⚠️ ERROR RATES:")
        print(f"   False Positive Rate: {self.results['fpr']:.4f} ({self.results['fpr']*100:.2f}%)")
        print(f"   False Negative Rate: {self.results['fnr']:.4f} ({self.results['fnr']*100:.2f}%)")
        
        print(f"\n🎭 ROC ANALYSIS:")
        print(f"   ROC-AUC Score:      {self.results['roc_auc']:.4f}")
        
        print("\n" + "="*80)
        
    def create_visualizations(self, predictions):
        """Create comprehensive visualizations"""
        logger.info("\n📊 Creating visualizations...")
        
        # 1. Sample predictions grid
        fig = plt.figure(figsize=(24, 12))
        n_samples = min(6, len(self.X_after))
        
        for i in range(n_samples):
            # Before image
            plt.subplot(5, n_samples, i + 1)
            plt.imshow(self.X_before[i])
            plt.title(f'Sample {i+1} - Before\n{self.sample_info[i]["date_first"]}', fontsize=9)
            plt.axis('off')
            
            # After image
            plt.subplot(5, n_samples, n_samples + i + 1)
            plt.imshow(self.X_after[i])
            plt.title(f'After\n{self.sample_info[i]["date_second"]}', fontsize=9)
            plt.axis('off')
            
            # Ground truth
            plt.subplot(5, n_samples, 2*n_samples + i + 1)
            plt.imshow(self.y_true[i].squeeze(), cmap='RdYlGn_r', vmin=0, vmax=1)
            gt_pct = (self.y_true[i].sum() / (256*256)) * 100
            plt.title(f'Ground Truth\n{gt_pct:.1f}% deforested', fontsize=9)
            plt.axis('off')
            
            # Prediction
            plt.subplot(5, n_samples, 3*n_samples + i + 1)
            plt.imshow(predictions[i].squeeze(), cmap='RdYlGn_r', vmin=0, vmax=1)
            pred_pct = (predictions[i].sum() / (256*256)) * 100
            plt.title(f'Prediction\n{pred_pct:.1f}% predicted', fontsize=9)
            plt.axis('off')
            
            # Error map
            plt.subplot(5, n_samples, 4*n_samples + i + 1)
            error = np.abs(self.y_true[i].squeeze() - predictions[i].squeeze())
            plt.imshow(error, cmap='hot', vmin=0, vmax=1)
            plt.title(f'Error Map\nMAE: {error.mean():.3f}', fontsize=9)
            plt.axis('off')
        
        plt.suptitle('Ground Truth Evaluation: Before/After Images with Real Labels', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        samples_path = self.output_dir / 'ground_truth_predictions.png'
        plt.savefig(samples_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Sample predictions saved to: {samples_path}")
        
        # 2. Confusion Matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = np.array([[self.results['true_negatives'], self.results['false_positives']],
                       [self.results['false_negatives'], self.results['true_positives']]])
        
        sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax, cbar_kws={'label': 'Pixel Count'},
                   xticklabels=['No Deforestation', 'Deforestation'],
                   yticklabels=['No Deforestation', 'Deforestation'])
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label (Ground Truth)', fontsize=14, fontweight='bold')
        ax.set_title('Confusion Matrix - Ground Truth Evaluation', fontsize=16, fontweight='bold', pad=20)
        
        cm_path = self.output_dir / 'confusion_matrix_ground_truth.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Confusion matrix saved to: {cm_path}")
        
        # 3. ROC Curve
        if self.results['roc_auc'] > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(self.results['roc_fpr'], self.results['roc_tpr'], 
                   linewidth=3, label=f'Model (AUC = {self.results["roc_auc"]:.3f})', color='#667eea')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')
            ax.fill_between(self.results['roc_fpr'], self.results['roc_tpr'], alpha=0.2, color='#667eea')
            ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
            ax.set_ylabel('True Positive Rate (Recall)', fontsize=14, fontweight='bold')
            ax.set_title('ROC Curve - Ground Truth Evaluation', fontsize=16, fontweight='bold', pad=20)
            ax.legend(fontsize=12, loc='lower right')
            ax.grid(True, alpha=0.3)
            
            roc_path = self.output_dir / 'roc_curve_ground_truth.png'
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✅ ROC curve saved to: {roc_path}")
        
        # 4. Metrics comparison
        fig, ax = plt.subplots(figsize=(12, 7))
        
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'IoU', 'Dice', 'ROC-AUC']
        metrics_values = [
            self.results['accuracy'],
            self.results['precision'],
            self.results['recall'],
            self.results['f1_score'],
            self.results['iou'],
            self.results['dice_coefficient'],
            self.results['roc_auc']
        ]
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#fee140']
        bars = ax.barh(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars, metrics_values)):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Model Performance Metrics - Ground Truth Evaluation', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.15)
        ax.grid(True, axis='x', alpha=0.3)
        
        metrics_path = self.output_dir / 'metrics_summary_ground_truth.png'
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Metrics summary saved to: {metrics_path}")
        
    def save_results(self):
        """Save comprehensive results"""
        results_json = {
            'model_path': str(self.model_path),
            'evaluation_type': 'GROUND_TRUTH',
            'evaluation_date': datetime.now().isoformat(),
            'test_samples': len(self.X_after),
            'model_parameters': int(self.model.count_params()),
            'metrics': {
                'classification': {
                    'accuracy': self.results['accuracy'],
                    'precision': self.results['precision'],
                    'recall': self.results['recall'],
                    'f1_score': self.results['f1_score'],
                    'specificity': self.results['specificity'],
                    'sensitivity': self.results['sensitivity'],
                },
                'segmentation': {
                    'iou': self.results['iou'],
                    'mean_iou_per_sample': self.results['mean_iou_per_sample'],
                    'median_iou': self.results['median_iou'],
                    'dice_coefficient': self.results['dice_coefficient'],
                },
                'confusion_matrix': {
                    'true_positives': self.results['true_positives'],
                    'true_negatives': self.results['true_negatives'],
                    'false_positives': self.results['false_positives'],
                    'false_negatives': self.results['false_negatives'],
                },
                'error_rates': {
                    'false_positive_rate': self.results['fpr'],
                    'false_negative_rate': self.results['fnr'],
                },
                'roc': {
                    'auc': self.results['roc_auc'],
                }
            },
            'sample_info': self.sample_info
        }
        
        json_path = self.output_dir / 'ground_truth_evaluation_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=4)
        
        logger.info(f"\n✅ Results saved to: {json_path}")


def main():
    """Main evaluation function"""
    
    model_path = './outputs/models/mobilenet_unet_model.h5'
    
    print("\n" + "="*80)
    print("🌲 MOBILENETV2 U-NET - GROUND TRUTH EVALUATION")
    print("="*80)
    print("\n✅ Using REAL labeled ground truth masks from competition dataset")
    
    if not Path(model_path).exists():
        print(f"\n❌ Model not found at: {model_path}")
        print("Please train the model first using: python train_with_ground_truth.py")
        return
    
    try:
        # Create evaluator
        evaluator = GroundTruthEvaluator(model_path)
        
        # Load model
        evaluator.load_model()
        
        # Load ground truth data
        evaluator.load_ground_truth_data(num_samples=30)
        
        # Run evaluation
        evaluator.evaluate()
        
        print("\n" + "="*80)
        print("✅ GROUND TRUTH EVALUATION COMPLETE!")
        print("="*80)
        print("\n📁 Output Files:")
        print("   📊 Metrics JSON:        outputs/evaluation_ground_truth/ground_truth_evaluation_metrics.json")
        print("   📸 Predictions:         outputs/evaluation_ground_truth/ground_truth_predictions.png")
        print("   📉 Confusion Matrix:    outputs/evaluation_ground_truth/confusion_matrix_ground_truth.png")
        print("   📈 ROC Curve:           outputs/evaluation_ground_truth/roc_curve_ground_truth.png")
        print("   📊 Metrics Summary:     outputs/evaluation_ground_truth/metrics_summary_ground_truth.png")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
