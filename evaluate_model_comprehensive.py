"""
Comprehensive Model Evaluation with Multiple Metrics
Generates detailed performance analysis of trained deforestation detection model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import cv2
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

from data.data_loader import DeforestationDataLoader, COMPETITION_DATASET_PATH
from models.deforestation_model import DeforestationDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveModelEvaluator:
    """Comprehensive evaluation with multiple metrics"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.results = {}
        self.output_dir = Path('./outputs/evaluation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self):
        """Load trained model"""
        logger.info(f"\n📂 Loading model from: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        detector = DeforestationDetector(input_shape=(256, 256, 3), model_type='unet')
        detector.load_model(str(self.model_path))
        self.model = detector.model
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"   Model parameters: {self.model.count_params():,}")
        
    def prepare_test_data(self, num_samples=30):
        """Load and prepare test data"""
        logger.info(f"\n📊 Preparing test data ({num_samples} samples)...")
        
        loader = DeforestationDataLoader(str(Path.cwd() / 'data'))
        data = loader.load_kaggle_dataset(str(COMPETITION_DATASET_PATH), 'competition')
        
        train_paths = data.get('train_paths', [])
        logger.info(f"Found {len(train_paths)} total samples")
        
        # Use different samples than training
        test_paths = train_paths[-num_samples:] if len(train_paths) > num_samples else train_paths
        
        X_test = []
        y_test = []
        
        logger.info("Loading test samples...")
        for i, path in enumerate(test_paths):
            if i % 10 == 0:
                logger.info(f"  Loading {i+1}/{len(test_paths)}...")
            
            try:
                # Load .npy file
                data_array = np.load(str(path))
                
                # Handle 2D or 3D data
                if len(data_array.shape) == 2:
                    img = cv2.cvtColor(data_array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                elif len(data_array.shape) == 3:
                    img = data_array[:, :, :3]
                else:
                    continue
                
                # Normalize to 0-255
                if img.max() > 255:
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                
                # Resize
                img = cv2.resize(img, (256, 256))
                
                # Normalize to [0, 1]
                img_norm = img.astype(np.float32) / 255.0
                X_test.append(img_norm)
                
                # Create ground truth mask (NDVI-based)
                if len(data_array.shape) == 3 and data_array.shape[-1] >= 4:
                    nir = data_array[:, :, 3].astype(float)
                    red = data_array[:, :, 2].astype(float)
                    ndvi = (nir - red) / (nir + red + 1e-8)
                    mask = (ndvi < 0.3).astype(np.float32)
                    mask = cv2.resize(mask, (256, 256))
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    mask = (gray < 100).astype(np.float32)
                
                y_test.append(mask)
                
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue
        
        self.X_test = np.array(X_test)
        self.y_test = np.expand_dims(np.array(y_test), axis=-1)
        
        logger.info(f"✅ Test data prepared:")
        logger.info(f"   X_test shape: {self.X_test.shape}")
        logger.info(f"   y_test shape: {self.y_test.shape}")
        
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
        
        # Basic classification metrics
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
            
            # Specificity and Sensitivity
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            
            # False Positive Rate and False Negative Rate
            metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics['fnr'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        
        # IoU (Intersection over Union) - Jaccard Index
        intersection = np.logical_and(y_true_binary, y_pred_binary).sum()
        union = np.logical_or(y_true_binary, y_pred_binary).sum()
        metrics['iou'] = float(intersection / union) if union > 0 else 0.0
        
        # Dice Coefficient (F1 for segmentation)
        dice = 2 * intersection / (y_true_binary.sum() + y_pred_binary.sum())
        metrics['dice_coefficient'] = float(dice) if not np.isnan(dice) else 0.0
        
        # Pixel Accuracy
        metrics['pixel_accuracy'] = float(np.mean(y_true_binary == y_pred_binary))
        
        # Mean IoU per sample
        sample_ious = []
        for i in range(len(y_true)):
            y_t = y_true_binary[i].flatten()
            y_p = y_pred_binary[i].flatten()
            inter = np.logical_and(y_t, y_p).sum()
            un = np.logical_or(y_t, y_p).sum()
            sample_ious.append(inter / un if un > 0 else 0.0)
        metrics['mean_iou_per_sample'] = float(np.mean(sample_ious))
        
        # ROC AUC
        try:
            fpr_arr, tpr_arr, thresholds = roc_curve(y_true_flat, y_pred_prob)
            metrics['roc_auc'] = float(auc(fpr_arr, tpr_arr))
            metrics['roc_fpr'] = fpr_arr.tolist()[:100]  # Limit size
            metrics['roc_tpr'] = tpr_arr.tolist()[:100]
        except:
            metrics['roc_auc'] = 0.0
            metrics['roc_fpr'] = []
            metrics['roc_tpr'] = []
        
        return metrics
    
    def evaluate(self):
        """Run complete evaluation"""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        # Make predictions
        logger.info("\n🔮 Generating predictions...")
        predictions = self.model.predict(self.X_test, batch_size=4, verbose=1)
        
        # Calculate metrics
        logger.info("\n📊 Calculating comprehensive metrics...")
        self.results = self.calculate_metrics(self.y_test, predictions)
        
        # Print results
        self.print_results()
        
        # Create visualizations
        self.create_visualizations(predictions)
        
        # Save results
        self.save_results()
        
    def print_results(self):
        """Print comprehensive results"""
        print("\n" + "="*80)
        print("📊 EVALUATION METRICS")
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
        print(f"   Dice Coefficient:   {self.results['dice_coefficient']:.4f}")
        print(f"   Pixel Accuracy:     {self.results['pixel_accuracy']:.4f} ({self.results['pixel_accuracy']*100:.2f}%)")
        
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
        
        # 1. Sample Predictions Grid
        fig = plt.figure(figsize=(20, 10))
        n_samples = min(6, len(self.X_test))
        
        for i in range(n_samples):
            # Original image
            plt.subplot(4, n_samples, i + 1)
            plt.imshow(self.X_test[i])
            plt.title(f'Sample {i+1}', fontsize=10)
            plt.axis('off')
            
            # Ground truth
            plt.subplot(4, n_samples, n_samples + i + 1)
            plt.imshow(self.y_test[i].squeeze(), cmap='RdYlGn_r', vmin=0, vmax=1)
            plt.title('Ground Truth', fontsize=10)
            plt.axis('off')
            
            # Prediction
            plt.subplot(4, n_samples, 2*n_samples + i + 1)
            plt.imshow(predictions[i].squeeze(), cmap='RdYlGn_r', vmin=0, vmax=1)
            plt.title('Prediction', fontsize=10)
            plt.axis('off')
            
            # Error map
            plt.subplot(4, n_samples, 3*n_samples + i + 1)
            error = np.abs(self.y_test[i].squeeze() - predictions[i].squeeze())
            plt.imshow(error, cmap='hot', vmin=0, vmax=1)
            plt.title('Error Map', fontsize=10)
            plt.axis('off')
        
        plt.tight_layout()
        samples_path = self.output_dir / 'sample_predictions.png'
        plt.savefig(samples_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Sample predictions saved to: {samples_path}")
        
        # 2. Confusion Matrix Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = np.array([[self.results['true_negatives'], self.results['false_positives']],
                       [self.results['false_negatives'], self.results['true_positives']]])
        
        sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                   xticklabels=['No Deforestation', 'Deforestation'],
                   yticklabels=['No Deforestation', 'Deforestation'])
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        
        cm_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Confusion matrix saved to: {cm_path}")
        
        # 3. ROC Curve
        if self.results['roc_auc'] > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(self.results['roc_fpr'], self.results['roc_tpr'], 
                   linewidth=2, label=f'Model (AUC = {self.results["roc_auc"]:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')
            ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ax.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            roc_path = self.output_dir / 'roc_curve.png'
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✅ ROC curve saved to: {roc_path}")
        
        # 4. Metrics Summary Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
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
        bars = ax.barh(metrics_names, metrics_values, color=colors)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, metrics_values)):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.1)
        ax.grid(True, axis='x', alpha=0.3)
        
        metrics_path = self.output_dir / 'metrics_summary.png'
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Metrics summary saved to: {metrics_path}")
        
    def save_results(self):
        """Save comprehensive results to JSON"""
        results_json = {
            'model_path': str(self.model_path),
            'evaluation_date': datetime.now().isoformat(),
            'test_samples': int(len(self.X_test)),
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
                    'dice_coefficient': self.results['dice_coefficient'],
                    'pixel_accuracy': self.results['pixel_accuracy'],
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
            }
        }
        
        json_path = self.output_dir / 'comprehensive_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=4)
        
        logger.info(f"\n✅ Results saved to: {json_path}")


def main():
    """Main evaluation function"""
    
    model_path = './outputs/models/competition_unet_model.h5'
    
    print("\n" + "="*80)
    print("🌲 DEFORESTATION MODEL - COMPREHENSIVE EVALUATION")
    print("="*80)
    
    if not Path(model_path).exists():
        print(f"\n❌ Model not found at: {model_path}")
        print("Please train the model first using: python train_model.py")
        return
    
    try:
        # Create evaluator
        evaluator = ComprehensiveModelEvaluator(model_path)
        
        # Load model
        evaluator.load_model()
        
        # Prepare test data
        evaluator.prepare_test_data(num_samples=30)
        
        # Run comprehensive evaluation
        evaluator.evaluate()
        
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE EVALUATION COMPLETE!")
        print("="*80)
        print("\n📁 Output Files:")
        print("   📊 Metrics JSON:        outputs/evaluation/comprehensive_metrics.json")
        print("   📸 Sample Predictions:  outputs/evaluation/sample_predictions.png")
        print("   📉 Confusion Matrix:    outputs/evaluation/confusion_matrix.png")
        print("   📈 ROC Curve:           outputs/evaluation/roc_curve.png")
        print("   📊 Metrics Summary:     outputs/evaluation/metrics_summary.png")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
