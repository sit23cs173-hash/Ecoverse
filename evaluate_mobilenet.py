"""
EVALUATE MOBILENETV2 U-NET MODEL
Comprehensive evaluation with ground truth data
"""

import numpy as np
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc
)
import logging

from models.deforestation_model import DeforestationDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileNetEvaluator:
    """Evaluate MobileNetV2 U-Net model performance"""
    
    def __init__(self):
        self.data_dir = Path('./data/raw/deforestation_competition')
        self.train_json = self.data_dir / 'train.json'
        self.model_path = Path('./outputs/models/mobilenet_unet_model.h5')
        self.output_dir = Path('./outputs/evaluation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        
    def load_model(self):
        """Load trained MobileNetV2 U-Net model"""
        logger.info(f"Loading model from {self.model_path}")
        
        detector = DeforestationDetector(
            input_shape=(256, 256, 3),
            model_type='mobilenet_unet'
        )
        detector.load_model(str(self.model_path))
        self.model = detector
        
        logger.info("Model loaded successfully")
    
    def load_test_data(self, num_samples=100):
        """Load test data (samples not used in training)"""
        logger.info(f"Loading {num_samples} test samples...")
        
        with open(self.train_json, 'r') as f:
            metadata = json.load(f)
        
        # Use last N samples as test set (different from training samples 0-300)
        all_indices = list(metadata.keys())
        test_indices = all_indices[300:300+num_samples]
        
        X_test = []
        y_test = []
        
        for i, idx in enumerate(test_indices, 1):
            if i % 25 == 0:
                logger.info(f"  Loading {i}/{num_samples}...")
            
            entry = metadata[idx]
            
            try:
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
                
                X_test.append(after_processed)
                y_test.append(mask_binary)
                
            except Exception as e:
                logger.warning(f"Error loading sample {idx}: {e}")
                continue
        
        logger.info(f"Loaded {len(X_test)} test samples")
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        return X_test, y_test
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        logger.info("\n" + "="*80)
        logger.info("EVALUATING MOBILENETV2 U-NET MODEL")
        logger.info("="*80)
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = self.model.predict(X_test)
        
        # Threshold predictions
        y_pred_binary = (y_pred > 0.5).astype(np.uint8)
        
        # Flatten for metrics
        y_true_flat = y_test.reshape(-1)
        y_pred_flat = y_pred_binary.reshape(-1)
        y_pred_proba_flat = y_pred.reshape(-1)
        
        # Basic metrics
        logger.info("\n" + "="*80)
        logger.info("PIXEL-LEVEL METRICS")
        logger.info("="*80)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  True Negatives:  {tn:,}")
        logger.info(f"  False Positives: {fp:,}")
        logger.info(f"  False Negatives: {fn:,}")
        logger.info(f"  True Positives:  {tp:,}")
        
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)")
        logger.info(f"  Recall:      {recall:.4f} ({recall*100:.2f}%)")
        logger.info(f"  F1-Score:    {f1_score:.4f}")
        logger.info(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
        logger.info(f"  FP Rate:     {fpr:.4f} ({fpr*100:.2f}%)")
        
        # Sample-level metrics
        logger.info("\n" + "="*80)
        logger.info("SAMPLE-LEVEL METRICS")
        logger.info("="*80)
        
        n_samples = len(y_test)
        samples_with_deforestation_true = 0
        samples_with_deforestation_pred = 0
        correct_detections = 0
        
        for i in range(n_samples):
            has_deforestation_true = y_test[i].sum() > 0
            has_deforestation_pred = y_pred_binary[i].sum() > 0
            
            if has_deforestation_true:
                samples_with_deforestation_true += 1
            if has_deforestation_pred:
                samples_with_deforestation_pred += 1
            
            # Correct if both agree on presence/absence
            if has_deforestation_true == has_deforestation_pred:
                correct_detections += 1
        
        sample_accuracy = correct_detections / n_samples
        
        logger.info(f"\nSamples with deforestation:")
        logger.info(f"  Ground truth: {samples_with_deforestation_true}/{n_samples} ({samples_with_deforestation_true/n_samples*100:.1f}%)")
        logger.info(f"  Predicted:    {samples_with_deforestation_pred}/{n_samples} ({samples_with_deforestation_pred/n_samples*100:.1f}%)")
        logger.info(f"  Sample-level accuracy: {sample_accuracy:.4f} ({sample_accuracy*100:.2f}%)")
        
        # ROC curve
        fpr_curve, tpr_curve, _ = roc_curve(y_true_flat, y_pred_proba_flat)
        roc_auc = auc(fpr_curve, tpr_curve)
        logger.info(f"\nROC AUC Score: {roc_auc:.4f}")
        
        # Save results
        results = {
            'pixel_level': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'specificity': float(specificity),
                'false_positive_rate': float(fpr),
                'confusion_matrix': {
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp)
                }
            },
            'sample_level': {
                'accuracy': float(sample_accuracy),
                'samples_with_deforestation_true': int(samples_with_deforestation_true),
                'samples_with_deforestation_pred': int(samples_with_deforestation_pred),
                'total_samples': int(n_samples)
            },
            'roc_auc': float(roc_auc)
        }
        
        results_file = self.output_dir / 'mobilenet_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        # Visualize predictions
        self.visualize_predictions(X_test, y_test, y_pred_binary, num_samples=5)
        
        return results
    
    def visualize_predictions(self, X_test, y_test, y_pred, num_samples=5):
        """Visualize sample predictions"""
        logger.info("\nGenerating visualization...")
        
        # Select samples with deforestation
        samples_with_deforestation = []
        for i in range(len(y_test)):
            if y_test[i].sum() > 0:
                samples_with_deforestation.append(i)
        
        if len(samples_with_deforestation) < num_samples:
            sample_indices = samples_with_deforestation + list(range(num_samples - len(samples_with_deforestation)))
        else:
            sample_indices = samples_with_deforestation[:num_samples]
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        
        for row, idx in enumerate(sample_indices):
            # Original image
            axes[row, 0].imshow(X_test[idx])
            axes[row, 0].set_title(f'Sample {idx}: Input Image')
            axes[row, 0].axis('off')
            
            # Ground truth
            axes[row, 1].imshow(y_test[idx][:, :, 0], cmap='Reds', vmin=0, vmax=1)
            axes[row, 1].set_title('Ground Truth')
            axes[row, 1].axis('off')
            
            # Prediction
            axes[row, 2].imshow(y_pred[idx][:, :, 0], cmap='Reds', vmin=0, vmax=1)
            axes[row, 2].set_title('MobileNetV2 U-Net Prediction')
            axes[row, 2].axis('off')
        
        plt.tight_layout()
        
        viz_file = self.output_dir / 'mobilenet_predictions.png'
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to: {viz_file}")
        plt.close()
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        try:
            # Load model
            self.load_model()
            
            # Load test data
            X_test, y_test = self.load_test_data(num_samples=100)
            
            # Evaluate
            results = self.evaluate_model(X_test, y_test)
            
            logger.info("\n" + "="*80)
            logger.info("EVALUATION COMPLETE!")
            logger.info("="*80)
            logger.info(f"\n✅ MobileNetV2 U-Net evaluation finished")
            logger.info(f"📊 Key Results:")
            logger.info(f"   - Pixel Accuracy: {results['pixel_level']['accuracy']*100:.2f}%")
            logger.info(f"   - Precision: {results['pixel_level']['precision']*100:.2f}%")
            logger.info(f"   - Recall: {results['pixel_level']['recall']*100:.2f}%")
            logger.info(f"   - F1-Score: {results['pixel_level']['f1_score']:.4f}")
            logger.info(f"   - Sample Accuracy: {results['sample_level']['accuracy']*100:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    evaluator = MobileNetEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
