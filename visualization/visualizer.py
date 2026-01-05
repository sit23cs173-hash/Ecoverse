"""
STEP 8: VISUALIZATION MODULE
Comprehensive visualization functions for deforestation analysis.
Creates maps, charts, and dashboard-ready outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import cv2
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


class DeforestationVisualizer:
    """
    Create comprehensive visualizations for deforestation analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 150):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Dots per inch for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_image_comparison(self,
                             before_img: np.ndarray,
                             after_img: np.ndarray,
                             mask: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None):
        """
        Plot before/after image comparison with optional deforestation mask.
        
        Args:
            before_img: Before image [H, W, C]
            after_img: After image [H, W, C]
            mask: Optional deforestation mask [H, W]
            save_path: Path to save figure
        """
        n_plots = 3 if mask is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        
        # Before
        axes[0].imshow(before_img)
        axes[0].set_title('Before', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # After
        axes[1].imshow(after_img)
        axes[1].set_title('After', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Mask overlay
        if mask is not None:
            # Create overlay
            overlay = after_img.copy()
            red_mask = np.zeros_like(overlay)
            red_mask[:, :, 0] = mask * 255  # Red channel
            overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
            
            axes[2].imshow(overlay)
            axes[2].set_title('Deforestation Detected', fontsize=14, fontweight='bold')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved comparison to {save_path}")
        
        plt.show()
    
    def plot_deforestation_heatmap(self,
                                   mask: np.ndarray,
                                   title: str = "Deforestation Hotspot Map",
                                   save_path: Optional[str] = None):
        """
        Create heatmap showing deforestation intensity.
        
        Args:
            mask: Deforestation mask or intensity map [H, W]
            title: Plot title
            save_path: Path to save figure
        """
        plt.figure(figsize=self.figsize)
        
        # Create heatmap
        im = plt.imshow(mask, cmap='Reds', interpolation='bilinear')
        plt.colorbar(im, label='Deforestation Intensity')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")
        
        plt.show()
    
    def plot_ndvi_maps(self,
                      ndvi_before: np.ndarray,
                      ndvi_after: np.ndarray,
                      ndvi_change: np.ndarray,
                      save_path: Optional[str] = None):
        """
        Plot NDVI before, after, and change.
        
        Args:
            ndvi_before: NDVI before [H, W]
            ndvi_after: NDVI after [H, W]
            ndvi_change: NDVI change [H, W]
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Before
        im1 = axes[0].imshow(ndvi_before, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0].set_title('NDVI Before', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # After
        im2 = axes[1].imshow(ndvi_after, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[1].set_title('NDVI After', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # Change
        im3 = axes[2].imshow(ndvi_change, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2].set_title('NDVI Change (Loss in Red)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved NDVI maps to {save_path}")
        
        plt.show()
    
    def plot_deforestation_statistics(self,
                                     stats_dict: dict,
                                     save_path: Optional[str] = None):
        """
        Plot key deforestation statistics as a dashboard.
        
        Args:
            stats_dict: Dictionary with statistics
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Deforestation Analysis Dashboard', 
                    fontsize=16, fontweight='bold')
        
        # 1. Total deforested area
        ax1 = fig.add_subplot(gs[0, 0])
        area_ha = stats_dict.get('deforested_area_ha', 0)
        ax1.bar(['Area'], [area_ha], color='brown', alpha=0.7)
        ax1.set_ylabel('Hectares')
        ax1.set_title('Total Deforested Area')
        ax1.text(0, area_ha/2, f'{area_ha:.1f} ha', 
                ha='center', va='center', fontweight='bold', fontsize=12)
        
        # 2. CO2 emissions
        ax2 = fig.add_subplot(gs[0, 1])
        co2 = stats_dict.get('co2_emissions_tons', 0)
        ax2.bar(['CO2'], [co2], color='red', alpha=0.7)
        ax2.set_ylabel('Tons')
        ax2.set_title('CO2 Emissions')
        ax2.text(0, co2/2, f'{co2:.1f} tons', 
                ha='center', va='center', fontweight='bold', fontsize=12)
        
        # 3. Detection metrics
        ax3 = fig.add_subplot(gs[1, :])
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            stats_dict.get('accuracy', 0),
            stats_dict.get('precision', 0),
            stats_dict.get('recall', 0),
            stats_dict.get('f1_score', 0)
        ]
        bars = ax3.barh(metrics, values, color=['blue', 'green', 'orange', 'purple'], alpha=0.7)
        ax3.set_xlim([0, 1])
        ax3.set_xlabel('Score')
        ax3.set_title('Model Performance Metrics')
        ax3.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax3.text(val + 0.02, i, f'{val:.3f}', va='center', fontweight='bold')
        
        # 4. Forest cover change
        ax4 = fig.add_subplot(gs[2, 0])
        forest_before = stats_dict.get('forest_area_before', 1000)
        forest_after = stats_dict.get('forest_area_after', 800)
        ax4.bar(['Before', 'After'], [forest_before, forest_after], 
               color=['green', 'orange'], alpha=0.7)
        ax4.set_ylabel('Forest Area (ha)')
        ax4.set_title('Forest Cover Change')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Text summary
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        summary_text = f"""
        KEY FINDINGS:
        
        • Deforested: {area_ha:.1f} hectares
        • Carbon Loss: {stats_dict.get('carbon_loss_tons', 0):.1f} tons
        • CO2 Emissions: {co2:.1f} tons
        • Model Accuracy: {stats_dict.get('accuracy', 0):.1%}
        • Equivalent: {stats_dict.get('equivalent_cars_year', 0):.0f} cars/year
        """
        ax5.text(0.1, 0.5, summary_text, fontsize=10, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved dashboard to {save_path}")
        
        plt.show()
    
    def plot_training_history(self,
                             history: dict,
                             save_path: Optional[str] = None):
        """
        Plot model training history.
        
        Args:
            history: Training history dictionary
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved training history to {save_path}")
        
        plt.show()
    
    def create_detection_overlay(self,
                                background_img: np.ndarray,
                                mask: np.ndarray,
                                alpha: float = 0.4,
                                color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """
        Create overlay of deforestation mask on background image.
        
        Args:
            background_img: Background image [H, W, C]
            mask: Deforestation mask [H, W]
            alpha: Transparency of overlay
            color: RGB color for overlay
            
        Returns:
            Overlay image
        """
        overlay = background_img.copy()
        
        # Create colored mask
        colored_mask = np.zeros_like(overlay)
        for i in range(3):
            colored_mask[:, :, i] = mask * color[i]
        
        # Blend
        result = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
        
        return result
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix for binary classification.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            save_path: Path to save figure
        """
        from sklearn.metrics import confusion_matrix
        
        # Flatten arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Deforestation', 'Deforestation'],
                   yticklabels=['No Deforestation', 'Deforestation'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.show()


def create_multi_panel_figure(images_dict: dict,
                              titles_dict: dict,
                              save_path: Optional[str] = None):
    """
    Create multi-panel figure from dictionary of images.
    
    Args:
        images_dict: Dictionary mapping keys to images
        titles_dict: Dictionary mapping keys to titles
        save_path: Path to save figure
    """
    n_images = len(images_dict)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (key, img) in enumerate(images_dict.items()):
        axes[idx].imshow(img)
        axes[idx].set_title(titles_dict.get(key, key), fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved multi-panel figure to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization Module")
    print("=" * 50)
    
    # Create dummy data
    before = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    after = before.copy()
    after[50:150, 50:150] = [150, 100, 70]  # Simulate deforestation
    
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:150, 50:150] = 1
    
    # Initialize visualizer
    viz = DeforestationVisualizer()
    
    # Test visualizations
    print("Creating image comparison...")
    viz.plot_image_comparison(before, after, mask)
    
    print("Creating heatmap...")
    viz.plot_deforestation_heatmap(mask)
    
    print("Visualizations created successfully!")
