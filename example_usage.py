"""
Example script demonstrating how to use the deforestation detection system.
This script shows various usage patterns and workflows.
"""

import numpy as np
import cv2
from pathlib import Path

# Import modules
from data.data_loader import DeforestationDataLoader
from preprocessing.preprocessor import ImagePreprocessor
from analysis.vegetation_analysis import VegetationIndexCalculator, visualize_ndvi_comparison
from analysis.carbon_impact import CarbonImpactCalculator, CarbonImpactVisualizer
from models.deforestation_model import NDVIBasedDetector
from visualization.visualizer import DeforestationVisualizer

print("="*70)
print("  DEFORESTATION DETECTION - EXAMPLE USAGE")
print("="*70)
print()

# ============================================================================
# EXAMPLE 1: Basic NDVI Calculation
# ============================================================================
print("EXAMPLE 1: Calculate NDVI from Satellite Image")
print("-" * 70)

# Create synthetic satellite image (replace with real data)
satellite_img = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
satellite_img[:, :, 1] = np.random.randint(100, 200, (256, 256))  # More green

# Calculate NDVI
veg_calc = VegetationIndexCalculator()
ndvi = veg_calc.calculate_ndvi(satellite_img)

print(f"✓ NDVI calculated successfully")
print(f"  Mean NDVI: {ndvi.mean():.3f}")
print(f"  NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
print(f"  Forest pixels (NDVI > 0.4): {(ndvi > 0.4).sum()} ({(ndvi > 0.4).sum()/ndvi.size*100:.1f}%)")
print()

# ============================================================================
# EXAMPLE 2: Deforestation Detection
# ============================================================================
print("EXAMPLE 2: Detect Deforestation from Before/After Images")
print("-" * 70)

# Create before image (forest)
before_img = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
before_img[:, :, 1] = np.random.randint(120, 200, (256, 256))

# Create after image (with deforestation)
after_img = before_img.copy()
after_img[80:180, 80:180] = [140, 90, 60]  # Simulate deforested area

# Calculate NDVI
ndvi_before = veg_calc.calculate_ndvi(before_img)
ndvi_after = veg_calc.calculate_ndvi(after_img)

# Detect deforestation
detector = NDVIBasedDetector(ndvi_threshold=-0.15)
deforestation_mask = detector.detect_deforestation(ndvi_before, ndvi_after)

print(f"✓ Deforestation detected")
print(f"  Deforested pixels: {deforestation_mask.sum()}")
print(f"  Deforested area: {deforestation_mask.sum()} pixels")
print()

# ============================================================================
# EXAMPLE 3: Carbon Impact Assessment
# ============================================================================
print("EXAMPLE 3: Estimate Carbon Impact")
print("-" * 70)

# Initialize carbon calculator
carbon_calc = CarbonImpactCalculator(
    carbon_density=190.0,  # Amazon rainforest
    pixel_area_ha=0.01     # 10m resolution
)

# Calculate impact
impact = carbon_calc.calculate_carbon_from_mask(deforestation_mask)

print(f"✓ Carbon impact calculated")
print(f"  Deforested area: {impact['deforested_area_ha']:.2f} hectares")
print(f"  Carbon loss: {impact['carbon_loss_tons']:.2f} tons C")
print(f"  CO2 emissions: {impact['co2_emissions_tons']:.2f} tons CO2")
print(f"  Equivalent to: {impact['equivalent_cars_year']:.0f} cars per year")
print()

# ============================================================================
# EXAMPLE 4: Visualization
# ============================================================================
print("EXAMPLE 4: Create Visualizations")
print("-" * 70)

# Initialize visualizer
viz = DeforestationVisualizer()

# Create output directory
output_dir = Path('./outputs/examples')
output_dir.mkdir(parents=True, exist_ok=True)

# Visualize before/after comparison
viz.plot_image_comparison(
    before_img=before_img,
    after_img=after_img,
    mask=deforestation_mask,
    save_path=output_dir / 'example_comparison.png'
)
print(f"✓ Saved: example_comparison.png")

# Visualize NDVI maps
viz.plot_ndvi_maps(
    ndvi_before=ndvi_before,
    ndvi_after=ndvi_after,
    ndvi_change=ndvi_after - ndvi_before,
    save_path=output_dir / 'example_ndvi.png'
)
print(f"✓ Saved: example_ndvi.png")

# Visualize carbon impact
visualizer_carbon = CarbonImpactVisualizer()
visualizer_carbon.plot_carbon_impact_summary(
    impact_dict=impact,
    save_path=output_dir / 'example_carbon.png'
)
print(f"✓ Saved: example_carbon.png")

print()

# ============================================================================
# EXAMPLE 5: Batch Processing
# ============================================================================
print("EXAMPLE 5: Process Multiple Image Pairs")
print("-" * 70)

# Generate multiple image pairs
num_samples = 10
total_carbon = 0

for i in range(num_samples):
    # Generate synthetic pair
    before = np.random.randint(50, 150, (128, 128, 3), dtype=np.uint8)
    before[:, :, 1] = np.random.randint(120, 200, (128, 128))
    
    after = before.copy()
    # Random deforestation patch
    x, y = np.random.randint(0, 80), np.random.randint(0, 80)
    w, h = np.random.randint(20, 40), np.random.randint(20, 40)
    after[y:y+h, x:x+w] = [140, 90, 60]
    
    # Detect and calculate
    ndvi_b = veg_calc.calculate_ndvi(before)
    ndvi_a = veg_calc.calculate_ndvi(after)
    mask = detector.detect_deforestation(ndvi_b, ndvi_a)
    impact_i = carbon_calc.calculate_carbon_from_mask(mask)
    
    total_carbon += impact_i['co2_emissions_tons']

print(f"✓ Processed {num_samples} image pairs")
print(f"  Total CO2 emissions: {total_carbon:.2f} tons")
print(f"  Average per sample: {total_carbon/num_samples:.2f} tons")
print()

# ============================================================================
# EXAMPLE 6: Custom Workflow
# ============================================================================
print("EXAMPLE 6: Custom Detection Workflow")
print("-" * 70)

# Load data
loader = DeforestationDataLoader(data_dir='./data/raw')

# Create small demo dataset
demo_path = Path('./data/raw/example_demo')
loader.create_dummy_dataset(demo_path, num_samples=5)

# Load images
before_imgs, after_imgs, masks = loader.load_image_pairs(
    before_dir=demo_path / 'before',
    after_dir=demo_path / 'after',
    mask_dir=demo_path / 'masks'
)

print(f"✓ Loaded {len(before_imgs)} image pairs")

# Preprocess
preprocessor = ImagePreprocessor(target_size=(256, 256))
before_resized = preprocessor.resize_images(before_imgs)
after_resized = preprocessor.resize_images(after_imgs)
before_norm = preprocessor.normalize_images(before_resized, method='0-1')

print(f"✓ Images preprocessed")
print(f"  Shape: {before_norm.shape}")
print(f"  Range: [{before_norm.min():.3f}, {before_norm.max():.3f}]")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*70)
print("  EXAMPLES COMPLETED SUCCESSFULLY!")
print("="*70)
print()
print("Key Takeaways:")
print("  1. NDVI is calculated from satellite images to measure vegetation")
print("  2. Deforestation is detected by comparing NDVI before and after")
print("  3. Carbon impact is estimated from deforested area")
print("  4. Visualizations help interpret and communicate results")
print("  5. The system supports batch processing for large-scale analysis")
print()
print("Next Steps:")
print("  • Run main_pipeline.py for complete end-to-end analysis")
print("  • Launch dashboard_app.py for interactive exploration")
print("  • Use individual modules for custom workflows")
print()
print(f"Outputs saved to: {output_dir}")
print()
