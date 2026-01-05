"""
Test script to verify all modules are properly installed and working.
Run this after installation to ensure everything is set up correctly.
"""

import sys
from pathlib import Path

print("="*70)
print("  DEFORESTATION DETECTION SYSTEM - INSTALLATION TEST")
print("="*70)
print()

# Track test results
all_tests_passed = True
failed_tests = []

# ============================================================================
# TEST 1: Python Version
# ============================================================================
print("[1/10] Checking Python version...")
if sys.version_info >= (3, 8):
    print("✅ Python version OK:", sys.version.split()[0])
else:
    print("❌ Python version too old. Need 3.8+")
    all_tests_passed = False
    failed_tests.append("Python version")
print()

# ============================================================================
# TEST 2: Core Dependencies
# ============================================================================
print("[2/10] Checking core dependencies...")
try:
    import numpy
    import pandas
    import cv2
    import matplotlib
    print("✅ Core dependencies installed")
    print(f"   numpy: {numpy.__version__}")
    print(f"   pandas: {pandas.__version__}")
    print(f"   opencv: {cv2.__version__}")
    print(f"   matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Missing core dependency: {e}")
    all_tests_passed = False
    failed_tests.append("Core dependencies")
print()

# ============================================================================
# TEST 3: Deep Learning Framework
# ============================================================================
print("[3/10] Checking TensorFlow/Keras...")
try:
    import tensorflow as tf
    print("✅ TensorFlow installed:", tf.__version__)
except ImportError:
    print("⚠️ TensorFlow not installed (optional for deep learning)")
print()

# ============================================================================
# TEST 4: Time-Series Libraries
# ============================================================================
print("[4/10] Checking time-series libraries...")
try:
    import statsmodels
    print("✅ Statsmodels installed:", statsmodels.__version__)
except ImportError:
    print("❌ Statsmodels not installed")
    all_tests_passed = False
    failed_tests.append("Statsmodels")
print()

# ============================================================================
# TEST 5: Dashboard Framework
# ============================================================================
print("[5/10] Checking Streamlit...")
try:
    import streamlit
    print("✅ Streamlit installed:", streamlit.__version__)
except ImportError:
    print("❌ Streamlit not installed")
    all_tests_passed = False
    failed_tests.append("Streamlit")
print()

# ============================================================================
# TEST 6: Custom Modules
# ============================================================================
print("[6/10] Testing custom modules import...")
try:
    from data.data_loader import DeforestationDataLoader
    from preprocessing.preprocessor import ImagePreprocessor
    from analysis.vegetation_analysis import VegetationIndexCalculator
    from analysis.carbon_impact import CarbonImpactCalculator
    from models.deforestation_model import DeforestationDetector
    from visualization.visualizer import DeforestationVisualizer
    print("✅ All custom modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import custom modules: {e}")
    all_tests_passed = False
    failed_tests.append("Custom modules")
print()

# ============================================================================
# TEST 7: Configuration
# ============================================================================
print("[7/10] Testing configuration...")
try:
    import config
    print("✅ Configuration loaded")
    print(f"   Data directory: {config.DATA_DIR}")
    print(f"   Image size: {config.IMG_HEIGHT}x{config.IMG_WIDTH}")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    all_tests_passed = False
    failed_tests.append("Configuration")
print()

# ============================================================================
# TEST 8: Directory Structure
# ============================================================================
print("[8/10] Checking directory structure...")
required_dirs = [
    'data', 'data/raw', 'data/processed',
    'preprocessing', 'models', 'analysis',
    'visualization', 'api', 'outputs', 'outputs/figures'
]
missing_dirs = []
for dir_name in required_dirs:
    if not Path(dir_name).exists():
        missing_dirs.append(dir_name)

if missing_dirs:
    print(f"⚠️ Missing directories: {', '.join(missing_dirs)}")
    print("   (These will be created automatically when needed)")
else:
    print("✅ All directories present")
print()

# ============================================================================
# TEST 9: Functional Test - NDVI Calculation
# ============================================================================
print("[9/10] Running functional test (NDVI calculation)...")
try:
    import numpy as np
    from analysis.vegetation_analysis import VegetationIndexCalculator
    
    # Create test image
    test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Calculate NDVI
    veg_calc = VegetationIndexCalculator()
    ndvi = veg_calc.calculate_ndvi(test_img)
    
    # Verify output
    assert ndvi.shape == (64, 64), "NDVI shape incorrect"
    assert -1 <= ndvi.min() <= ndvi.max() <= 1, "NDVI values out of range"
    
    print("✅ NDVI calculation working correctly")
    print(f"   Test NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
except Exception as e:
    print(f"❌ Functional test failed: {e}")
    all_tests_passed = False
    failed_tests.append("NDVI calculation")
print()

# ============================================================================
# TEST 10: Functional Test - Carbon Calculation
# ============================================================================
print("[10/10] Running functional test (carbon impact)...")
try:
    import numpy as np
    from analysis.carbon_impact import CarbonImpactCalculator
    
    # Create test mask
    test_mask = np.zeros((64, 64), dtype=np.uint8)
    test_mask[20:40, 20:40] = 1  # 400 pixels
    
    # Calculate impact
    carbon_calc = CarbonImpactCalculator(carbon_density=190.0, pixel_area_ha=0.01)
    impact = carbon_calc.calculate_carbon_from_mask(test_mask)
    
    # Verify output
    assert impact['deforested_area_ha'] > 0, "Area calculation failed"
    assert impact['co2_emissions_tons'] > 0, "CO2 calculation failed"
    
    print("✅ Carbon impact calculation working correctly")
    print(f"   Test area: {impact['deforested_area_ha']:.2f} ha")
    print(f"   Test CO2: {impact['co2_emissions_tons']:.2f} tons")
except Exception as e:
    print(f"❌ Functional test failed: {e}")
    all_tests_passed = False
    failed_tests.append("Carbon calculation")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*70)
if all_tests_passed:
    print("  ✅ ALL TESTS PASSED!")
    print("="*70)
    print()
    print("Your installation is complete and working correctly!")
    print()
    print("Next steps:")
    print("  1. Run demo pipeline: python main_pipeline.py")
    print("  2. Launch dashboard: streamlit run dashboard_app.py")
    print("  3. Try examples: python example_usage.py")
    print()
else:
    print("  ⚠️ SOME TESTS FAILED")
    print("="*70)
    print()
    print("Failed tests:", ", ".join(failed_tests))
    print()
    print("Please install missing dependencies:")
    print("  pip install -r requirements.txt")
    print()
    print("If problems persist, check:")
    print("  - Python version (need 3.8+)")
    print("  - Virtual environment activated")
    print("  - requirements.txt exists")
    print()

print("="*70)
