"""
DATASET DOWNLOAD SCRIPT
Simple script to download all real Kaggle datasets for the project.

Usage:
    python download_datasets.py
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('download_log.txt')
    ]
)

logger = logging.getLogger(__name__)

# Import the downloader
from data.real_data_downloader import RealDataDownloader, KAGGLE_DATASETS


def main():
    """Main download function."""
    
    print("\n" + "=" * 70)
    print(" " * 15 + "DEFORESTATION DETECTION PROJECT")
    print(" " * 20 + "Dataset Downloader")
    print("=" * 70)
    
    print("\n📦 This script will download the following datasets:")
    print("-" * 70)
    
    total_size = 0
    for key, info in KAGGLE_DATASETS.items():
        print(f"\n✓ {info['description']}")
        print(f"  Size: ~{info['size_gb']} GB")
        total_size += info['size_gb']
    
    print(f"\n📊 Total size: ~{total_size} GB")
    print("-" * 70)
    
    # Check Kaggle credentials first
    print("\n🔑 Checking Kaggle API credentials...")
    downloader = RealDataDownloader(data_root='./data')
    
    if not downloader.check_kaggle_credentials():
        print("\n❌ Kaggle credentials not found!")
        print("\nPlease set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. This downloads kaggle.json")
        print("5. Move kaggle.json to:")
        kaggle_dir = Path.home() / '.kaggle'
        print(f"   {kaggle_dir}")
        print("\nThen run this script again.")
        return
    
    # Prompt user to continue
    print("\n⚠️  WARNING: This will download several GB of data!")
    print("Make sure you have:")
    print("  - Sufficient disk space")
    print("  - Stable internet connection")
    print("  - Time (downloads may take 10-30 minutes)")
    
    response = input("\nDo you want to continue? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Download cancelled.")
        return
    
    # Start downloading
    print("\n" + "=" * 70)
    print("STARTING DOWNLOADS...")
    print("=" * 70)
    
    results = downloader.download_all_kaggle_datasets()
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"\n✅ Successfully downloaded: {successful}/{total} datasets")
    
    if successful > 0:
        print("\n📁 Data location:")
        print(f"   {Path('./data/raw').absolute()}")
        
        print("\n🎯 Next steps:")
        print("1. Run the dashboard:")
        print("   streamlit run dashboard_app.py")
        print("\n2. Or run the main pipeline:")
        print("   python main_pipeline.py --use-real-data")
        print("\n3. Check dataset info:")
        print("   python -c \"from data.real_data_downloader import RealDataDownloader; ")
        print("   d = RealDataDownloader(); print(d.get_dataset_info())\"")
    else:
        print("\n❌ No datasets were downloaded successfully.")
        print("Please check the errors above and try again.")
    
    print("\n" + "=" * 70)
    print("Log saved to: download_log.txt")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
