"""
Quick script to download the time-series Brazil dataset after accepting terms on Kaggle.
"""
import kagglehub
import os

def download_timeseries():
    print("🌳 Downloading Time-Series Brazil Deforestation Dataset...")
    print("Note: You must have accepted the dataset terms on Kaggle first!\n")
    
    try:
        # Create directory
        os.makedirs('./data/raw/timeseries_brazil', exist_ok=True)
        
        # Download dataset
        path = kagglehub.dataset_download('gallo33henrique/time-series-arima-sarima-deforestation-brazil')
        
        print(f"\n✅ SUCCESS! Dataset downloaded to: {path}")
        print(f"\n📊 Dataset contains historical deforestation time-series data for Brazil")
        print(f"🔍 Check the downloaded files and integrate them into the dashboard\n")
        
        # List downloaded files
        if os.path.exists(path):
            print("📁 Downloaded files:")
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                print(f"   - {file} ({size:.2f} MB)")
        
        return path
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n💡 Solution:")
        print("1. Visit: https://www.kaggle.com/datasets/gallo33henrique/time-series-arima-sarima-deforestation-brazil")
        print("2. Click 'Download' to accept the dataset terms")
        print("3. Wait a few seconds for permissions to update")
        print("4. Run this script again")
        return None

if __name__ == "__main__":
    download_timeseries()
