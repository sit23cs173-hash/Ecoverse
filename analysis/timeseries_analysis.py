"""
STEP 6: TIME-SERIES ANALYSIS MODULE
Analyzes deforestation trends over time using ARIMA/SARIMA models.
Provides forecasting and trend visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import logging
from typing import Tuple, Optional

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeforestationTimeSeriesAnalyzer:
    """
    Analyzes time-series deforestation data and provides forecasts.
    """
    
    def __init__(self, data: pd.DataFrame, date_column: str, value_column: str):
        """
        Initialize time-series analyzer.
        
        Args:
            data: DataFrame containing time-series data
            date_column: Name of date column
            value_column: Name of value column (deforestation area/rate)
        """
        self.data = data.copy()
        self.date_column = date_column
        self.value_column = value_column
        
        # Ensure date column is datetime
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        self.data = self.data.sort_values(date_column)
        self.data.set_index(date_column, inplace=True)
        
        self.model = None
        self.forecast = None
        
        logger.info(f"Initialized time-series analyzer with {len(self.data)} data points")
    
    def explore_data(self) -> dict:
        """
        Perform exploratory data analysis on time-series.
        
        Returns:
            Dictionary with summary statistics
        """
        logger.info("Exploring time-series data")
        
        stats = {
            'start_date': self.data.index.min(),
            'end_date': self.data.index.max(),
            'total_points': len(self.data),
            'mean': self.data[self.value_column].mean(),
            'median': self.data[self.value_column].median(),
            'std': self.data[self.value_column].std(),
            'min': self.data[self.value_column].min(),
            'max': self.data[self.value_column].max(),
            'total_deforestation': self.data[self.value_column].sum()
        }
        
        return stats
    
    def decompose_series(self, period: int = 12, save_path: Optional[str] = None):
        """
        Decompose time-series into trend, seasonal, and residual components.
        
        Args:
            period: Seasonal period (12 for monthly data with yearly seasonality)
            save_path: Optional path to save plot
        """
        logger.info(f"Decomposing time-series with period={period}")
        
        # Perform decomposition
        decomposition = seasonal_decompose(
            self.data[self.value_column],
            model='additive',
            period=period
        )
        
        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Original
        decomposition.observed.plot(ax=axes[0], color='blue')
        axes[0].set_ylabel('Observed')
        axes[0].set_title('Time Series Decomposition', fontsize=14, fontweight='bold')
        
        # Trend
        decomposition.trend.plot(ax=axes[1], color='green')
        axes[1].set_ylabel('Trend')
        
        # Seasonal
        decomposition.seasonal.plot(ax=axes[2], color='orange')
        axes[2].set_ylabel('Seasonal')
        
        # Residual
        decomposition.resid.plot(ax=axes[3], color='red')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved decomposition plot to {save_path}")
        
        plt.show()
        
        return decomposition
    
    def plot_acf_pacf(self, lags: int = 40, save_path: Optional[str] = None):
        """
        Plot Autocorrelation and Partial Autocorrelation functions.
        Helps determine ARIMA parameters.
        
        Args:
            lags: Number of lags to plot
            save_path: Optional path to save plot
        """
        logger.info("Plotting ACF and PACF")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # ACF
        plot_acf(self.data[self.value_column].dropna(), lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
        
        # PACF
        plot_pacf(self.data[self.value_column].dropna(), lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def fit_arima(self, 
                  order: Tuple[int, int, int] = (1, 1, 1),
                  train_size: float = 0.8) -> dict:
        """
        Fit ARIMA model to time-series data.
        
        Args:
            order: ARIMA order (p, d, q)
            train_size: Proportion of data for training
            
        Returns:
            Dictionary with model results
        """
        logger.info(f"Fitting ARIMA{order} model")
        
        # Split data
        train_len = int(len(self.data) * train_size)
        train = self.data[self.value_column][:train_len]
        test = self.data[self.value_column][train_len:]
        
        # Fit model
        model = ARIMA(train, order=order)
        self.model = model.fit()
        
        logger.info(f"ARIMA model fitted with AIC: {self.model.aic:.2f}")
        
        # Forecast on test set
        forecast = self.model.forecast(steps=len(test))
        
        # Calculate errors
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mape = np.mean(np.abs((test - forecast) / test)) * 100
        
        results = {
            'model': self.model,
            'order': order,
            'aic': self.model.aic,
            'bic': self.model.bic,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'train_size': len(train),
            'test_size': len(test)
        }
        
        logger.info(f"Test MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return results
    
    def fit_sarima(self,
                   order: Tuple[int, int, int] = (1, 1, 1),
                   seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
                   train_size: float = 0.8) -> dict:
        """
        Fit SARIMA model to time-series data with seasonal components.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            train_size: Proportion of data for training
            
        Returns:
            Dictionary with model results
        """
        logger.info(f"Fitting SARIMA{order}x{seasonal_order} model")
        
        # Split data
        train_len = int(len(self.data) * train_size)
        train = self.data[self.value_column][:train_len]
        test = self.data[self.value_column][train_len:]
        
        # Fit model
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        self.model = model.fit(disp=False)
        
        logger.info(f"SARIMA model fitted with AIC: {self.model.aic:.2f}")
        
        # Forecast on test set
        forecast = self.model.forecast(steps=len(test))
        
        # Calculate errors
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mape = np.mean(np.abs((test - forecast) / test)) * 100
        
        results = {
            'model': self.model,
            'order': order,
            'seasonal_order': seasonal_order,
            'aic': self.model.aic,
            'bic': self.model.bic,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        logger.info(f"Test MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return results
    
    def forecast_future(self, steps: int = 24) -> pd.DataFrame:
        """
        Forecast future deforestation values.
        
        Args:
            steps: Number of steps ahead to forecast
            
        Returns:
            DataFrame with forecasted values and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_arima() or fit_sarima() first.")
        
        logger.info(f"Forecasting {steps} steps ahead")
        
        # Generate forecast
        forecast_result = self.model.get_forecast(steps=steps)
        forecast_values = forecast_result.predicted_mean
        confidence_intervals = forecast_result.conf_int()
        
        # Create forecast DataFrame
        last_date = self.data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq=pd.infer_freq(self.data.index)
        )
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_values.values,
            'lower_bound': confidence_intervals.iloc[:, 0].values,
            'upper_bound': confidence_intervals.iloc[:, 1].values
        })
        
        self.forecast = forecast_df
        
        return forecast_df
    
    def plot_forecast(self, 
                     forecast_steps: int = 24,
                     save_path: Optional[str] = None):
        """
        Plot historical data and forecast.
        
        Args:
            forecast_steps: Number of steps to forecast
            save_path: Optional path to save plot
        """
        if self.forecast is None:
            self.forecast_future(steps=forecast_steps)
        
        logger.info("Plotting forecast")
        
        plt.figure(figsize=(14, 6))
        
        # Plot historical data
        plt.plot(self.data.index, self.data[self.value_column], 
                label='Historical', color='blue', linewidth=2)
        
        # Plot forecast
        plt.plot(self.forecast['date'], self.forecast['forecast'],
                label='Forecast', color='red', linewidth=2, linestyle='--')
        
        # Plot confidence interval
        plt.fill_between(
            self.forecast['date'],
            self.forecast['lower_bound'],
            self.forecast['upper_bound'],
            color='red', alpha=0.2, label='95% Confidence Interval'
        )
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Deforestation Area', fontsize=12)
        plt.title('Deforestation Forecast', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved forecast plot to {save_path}")
        
        plt.show()
    
    def analyze_trends(self, save_path: Optional[str] = None):
        """
        Analyze and visualize deforestation trends.
        
        Args:
            save_path: Optional path to save plot
        """
        logger.info("Analyzing trends")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Time series plot
        axes[0, 0].plot(self.data.index, self.data[self.value_column], 
                       color='green', linewidth=2)
        axes[0, 0].set_title('Deforestation Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Deforestation Area')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Yearly aggregation
        yearly_data = self.data.resample('Y').sum()
        axes[0, 1].bar(yearly_data.index.year, yearly_data[self.value_column],
                      color='orange', alpha=0.7)
        axes[0, 1].set_title('Yearly Total Deforestation', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Total Deforestation')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Monthly pattern
        self.data['month'] = self.data.index.month
        monthly_avg = self.data.groupby('month')[self.value_column].mean()
        axes[1, 0].plot(monthly_avg.index, monthly_avg.values, 
                       marker='o', color='red', linewidth=2)
        axes[1, 0].set_title('Average Monthly Pattern', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Average Deforestation')
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Rolling average
        rolling_avg = self.data[self.value_column].rolling(window=12).mean()
        axes[1, 1].plot(self.data.index, self.data[self.value_column],
                       label='Original', alpha=0.5, color='blue')
        axes[1, 1].plot(rolling_avg.index, rolling_avg,
                       label='12-month Moving Average', color='red', linewidth=2)
        axes[1, 1].set_title('Trend with Moving Average', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Deforestation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved trend analysis to {save_path}")
        
        plt.show()


def create_dummy_timeseries(periods: int = 120) -> pd.DataFrame:
    """
    Create dummy time-series data for testing.
    
    Args:
        periods: Number of time periods
        
    Returns:
        DataFrame with synthetic deforestation data
    """
    dates = pd.date_range(start='2015-01-01', periods=periods, freq='M')
    
    # Generate synthetic data with trend and seasonality
    trend = np.linspace(100, 200, periods)
    seasonal = 30 * np.sin(np.arange(periods) * 2 * np.pi / 12)
    noise = np.random.normal(0, 10, periods)
    
    values = trend + seasonal + noise
    values = np.maximum(values, 0)  # Ensure non-negative
    
    df = pd.DataFrame({
        'date': dates,
        'deforestation_area': values
    })
    
    return df


if __name__ == "__main__":
    print("Time-Series Analysis Module")
    print("=" * 50)
    
    # Create dummy data
    df = create_dummy_timeseries(periods=120)
    print(f"Created dummy time-series with {len(df)} months")
    
    # Initialize analyzer
    analyzer = DeforestationTimeSeriesAnalyzer(
        data=df,
        date_column='date',
        value_column='deforestation_area'
    )
    
    # Explore data
    stats = analyzer.explore_data()
    print("\nTime-series statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Fit ARIMA
    results = analyzer.fit_arima(order=(1, 1, 1))
    print(f"\nARIMA Results - AIC: {results['aic']:.2f}, MAE: {results['mae']:.2f}")
    
    # Forecast
    forecast = analyzer.forecast_future(steps=12)
    print(f"\nForecast shape: {forecast.shape}")
    print(forecast.head())
