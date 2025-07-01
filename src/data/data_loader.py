import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging
from pathlib import Path

from ..config import DataConfig, get_config


class DataLoader:
    """
    Handles loading and preprocessing of asset price data.
    
    This class provides methods to load CSV data, handle missing values,
    resample to daily frequency, and calculate log returns.
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize the DataLoader.
        
        Args:
            config: Data configuration object. If None, uses default config.
        """
        self.config = config or get_config().data
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.log_returns: Optional[pd.DataFrame] = None
        
    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Args:
            file_path: Path to the CSV file. If None, uses config path.
            
        Returns:
            Raw DataFrame with datetime index.
            
        Raises:
            FileNotFoundError: If the data file doesn't exist.
            ValueError: If the data format is invalid.
        """
        file_path = file_path or self.config.data_file
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            # Load data with datetime parsing
            df = pd.read_csv(
                file_path,
                parse_dates=True,
                index_col=0
            )
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Filter by date range if specified
            if self.config.start_date:
                df = df[df.index >= self.config.start_date]
            if self.config.end_date:
                df = df[df.index <= self.config.end_date]
            
            self.raw_data = df
            self.logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} assets")
            self.logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading data from {file_path}: {str(e)}")
    
    def preprocess_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Preprocess the raw data.
        
        This includes resampling to daily frequency and forward-filling
        missing values.
        
        Args:
            df: DataFrame to preprocess. If None, uses loaded raw data.
            
        Returns:
            Preprocessed DataFrame.
        """
        if df is None:
            if self.raw_data is None:
                raise ValueError("No data loaded. Call load_raw_data() first.")
            df = self.raw_data.copy()
        
        # Resample to daily frequency and take last value
        df_daily = df.resample(self.config.resample_frequency).last()
        
        # Forward fill missing values
        if self.config.fill_method == "ffill":
            df_daily = df_daily.ffill()
        elif self.config.fill_method == "bfill":
            df_daily = df_daily.bfill()
        
        self.processed_data = df_daily
        self.logger.info(f"Preprocessed data: {df_daily.shape}")
        
        return df_daily
    
    def calculate_log_returns(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate log returns from price data.
        
        Args:
            df: DataFrame with price data. If None, uses processed data.
            
        Returns:
            DataFrame with log returns.
        """
        if df is None:
            if self.processed_data is None:
                raise ValueError("No processed data available. Call preprocess_data() first.")
            df = self.processed_data
        
        # Calculate log returns
        log_returns = df.apply(np.log).diff()
        
        self.log_returns = log_returns
        self.logger.info(f"Calculated log returns: {log_returns.shape}")
        
        return log_returns
    
    def get_asset_availability(self, df: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Get the number of available assets over time.
        
        Args:
            df: DataFrame to analyze. If None, uses processed data.
            
        Returns:
            Series with number of available assets per date.
        """
        if df is None:
            if self.processed_data is None:
                raise ValueError("No processed data available.")
            df = self.processed_data
        
        return df.notnull().sum(axis=1)
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the loaded data.
        
        Returns:
            Dictionary with data summary information.
        """
        if self.processed_data is None:
            return {"error": "No data loaded"}
        
        df = self.processed_data
        
        summary = {
            "total_assets": df.shape[1],
            "total_observations": df.shape[0],
            "date_range": {
                "start": df.index.min().strftime("%Y-%m-%d"),
                "end": df.index.max().strftime("%Y-%m-%d")
            },
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "missing_percentage": (df.isnull().sum().sum() / df.size) * 100
            },
            "assets_with_data": (df.notnull().any()).sum(),
            "avg_assets_per_day": df.notnull().sum(axis=1).mean()
        }
        
        return summary
    
    def load_and_preprocess(self, file_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete data loading and preprocessing pipeline.
        
        Args:
            file_path: Path to the CSV file. If None, uses config path.
            
        Returns:
            Tuple of (processed_data, log_returns).
        """
        # Load raw data
        self.load_raw_data(file_path)
        
        # Preprocess data
        processed_data = self.preprocess_data()
        
        # Calculate log returns
        log_returns = self.calculate_log_returns()
        
        return processed_data, log_returns
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the loaded data for basic requirements.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            True if data is valid, False otherwise.
        """
        try:
            # Check if DataFrame is not empty
            if df.empty:
                self.logger.error("Data is empty")
                return False
            
            # Check if index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.error("Index is not datetime")
                return False
            
            # Check if there's at least some numeric data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                self.logger.error("No numeric columns found")
                return False
            
            # Check for reasonable data range (prices should be positive)
            if (df[numeric_cols] <= 0).any().any():
                self.logger.warning("Found non-positive values in price data")
            
            self.logger.info("Data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False
