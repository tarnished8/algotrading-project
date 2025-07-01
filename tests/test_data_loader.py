import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.data_loader import DataLoader
from src.config import DataConfig


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        data = pd.DataFrame({
            "asset1": np.random.randn(len(dates)).cumsum() + 100,
            "asset2": np.random.randn(len(dates)).cumsum() + 50,
            "asset3": np.random.randn(len(dates)).cumsum() + 200
        }, index=dates)
        
        # Add some missing values
        data.iloc[10:15, 0] = np.nan
        data.iloc[50:55, 1] = np.nan
        
        return data
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name)
            return f.name
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.config is not None
        assert loader.raw_data is None
        assert loader.processed_data is None
        assert loader.log_returns is None
    
    def test_init_with_config(self):
        """Test DataLoader initialization with custom config."""
        config = DataConfig(data_file="test.csv")
        loader = DataLoader(config)
        assert loader.config.data_file == "test.csv"
    
    def test_load_raw_data(self, temp_csv_file):
        """Test loading raw data from CSV."""
        loader = DataLoader()
        data = loader.load_raw_data(temp_csv_file)
        
        assert isinstance(data, pd.DataFrame)
        assert isinstance(data.index, pd.DatetimeIndex)
        assert len(data.columns) == 3
        assert loader.raw_data is not None
    
    def test_load_raw_data_file_not_found(self):
        """Test loading data from non-existent file."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_raw_data("non_existent_file.csv")
    
    def test_preprocess_data(self, temp_csv_file):
        """Test data preprocessing."""
        loader = DataLoader()
        loader.load_raw_data(temp_csv_file)
        processed = loader.preprocess_data()
        
        assert isinstance(processed, pd.DataFrame)
        assert loader.processed_data is not None
        # Check that forward fill was applied
        assert processed.isnull().sum().sum() < loader.raw_data.isnull().sum().sum()
    
    def test_calculate_log_returns(self, temp_csv_file):
        """Test log returns calculation."""
        loader = DataLoader()
        loader.load_raw_data(temp_csv_file)
        loader.preprocess_data()
        returns = loader.calculate_log_returns()
        
        assert isinstance(returns, pd.DataFrame)
        assert loader.log_returns is not None
        assert returns.shape == loader.processed_data.shape
        # First row should be NaN due to diff()
        assert returns.iloc[0].isnull().all()
    
    def test_get_asset_availability(self, temp_csv_file):
        """Test asset availability calculation."""
        loader = DataLoader()
        loader.load_raw_data(temp_csv_file)
        loader.preprocess_data()
        availability = loader.get_asset_availability()
        
        assert isinstance(availability, pd.Series)
        assert len(availability) == len(loader.processed_data)
        assert availability.max() <= loader.processed_data.shape[1]
    
    def test_get_data_summary(self, temp_csv_file):
        """Test data summary generation."""
        loader = DataLoader()
        loader.load_raw_data(temp_csv_file)
        loader.preprocess_data()
        summary = loader.get_data_summary()
        
        assert isinstance(summary, dict)
        assert "total_assets" in summary
        assert "total_observations" in summary
        assert "date_range" in summary
        assert "missing_data" in summary
    
    def test_load_and_preprocess(self, temp_csv_file):
        """Test complete loading and preprocessing pipeline."""
        loader = DataLoader()
        processed_data, log_returns = loader.load_and_preprocess(temp_csv_file)
        
        assert isinstance(processed_data, pd.DataFrame)
        assert isinstance(log_returns, pd.DataFrame)
        assert processed_data.shape == log_returns.shape
        assert loader.raw_data is not None
        assert loader.processed_data is not None
        assert loader.log_returns is not None
    
    def test_validate_data(self, sample_data):
        """Test data validation."""
        loader = DataLoader()
        
        # Valid data
        assert loader.validate_data(sample_data) == True
        
        # Empty data
        empty_df = pd.DataFrame()
        assert loader.validate_data(empty_df) == False
        
        # Non-datetime index
        invalid_df = sample_data.reset_index()
        assert loader.validate_data(invalid_df) == False
    
    def test_date_filtering(self, temp_csv_file):
        """Test date range filtering."""
        config = DataConfig(
            data_file=temp_csv_file,
            start_date="2020-06-01",
            end_date="2020-08-31"
        )
        loader = DataLoader(config)
        data = loader.load_raw_data()
        
        assert data.index.min() >= pd.Timestamp("2020-06-01")
        assert data.index.max() <= pd.Timestamp("2020-08-31")
    
    def teardown_method(self, method):
        """Clean up after each test."""
        # Remove temporary files if they exist
        for temp_file in Path.cwd().glob("*.csv"):
            if temp_file.name.startswith("tmp"):
                temp_file.unlink(missing_ok=True)
