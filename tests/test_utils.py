import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.helpers import (
    validate_dataframe,
    clean_returns_data,
    calculate_portfolio_metrics,
    resample_data,
    align_dataframes,
    save_results,
    load_results,
    convert_numpy_types,
    create_date_range,
    calculate_correlation_matrix,
    winsorize_data,
    format_performance_summary
)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
        return pd.DataFrame({
            "col1": range(len(dates)),
            "col2": np.random.randn(len(dates)),
            "col3": np.random.randn(len(dates)) * 2
        }, index=dates)
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns series for testing."""
        return pd.Series([0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.012])
    
    def test_validate_dataframe_valid(self, sample_dataframe):
        """Test DataFrame validation with valid data."""
        assert validate_dataframe(sample_dataframe) == True
        assert validate_dataframe(sample_dataframe, required_columns=["col1", "col2"]) == True
        assert validate_dataframe(sample_dataframe, min_rows=5) == True
    
    def test_validate_dataframe_invalid(self, sample_dataframe):
        """Test DataFrame validation with invalid data."""
        # Empty DataFrame
        assert validate_dataframe(pd.DataFrame()) == False
        assert validate_dataframe(None) == False
        
        # Missing required columns
        assert validate_dataframe(sample_dataframe, required_columns=["missing_col"]) == False
        
        # Too few rows
        assert validate_dataframe(sample_dataframe, min_rows=20) == False
    
    def test_clean_returns_data(self):
        """Test returns data cleaning."""
        # Create data with outliers and infinite values
        dirty_data = pd.DataFrame({
            "asset1": [0.01, np.inf, -0.005, 2.0, 0.02],
            "asset2": [-np.inf, 0.008, -1.5, 0.015, np.nan]
        })
        
        cleaned = clean_returns_data(dirty_data, max_return=0.5, min_return=-0.5)
        
        # Check that infinite values are replaced
        assert not np.isinf(cleaned.values).any()
        
        # Check that extreme values are capped
        assert cleaned.max().max() <= 0.5
        assert cleaned.min().min() >= -0.5
    
    def test_calculate_portfolio_metrics(self, sample_returns):
        """Test portfolio metrics calculation."""
        metrics = calculate_portfolio_metrics(sample_returns)
        
        assert isinstance(metrics, dict)
        assert "total_return" in metrics
        assert "annualized_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        
        # Check that metrics are reasonable
        assert isinstance(metrics["total_return"], (int, float))
        assert isinstance(metrics["sharpe_ratio"], (int, float))
        assert 0 <= metrics["win_rate"] <= 1
        assert metrics["max_drawdown"] <= 0
    
    def test_calculate_portfolio_metrics_empty(self):
        """Test portfolio metrics with empty data."""
        empty_returns = pd.Series([])
        metrics = calculate_portfolio_metrics(empty_returns)
        
        assert "error" in metrics
    
    def test_resample_data(self, sample_dataframe):
        """Test data resampling."""
        # Test different resampling methods
        resampled_last = resample_data(sample_dataframe, frequency="2D", method="last")
        resampled_mean = resample_data(sample_dataframe, frequency="2D", method="mean")
        
        assert len(resampled_last) < len(sample_dataframe)
        assert len(resampled_mean) < len(sample_dataframe)
        assert isinstance(resampled_last.index, pd.DatetimeIndex)
        assert isinstance(resampled_mean.index, pd.DatetimeIndex)
    
    def test_resample_data_invalid_index(self):
        """Test resampling with invalid index."""
        df_no_datetime = pd.DataFrame({"col1": [1, 2, 3]})
        
        with pytest.raises(ValueError):
            resample_data(df_no_datetime)
    
    def test_align_dataframes(self):
        """Test DataFrame alignment."""
        # Create DataFrames with different indices
        dates1 = pd.date_range("2020-01-01", "2020-01-10", freq="D")
        dates2 = pd.date_range("2020-01-05", "2020-01-15", freq="D")
        
        df1 = pd.DataFrame({"col1": range(len(dates1))}, index=dates1)
        df2 = pd.DataFrame({"col2": range(len(dates2))}, index=dates2)
        
        aligned = align_dataframes(df1, df2)
        
        assert len(aligned) == 2
        assert len(aligned[0]) == len(aligned[1])
        assert aligned[0].index.equals(aligned[1].index)
    
    def test_save_and_load_results(self):
        """Test saving and loading results."""
        test_results = {
            "metric1": 0.123,
            "metric2": np.float64(0.456),
            "metric3": np.int32(789),
            "nested": {"value": np.array([1, 2, 3])}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test JSON format
            save_results(test_results, temp_path, format="json")
            loaded_results = load_results(temp_path, format="json")
            
            assert "metric1" in loaded_results
            assert "metric2" in loaded_results
            assert "nested" in loaded_results
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_convert_numpy_types(self):
        """Test numpy type conversion."""
        test_obj = {
            "int": np.int32(42),
            "float": np.float64(3.14),
            "array": np.array([1, 2, 3]),
            "list": [np.int64(1), np.float32(2.5)],
            "nested": {"value": np.nan}
        }
        
        converted = convert_numpy_types(test_obj)
        
        assert isinstance(converted["int"], int)
        assert isinstance(converted["float"], float)
        assert isinstance(converted["array"], list)
        assert isinstance(converted["list"][0], int)
        assert isinstance(converted["list"][1], float)
        assert converted["nested"]["value"] is None  # np.nan -> None
    
    def test_create_date_range(self):
        """Test date range creation."""
        date_range = create_date_range("2020-01-01", "2020-01-10", frequency="D")
        
        assert isinstance(date_range, pd.DatetimeIndex)
        assert len(date_range) == 10
        assert date_range[0] == pd.Timestamp("2020-01-01")
        assert date_range[-1] == pd.Timestamp("2020-01-10")
    
    def test_calculate_correlation_matrix(self, sample_dataframe):
        """Test correlation matrix calculation."""
        corr_matrix = calculate_correlation_matrix(sample_dataframe)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert corr_matrix.shape[0] == len(sample_dataframe.columns)
        
        # Diagonal should be 1 (or NaN if insufficient data)
        diagonal = np.diag(corr_matrix.values)
        valid_diagonal = diagonal[~np.isnan(diagonal)]
        assert np.allclose(valid_diagonal, 1.0)
    
    def test_winsorize_data(self):
        """Test data winsorization."""
        # Create data with extreme values
        data = pd.DataFrame({
            "col1": [1, 2, 3, 4, 100],  # 100 is an outlier
            "col2": [-50, 2, 3, 4, 5]   # -50 is an outlier
        })
        
        winsorized = winsorize_data(data, lower_percentile=0.2, upper_percentile=0.8)
        
        # Extreme values should be capped
        assert winsorized["col1"].max() < 100
        assert winsorized["col2"].min() > -50
    
    def test_format_performance_summary(self):
        """Test performance summary formatting."""
        metrics = {
            "total_return": 0.15,
            "annualized_return": 0.12,
            "volatility": 0.18,
            "sharpe_ratio": 0.67,
            "max_drawdown": -0.08,
            "win_rate": 0.55
        }
        
        summary = format_performance_summary(metrics)
        
        assert isinstance(summary, str)
        assert "Total Return: 15.00%" in summary
        assert "Sharpe Ratio: 0.670" in summary
        assert "Max Drawdown: -8.00%" in summary
        assert "Win Rate: 55.00%" in summary
