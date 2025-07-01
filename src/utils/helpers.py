import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Tuple
import logging
from pathlib import Path
import json
import pickle


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name.
        level: Logging level.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger


def validate_dataframe(df: pd.DataFrame, 
                      required_columns: Optional[List[str]] = None,
                      min_rows: int = 1) -> bool:
    """
    Validate a DataFrame meets basic requirements.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
        min_rows: Minimum number of rows required.
        
    Returns:
        True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        return False
    
    if len(df) < min_rows:
        return False
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            return False
    
    return True


def clean_returns_data(returns: pd.DataFrame, 
                      max_return: float = 1.0,
                      min_return: float = -1.0) -> pd.DataFrame:
    """
    Clean returns data by removing outliers and invalid values.
    
    Args:
        returns: DataFrame with return data.
        max_return: Maximum allowed return (e.g., 100% = 1.0).
        min_return: Minimum allowed return (e.g., -100% = -1.0).
        
    Returns:
        Cleaned DataFrame.
    """
    cleaned = returns.copy()
    
    # Replace infinite values with NaN
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    
    # Cap extreme returns
    cleaned = cleaned.clip(lower=min_return, upper=max_return)
    
    return cleaned


def calculate_portfolio_metrics(returns: pd.Series, 
                              risk_free_rate: float = 0.02,
                              trading_days: int = 252) -> Dict[str, float]:
    """
    Calculate standard portfolio performance metrics.
    
    Args:
        returns: Series with portfolio returns.
        risk_free_rate: Annual risk-free rate.
        trading_days: Number of trading days per year.
        
    Returns:
        Dictionary with performance metrics.
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return {"error": "No valid returns data"}
    
    # Basic metrics
    total_return = (1 + returns_clean).prod() - 1
    annualized_return = returns_clean.mean() * trading_days
    volatility = returns_clean.std() * np.sqrt(trading_days)
    
    # Risk-adjusted metrics
    excess_returns = returns_clean - (risk_free_rate / trading_days)
    sharpe_ratio = (excess_returns.mean() / returns_clean.std()) * np.sqrt(trading_days) if returns_clean.std() > 0 else 0
    
    # Drawdown calculation
    cumulative = (1 + returns_clean).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Additional metrics
    win_rate = (returns_clean > 0).mean()
    avg_win = returns_clean[returns_clean > 0].mean() if (returns_clean > 0).any() else 0
    avg_loss = returns_clean[returns_clean < 0].mean() if (returns_clean < 0).any() else 0
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "skewness": returns_clean.skew(),
        "kurtosis": returns_clean.kurtosis()
    }


def resample_data(data: pd.DataFrame, 
                 frequency: str = "1D",
                 method: str = "last") -> pd.DataFrame:
    """
    Resample time series data to a different frequency.
    
    Args:
        data: DataFrame with datetime index.
        frequency: Target frequency (e.g., "1D", "1W", "1M").
        method: Resampling method ("last", "first", "mean", "sum").
        
    Returns:
        Resampled DataFrame.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a datetime index")
    
    resampler = data.resample(frequency)
    
    if method == "last":
        return resampler.last()
    elif method == "first":
        return resampler.first()
    elif method == "mean":
        return resampler.mean()
    elif method == "sum":
        return resampler.sum()
    else:
        raise ValueError(f"Unknown resampling method: {method}")


def align_dataframes(*dataframes: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames to have the same index.
    
    Args:
        dataframes: Variable number of DataFrames to align.
        
    Returns:
        List of aligned DataFrames.
    """
    if len(dataframes) < 2:
        return list(dataframes)
    
    # Find common index
    common_index = dataframes[0].index
    for df in dataframes[1:]:
        common_index = common_index.intersection(df.index)
    
    # Align all DataFrames
    aligned = [df.loc[common_index] for df in dataframes]
    
    return aligned


def save_results(results: Dict, 
                file_path: Union[str, Path],
                format: str = "json") -> None:
    """
    Save results to file in specified format.
    
    Args:
        results: Dictionary with results to save.
        file_path: Path to save file.
        format: File format ("json", "pickle").
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        # Convert numpy types to native Python types for JSON serialization
        json_results = convert_numpy_types(results)
        with open(file_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
    elif format == "pickle":
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_results(file_path: Union[str, Path],
                format: str = "json") -> Dict:
    """
    Load results from file.
    
    Args:
        file_path: Path to file.
        format: File format ("json", "pickle").
        
    Returns:
        Loaded results dictionary.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if format == "json":
        with open(file_path, 'r') as f:
            return json.load(f)
    elif format == "pickle":
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types.
    
    Args:
        obj: Object to convert.
        
    Returns:
        Object with numpy types converted.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


def create_date_range(start_date: str, 
                     end_date: str,
                     frequency: str = "D") -> pd.DatetimeIndex:
    """
    Create a date range between two dates.
    
    Args:
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        frequency: Frequency string ("D", "B", "W", "M").
        
    Returns:
        DatetimeIndex with the specified range.
    """
    return pd.date_range(start=start_date, end=end_date, freq=frequency)


def calculate_correlation_matrix(data: pd.DataFrame,
                               method: str = "pearson",
                               min_periods: int = 30) -> pd.DataFrame:
    """
    Calculate correlation matrix with minimum period requirement.
    
    Args:
        data: DataFrame with time series data.
        method: Correlation method ("pearson", "spearman", "kendall").
        min_periods: Minimum number of observations required.
        
    Returns:
        Correlation matrix DataFrame.
    """
    return data.corr(method=method, min_periods=min_periods)


def winsorize_data(data: pd.DataFrame, 
                  lower_percentile: float = 0.01,
                  upper_percentile: float = 0.99) -> pd.DataFrame:
    """
    Winsorize data by capping extreme values at specified percentiles.
    
    Args:
        data: DataFrame to winsorize.
        lower_percentile: Lower percentile for capping (e.g., 0.01 for 1%).
        upper_percentile: Upper percentile for capping (e.g., 0.99 for 99%).
        
    Returns:
        Winsorized DataFrame.
    """
    winsorized = data.copy()
    
    for column in data.select_dtypes(include=[np.number]).columns:
        lower_bound = data[column].quantile(lower_percentile)
        upper_bound = data[column].quantile(upper_percentile)
        winsorized[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    
    return winsorized


def format_performance_summary(metrics: Dict[str, float]) -> str:
    """
    Format performance metrics into a readable summary string.
    
    Args:
        metrics: Dictionary with performance metrics.
        
    Returns:
        Formatted summary string.
    """
    summary = "Performance Summary:\n"
    summary += "-" * 30 + "\n"
    
    if "total_return" in metrics:
        summary += f"Total Return: {metrics['total_return']:.2%}\n"
    if "annualized_return" in metrics:
        summary += f"Annualized Return: {metrics['annualized_return']:.2%}\n"
    if "volatility" in metrics:
        summary += f"Volatility: {metrics['volatility']:.2%}\n"
    if "sharpe_ratio" in metrics:
        summary += f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n"
    if "max_drawdown" in metrics:
        summary += f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
    if "win_rate" in metrics:
        summary += f"Win Rate: {metrics['win_rate']:.2%}\n"
    
    return summary
