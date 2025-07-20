import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data file paths
    data_file: str = "data/assets.csv"
    
    # Data preprocessing parameters
    resample_frequency: str = "1d"  # Daily resampling
    fill_method: str = "ffill"  # Forward fill for missing values
    
    # Date range (None means use all available data)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class StrategyConfig:
    """Configuration for the trend following strategy."""
    
    # Lookback periods for momentum calculation
    min_lookback_days: int = 14  # 2 weeks
    max_lookback_days: int = 60  # ~2 months
    
    # Signal generation parameters
    signal_lag_days: int = 2  # 2-day lag for realistic trading
    
    # Position sizing
    max_position_size: float = 1.0  # Maximum position size per asset
    min_position_size: float = 0.01  # Minimum position size per asset
    
    # Portfolio constraints
    max_portfolio_leverage: float = 1.0  # No leverage
    
    # Risk management
    max_drawdown_threshold: float = 0.2  # 20% maximum drawdown
    volatility_lookback: int = 252  # Days for volatility calculation


@dataclass
class AnalysisConfig:
    """Configuration for analysis and visualization."""
    
    # EDA parameters
    correlation_year: int = 2018  # Year for correlation heatmap
    
    # Performance metrics
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    trading_days_per_year: int = 252
    
    # Visualization settings
    figure_size: tuple = (12, 8)
    dpi: int = 100
    style: str = "seaborn-v0_8"
    
    # Output settings
    save_plots: bool = True
    output_dir: str = "output"
    plot_format: str = "png"


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""

    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    # Global settings
    random_seed: int = 42
    verbose: bool = True
    
    def __post_init__(self):
        """Post-initialization to create output directories if needed."""
        if self.analysis.save_plots:
            os.makedirs(self.analysis.output_dir, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config instance from dictionary."""
        data_config = DataConfig(**config_dict.get("data", {}))
        strategy_config = StrategyConfig(**config_dict.get("strategy", {}))
        analysis_config = AnalysisConfig(**config_dict.get("analysis", {}))
        
        return cls(
            data=data_config,
            strategy=strategy_config,
            analysis=analysis_config,
            **{k: v for k, v in config_dict.items() 
               if k not in ["data", "strategy", "analysis"]}
        )
    
    def to_dict(self) -> dict:
        """Convert Config instance to dictionary."""
        return {
            "data": self.data.__dict__,
            "strategy": self.strategy.__dict__,
            "analysis": self.analysis.__dict__,
            "random_seed": self.random_seed,
            "verbose": self.verbose
        }


# Default configuration instance
DEFAULT_CONFIG = Config()


def get_config() -> Config:
    """Get the default configuration."""
    return DEFAULT_CONFIG


def update_config(**kwargs) -> Config:
    """Update the default configuration with new values."""
    config_dict = DEFAULT_CONFIG.to_dict()
    
    # Update nested dictionaries
    for key, value in kwargs.items():
        if key in config_dict and isinstance(config_dict[key], dict):
            config_dict[key].update(value)
        else:
            config_dict[key] = value
    
    return Config.from_dict(config_dict)


# Environment-specific configurations
DEVELOPMENT_CONFIG = Config(
    analysis=AnalysisConfig(
        save_plots=False,
        figure_size=(10, 6)
    ),
    verbose=True
)

PRODUCTION_CONFIG = Config(
    analysis=AnalysisConfig(
        save_plots=True,
        output_dir="results",
        dpi=300
    ),
    verbose=False
)


def get_config_by_env(env: str = "default") -> Config:
    """Get configuration by environment name."""
    configs = {
        "default": DEFAULT_CONFIG,
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG
    }
    
    return configs.get(env, DEFAULT_CONFIG)
