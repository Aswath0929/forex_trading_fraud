"""
Configuration for ForexGuard
Centralized settings with environment variable support.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


def env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    return int(os.getenv(key, str(default)))


def env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    return float(os.getenv(key, str(default)))


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: env_int("PORT", 8000))
    debug: bool = field(default_factory=lambda: env_bool("DEBUG", False))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    cors_origins: list[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))


@dataclass
class ModelConfig:
    """Model configuration."""
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "models/saved"))
    default_model_type: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "isolation_forest"))
    anomaly_threshold: float = field(default_factory=lambda: env_float("ANOMALY_THRESHOLD", 0.5))

    # Isolation Forest
    if_contamination: float = field(default_factory=lambda: env_float("IF_CONTAMINATION", 0.05))
    if_n_estimators: int = field(default_factory=lambda: env_int("IF_N_ESTIMATORS", 200))

    # LSTM Autoencoder
    lstm_seq_len: int = field(default_factory=lambda: env_int("LSTM_SEQ_LEN", 10))
    lstm_hidden_dim: int = field(default_factory=lambda: env_int("LSTM_HIDDEN_DIM", 64))
    lstm_latent_dim: int = field(default_factory=lambda: env_int("LSTM_LATENT_DIM", 32))
    lstm_epochs: int = field(default_factory=lambda: env_int("LSTM_EPOCHS", 30))


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    lookback_hours: int = field(default_factory=lambda: env_int("LOOKBACK_HOURS", 24))
    max_events_per_user: int = field(default_factory=lambda: env_int("MAX_EVENTS_PER_USER", 1000))


@dataclass
class KafkaConfig:
    """Kafka configuration."""
    enabled: bool = field(default_factory=lambda: env_bool("KAFKA_ENABLED", False))
    bootstrap_servers: str = field(default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    input_topic: str = field(default_factory=lambda: os.getenv("KAFKA_INPUT_TOPIC", "forex-events"))
    output_topic: str = field(default_factory=lambda: os.getenv("KAFKA_OUTPUT_TOPIC", "forex-alerts"))
    group_id: str = field(default_factory=lambda: os.getenv("KAFKA_GROUP_ID", "forexguard-processor"))


@dataclass
class AlertConfig:
    """Alert configuration."""
    threshold: float = field(default_factory=lambda: env_float("ALERT_THRESHOLD", 0.5))
    max_alerts: int = field(default_factory=lambda: env_int("MAX_ALERTS", 10000))

    # Severity thresholds
    critical_threshold: float = field(default_factory=lambda: env_float("CRITICAL_THRESHOLD", 0.95))
    high_threshold: float = field(default_factory=lambda: env_float("HIGH_THRESHOLD", 0.85))
    medium_threshold: float = field(default_factory=lambda: env_float("MEDIUM_THRESHOLD", 0.70))


@dataclass
class Config:
    """Main configuration container."""
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent / "models" / "saved")


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration."""
    return config


def reload_config() -> Config:
    """Reload configuration from environment."""
    global config
    config = Config()
    return config
