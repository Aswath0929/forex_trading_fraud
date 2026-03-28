"""API module for ForexGuard."""

from .schemas import (
    EventInput,
    AnomalyScore,
    BatchEventInput,
    BatchAnomalyScore,
    Alert,
    SeverityLevel,
    HealthResponse,
    ModelInfo
)
from .alerts import AlertGenerator, get_alert_generator

__all__ = [
    "EventInput",
    "AnomalyScore",
    "BatchEventInput",
    "BatchAnomalyScore",
    "Alert",
    "SeverityLevel",
    "HealthResponse",
    "ModelInfo",
    "AlertGenerator",
    "get_alert_generator"
]
