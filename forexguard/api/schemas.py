"""
Pydantic Schemas for ForexGuard API
Defines request/response models for the anomaly detection API.
"""

from datetime import datetime
from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Supported event types."""
    LOGIN = "login"
    LOGOUT = "logout"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRADE = "trade"
    KYC_CHANGE = "kyc_change"
    PASSWORD_CHANGE = "password_change"


class SeverityLevel(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventInput(BaseModel):
    """Input schema for a single event."""
    event_id: str = Field(..., description="Unique event identifier")
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Event timestamp")
    event_type: EventType = Field(..., description="Type of event")
    ip_address: str = Field(..., description="Client IP address")
    device: str = Field(..., description="Device type/identifier")

    # Optional fields based on event type
    session_id: Optional[str] = Field(None, description="Session ID for login events")
    login_success: Optional[bool] = Field(None, description="Login success status")
    session_duration: Optional[int] = Field(None, description="Session duration in seconds")
    amount: Optional[float] = Field(None, description="Transaction amount")
    currency: Optional[str] = Field("USD", description="Currency code")
    payment_method: Optional[str] = Field(None, description="Payment method")
    instrument: Optional[str] = Field(None, description="Trading instrument (e.g., EUR/USD)")
    lot_size: Optional[float] = Field(None, description="Trade lot size")
    direction: Optional[str] = Field(None, description="Trade direction (buy/sell)")
    margin_used: Optional[float] = Field(None, description="Margin used for trade")
    leverage: Optional[int] = Field(None, description="Leverage used")
    field_changed: Optional[str] = Field(None, description="KYC field that changed")
    change_method: Optional[str] = Field(None, description="Password change method")

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_123456",
                "user_id": "USER_00001",
                "timestamp": "2024-01-15T10:30:00Z",
                "event_type": "trade",
                "ip_address": "192.168.1.100",
                "device": "Windows Desktop",
                "instrument": "EUR/USD",
                "lot_size": 1.5,
                "direction": "buy",
                "margin_used": 750.0,
                "leverage": 100
            }
        }


class FeatureContribution(BaseModel):
    """Feature contribution to anomaly score."""
    feature_name: str = Field(..., description="Name of the feature")
    value: float = Field(..., description="Feature value")
    contribution: float = Field(..., description="Contribution to anomaly score")
    description: str = Field(..., description="Human-readable explanation")


class AnomalyScore(BaseModel):
    """Anomaly score response."""
    event_id: str = Field(..., description="Event identifier")
    user_id: str = Field(..., description="User identifier")
    anomaly_score: float = Field(..., ge=0, le=1, description="Anomaly score (0=normal, 1=anomalous)")
    is_anomaly: bool = Field(..., description="Whether the event is flagged as anomalous")
    threshold: float = Field(..., description="Threshold used for anomaly detection")
    model_type: str = Field(..., description="Model used for scoring")
    severity: SeverityLevel = Field(..., description="Alert severity level")
    feature_contributions: list[FeatureContribution] = Field(
        default_factory=list,
        description="Feature contributions to anomaly score"
    )
    explanation: str = Field(..., description="Human-readable explanation")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Scoring timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_123456",
                "user_id": "USER_00001",
                "anomaly_score": 0.85,
                "is_anomaly": True,
                "threshold": 0.7,
                "model_type": "isolation_forest",
                "severity": "high",
                "feature_contributions": [
                    {
                        "feature_name": "ip_change_frequency",
                        "value": 5.0,
                        "contribution": 0.35,
                        "description": "Unusually high IP change frequency"
                    }
                ],
                "explanation": "User flagged due to unusual IP switching pattern and trade volume spike",
                "timestamp": "2024-01-15T10:30:05Z"
            }
        }


class BatchEventInput(BaseModel):
    """Input schema for batch scoring."""
    events: list[EventInput] = Field(..., min_length=1, max_length=1000)


class BatchAnomalyScore(BaseModel):
    """Response schema for batch scoring."""
    scores: list[AnomalyScore]
    total_events: int
    anomalies_found: int
    processing_time_ms: float


class Alert(BaseModel):
    """Alert schema for notifications."""
    alert_id: str = Field(..., description="Unique alert identifier")
    event_id: str = Field(..., description="Related event identifier")
    user_id: str = Field(..., description="User identifier")
    severity: SeverityLevel = Field(..., description="Alert severity")
    anomaly_score: float = Field(..., description="Anomaly score")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Detailed alert description")
    recommended_action: str = Field(..., description="Recommended action to take")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = Field(default=False)
    acknowledged_by: Optional[str] = Field(None)
    acknowledged_at: Optional[datetime] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "alert_id": "alert_789",
                "event_id": "evt_123456",
                "user_id": "USER_00001",
                "severity": "high",
                "anomaly_score": 0.85,
                "title": "Suspicious Trading Activity Detected",
                "description": "User USER_00001 flagged for unusual IP switching pattern combined with abnormal trade volume spike",
                "recommended_action": "Review user activity and consider temporary account restriction",
                "created_at": "2024-01-15T10:30:05Z",
                "acknowledged": False
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: Optional[str]
    version: str


class ModelInfo(BaseModel):
    """Model information response."""
    model_id: str
    model_type: str
    version: str
    created_at: datetime
    metrics: dict
    is_active: bool
