"""
Feature Pipeline for ForexGuard
Transforms raw events into ML-ready features with rolling statistics and behavioral metrics.
"""

import numpy as np
import pandas as pd
from typing import Optional
from datetime import timedelta
from loguru import logger

from .feature_store import RollingFeatureStore


class FeaturePipeline:
    """
    Feature engineering pipeline for forex anomaly detection.
    Computes behavioral features, rolling statistics, and deviation metrics.
    """

    def __init__(self, lookback_window: int = 24):
        """
        Initialize the feature pipeline.

        Args:
            lookback_window: Hours to look back for rolling features
        """
        self.lookback_window = lookback_window
        self.feature_store = RollingFeatureStore(window_hours=lookback_window)
        self.feature_columns = []

    def extract_base_features(self, event: dict) -> dict:
        """Extract basic features from a single event."""
        features = {
            "user_id": event.get("user_id"),
            "timestamp": pd.to_datetime(event.get("timestamp")),
            "event_type": event.get("event_type"),

            # Numerical features
            "amount": event.get("amount", 0),
            "lot_size": event.get("lot_size", 0),
            "margin_used": event.get("margin_used", 0),
            "leverage": event.get("leverage", 0),
            "session_duration": event.get("session_duration", 0),
            "login_success": 1 if event.get("login_success", True) else 0,

            # Categorical encodings
            "is_deposit": 1 if event.get("event_type") == "deposit" else 0,
            "is_withdrawal": 1 if event.get("event_type") == "withdrawal" else 0,
            "is_trade": 1 if event.get("event_type") == "trade" else 0,
            "is_login": 1 if event.get("event_type") == "login" else 0,
            "is_kyc_change": 1 if event.get("event_type") == "kyc_change" else 0,
            "is_password_change": 1 if event.get("event_type") == "password_change" else 0,

            # IP and device hashes (for change detection)
            "ip_hash": hash(event.get("ip_address", "")) % 10000,
            "device_hash": hash(event.get("device", "")) % 100,

            # Payment method encoding
            "payment_crypto": 1 if event.get("payment_method") == "crypto" else 0,

            # Trade direction
            "is_buy": 1 if event.get("direction") == "buy" else 0,
        }

        return features

    def compute_rolling_features(self, user_id: str, current_event: dict) -> dict:
        """Compute rolling window features for a user."""
        history = self.feature_store.get_user_history(user_id)

        if not history:
            # No history - return defaults
            return {
                "event_count_24h": 0,
                "unique_ips_24h": 0,
                "unique_devices_24h": 0,
                "total_amount_24h": 0,
                "total_lot_size_24h": 0,
                "trade_count_24h": 0,
                "deposit_count_24h": 0,
                "withdrawal_count_24h": 0,
                "avg_amount_24h": 0,
                "std_amount_24h": 0,
                "avg_lot_size_24h": 0,
                "std_lot_size_24h": 0,
                "avg_inter_event_time": 0,
                "min_inter_event_time": 0,
                "ip_change_rate": 0,
                "device_change_rate": 0,
                "failed_login_rate": 0,
            }

        df_history = pd.DataFrame(history)

        # Event counts
        event_count = len(df_history)
        trade_count = df_history["is_trade"].sum()
        deposit_count = df_history["is_deposit"].sum()
        withdrawal_count = df_history["is_withdrawal"].sum()

        # Unique IPs and devices
        unique_ips = df_history["ip_hash"].nunique()
        unique_devices = df_history["device_hash"].nunique()

        # Amount statistics
        amounts = df_history["amount"]
        total_amount = amounts.sum()
        avg_amount = amounts.mean() if len(amounts) > 0 else 0
        std_amount = amounts.std() if len(amounts) > 1 else 0

        # Lot size statistics
        lot_sizes = df_history[df_history["is_trade"] == 1]["lot_size"]
        total_lot_size = lot_sizes.sum() if len(lot_sizes) > 0 else 0
        avg_lot_size = lot_sizes.mean() if len(lot_sizes) > 0 else 0
        std_lot_size = lot_sizes.std() if len(lot_sizes) > 1 else 0

        # Inter-event time
        timestamps = pd.to_datetime(df_history["timestamp"]).sort_values()
        if len(timestamps) > 1:
            inter_times = timestamps.diff().dt.total_seconds().dropna()
            avg_inter_event = inter_times.mean()
            min_inter_event = inter_times.min()
        else:
            avg_inter_event = 0
            min_inter_event = 0

        # Change rates
        ip_changes = (df_history["ip_hash"].diff() != 0).sum() if len(df_history) > 1 else 0
        ip_change_rate = ip_changes / event_count if event_count > 0 else 0

        device_changes = (df_history["device_hash"].diff() != 0).sum() if len(df_history) > 1 else 0
        device_change_rate = device_changes / event_count if event_count > 0 else 0

        # Failed login rate
        logins = df_history[df_history["is_login"] == 1]
        if len(logins) > 0:
            failed_login_rate = 1 - logins["login_success"].mean()
        else:
            failed_login_rate = 0

        return {
            "event_count_24h": event_count,
            "unique_ips_24h": unique_ips,
            "unique_devices_24h": unique_devices,
            "total_amount_24h": total_amount,
            "total_lot_size_24h": total_lot_size,
            "trade_count_24h": trade_count,
            "deposit_count_24h": deposit_count,
            "withdrawal_count_24h": withdrawal_count,
            "avg_amount_24h": avg_amount,
            "std_amount_24h": std_amount,
            "avg_lot_size_24h": avg_lot_size,
            "std_lot_size_24h": std_lot_size,
            "avg_inter_event_time": avg_inter_event,
            "min_inter_event_time": min_inter_event,
            "ip_change_rate": ip_change_rate,
            "device_change_rate": device_change_rate,
            "failed_login_rate": failed_login_rate,
        }

    def compute_deviation_features(self, current: dict, rolling: dict) -> dict:
        """Compute deviation from normal behavior."""
        deviations = {}

        # Amount deviation (z-score)
        if rolling["std_amount_24h"] > 0:
            deviations["amount_zscore"] = (current["amount"] - rolling["avg_amount_24h"]) / rolling["std_amount_24h"]
        else:
            deviations["amount_zscore"] = 0

        # Lot size deviation
        if rolling["std_lot_size_24h"] > 0:
            deviations["lot_size_zscore"] = (current["lot_size"] - rolling["avg_lot_size_24h"]) / rolling["std_lot_size_24h"]
        else:
            deviations["lot_size_zscore"] = 0

        # Activity spike detection
        deviations["activity_spike"] = 1 if rolling["event_count_24h"] > 50 else 0
        deviations["trade_spike"] = 1 if rolling["trade_count_24h"] > 30 else 0

        # Suspicious patterns
        deviations["rapid_ip_switching"] = 1 if rolling["unique_ips_24h"] > 5 else 0
        deviations["rapid_device_switching"] = 1 if rolling["unique_devices_24h"] > 3 else 0

        # Deposit-withdrawal imbalance
        if rolling["deposit_count_24h"] > 0:
            deviations["withdrawal_deposit_ratio"] = rolling["withdrawal_count_24h"] / rolling["deposit_count_24h"]
        else:
            deviations["withdrawal_deposit_ratio"] = rolling["withdrawal_count_24h"]

        # High velocity (events very close together)
        deviations["high_velocity"] = 1 if rolling["min_inter_event_time"] < 10 else 0  # Less than 10 seconds

        return deviations

    def transform_event(self, event: dict, update_store: bool = True) -> dict:
        """
        Transform a single event into features.

        Args:
            event: Raw event dictionary
            update_store: Whether to add this event to the rolling feature store

        Returns:
            Dictionary of features
        """
        user_id = event.get("user_id")

        # Extract base features
        base_features = self.extract_base_features(event)

        # Get rolling features (before adding current event)
        rolling_features = self.compute_rolling_features(user_id, event)

        # Compute deviation features
        deviation_features = self.compute_deviation_features(base_features, rolling_features)

        # Update feature store with current event
        if update_store:
            self.feature_store.add_event(user_id, base_features)

        # Combine all features
        all_features = {**base_features, **rolling_features, **deviation_features}

        return all_features

    def transform_batch(self, events: list[dict]) -> pd.DataFrame:
        """Transform a batch of events into a feature DataFrame."""
        features_list = []

        for event in events:
            features = self.transform_event(event, update_store=True)
            features_list.append(features)

        df = pd.DataFrame(features_list)
        self.feature_columns = [c for c in df.columns if c not in ["user_id", "timestamp", "event_type"]]

        return df

    def get_model_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract numerical features suitable for ML models."""
        feature_cols = [
            # Base features
            "amount", "lot_size", "margin_used", "leverage", "session_duration",
            "is_deposit", "is_withdrawal", "is_trade", "is_login",
            "is_kyc_change", "is_password_change", "payment_crypto", "is_buy",

            # Rolling features
            "event_count_24h", "unique_ips_24h", "unique_devices_24h",
            "total_amount_24h", "total_lot_size_24h", "trade_count_24h",
            "deposit_count_24h", "withdrawal_count_24h",
            "avg_amount_24h", "std_amount_24h",
            "avg_lot_size_24h", "std_lot_size_24h",
            "avg_inter_event_time", "min_inter_event_time",
            "ip_change_rate", "device_change_rate", "failed_login_rate",

            # Deviation features
            "amount_zscore", "lot_size_zscore",
            "activity_spike", "trade_spike",
            "rapid_ip_switching", "rapid_device_switching",
            "withdrawal_deposit_ratio", "high_velocity"
        ]

        # Filter to columns that exist
        available_cols = [c for c in feature_cols if c in df.columns]

        X = df[available_cols].fillna(0).values

        return X, available_cols

    def reset(self):
        """Reset the feature store."""
        self.feature_store.reset()


def create_training_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Create features for training from a raw events DataFrame.

    Args:
        df: DataFrame with raw events

    Returns:
        Tuple of (feature_matrix, feature_names, full_feature_df)
    """
    pipeline = FeaturePipeline()

    # Convert DataFrame to list of dicts
    events = df.to_dict(orient="records")

    # Transform batch
    feature_df = pipeline.transform_batch(events)

    # Get model-ready features
    X, feature_names = pipeline.get_model_features(feature_df)

    logger.info(f"Created {X.shape[1]} features for {X.shape[0]} events")

    return X, feature_names, feature_df


if __name__ == "__main__":
    # Test with sample data
    sample_events = [
        {
            "event_id": "1",
            "user_id": "USER_00001",
            "timestamp": "2024-01-15T10:30:00",
            "event_type": "login",
            "ip_address": "192.168.1.100",
            "device": "Windows Desktop",
            "login_success": True
        },
        {
            "event_id": "2",
            "user_id": "USER_00001",
            "timestamp": "2024-01-15T10:35:00",
            "event_type": "trade",
            "ip_address": "192.168.1.100",
            "device": "Windows Desktop",
            "instrument": "EUR/USD",
            "lot_size": 0.5,
            "direction": "buy",
            "margin_used": 250,
            "leverage": 100
        },
        {
            "event_id": "3",
            "user_id": "USER_00001",
            "timestamp": "2024-01-15T10:36:00",
            "event_type": "trade",
            "ip_address": "10.0.100.50",  # IP change!
            "device": "iOS Mobile",  # Device change!
            "instrument": "GBP/USD",
            "lot_size": 10.0,  # Much larger lot size
            "direction": "sell",
            "margin_used": 5000,
            "leverage": 200
        }
    ]

    pipeline = FeaturePipeline()
    feature_df = pipeline.transform_batch(sample_events)

    print("\nFeature DataFrame:")
    print(feature_df.T)

    X, feature_names = pipeline.get_model_features(feature_df)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Feature names: {feature_names}")
