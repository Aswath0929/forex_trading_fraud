"""
Feature Pipeline for ForexGuard
Transforms raw events into ML-ready features with rolling statistics and behavioral metrics.
"""

import numpy as np
import pandas as pd
from typing import Optional
from datetime import timedelta, timezone
from loguru import logger

from .feature_store import RollingFeatureStore, EntityHubStore, parse_timestamp


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
        self.entity_hub = EntityHubStore(window_hours=lookback_window)
        self.feature_columns = []

    def extract_base_features(self, event: dict) -> dict:
        """Extract basic features from a single event."""
        event_type = event.get("event_type")
        ts = parse_timestamp(event.get("timestamp"))

        # Login success is only meaningful for login events; default to 1 otherwise.
        if event_type == "login":
            login_success = 1 if event.get("login_success") is True else 0
        else:
            login_success = 1

        ticket_priority = (event.get("ticket_priority") or "").lower()
        ticket_priority_score = {"low": 0, "medium": 1, "high": 2, "urgent": 3}.get(ticket_priority, 0)

        features = {
            "user_id": event.get("user_id"),
            "timestamp": ts,
            "event_type": event_type,

            # Time features
            "event_hour": float(ts.hour),
            "is_night": 1 if ts.hour in (0, 1, 2, 3, 4, 5) else 0,

            # Numerical features
            "amount": float(event.get("amount") or 0),
            "lot_size": float(event.get("lot_size") or 0),
            "margin_used": float(event.get("margin_used") or 0),
            "leverage": float(event.get("leverage") or 0),
            "session_duration": float(event.get("session_duration") or 0),
            "login_success": float(login_success),
            "login_failure": 1.0 if (event_type == "login" and login_success == 0) else 0.0,

            # Client portal encodings
            "is_page_view": 1 if event_type == "page_view" else 0,
            "is_profile_update": 1 if event_type == "profile_update" else 0,
            "is_support_ticket": 1 if event_type == "support_ticket" else 0,
            "is_document_upload": 1 if event_type == "document_upload" else 0,
            "is_logout": 1 if event_type == "logout" else 0,

            # Trading / payments encodings
            "is_deposit": 1 if event_type == "deposit" else 0,
            "is_withdrawal": 1 if event_type == "withdrawal" else 0,
            "is_trade": 1 if event_type == "trade" else 0,
            "is_login": 1 if event_type == "login" else 0,
            "is_kyc_change": 1 if event_type == "kyc_change" else 0,
            "is_password_change": 1 if event_type == "password_change" else 0,

            # Hashes for change detection / diversity
            "ip_hash": hash(event.get("ip_address", "")) % 10000,
            "device_hash": hash(event.get("device", "")) % 100,
            "page_hash": hash(event.get("page", "")) % 500,
            "document_type_hash": hash(event.get("document_type", "")) % 100,
            "field_changed_hash": hash(event.get("field_changed", "")) % 50,

            # Support ticket signals
            "ticket_priority_score": float(ticket_priority_score),

            # Document upload signals
            "upload_success": 1.0 if event.get("upload_success", True) else 0.0,

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

                # Portal behavior
                "page_view_count_24h": 0,
                "unique_pages_24h": 0,
                "profile_update_count_24h": 0,
                "support_ticket_count_24h": 0,
                "document_upload_count_24h": 0,
                "document_upload_fail_rate_24h": 0,
                "avg_ticket_priority_24h": 0,

                # Account changes
                "kyc_change_count_24h": 0,
                "password_change_count_24h": 0,

                # Account access
                "login_count_24h": 0,
                "login_fail_count_24h": 0,
                "failed_login_rate": 0,
                "consecutive_login_failures": 0,

                # Session
                "avg_session_duration_24h": 0,
                "std_session_duration_24h": 0,

                # Temporal
                "time_since_last_event": 0,
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
        timestamps = pd.to_datetime(df_history["timestamp"], utc=True).sort_values()
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

        # Account access
        logins = df_history[df_history["is_login"] == 1]
        login_count = len(logins)
        login_fail_count = int(df_history.get("login_failure", pd.Series(dtype=float)).sum()) if "login_failure" in df_history else 0
        if login_count > 0:
            failed_login_rate = 1 - logins["login_success"].mean()
        else:
            failed_login_rate = 0

        # Consecutive login failures (most recent login events only)
        consecutive_login_failures = 0
        recent_logins = [e for e in reversed(history) if e.get("is_login") == 1]
        for e in recent_logins:
            if e.get("login_success", 1) == 0:
                consecutive_login_failures += 1
            else:
                break

        # Portal behavior
        page_views = df_history[df_history.get("is_page_view", 0) == 1] if "is_page_view" in df_history else pd.DataFrame()
        page_view_count = len(page_views) if len(page_views) > 0 else 0
        unique_pages = int(page_views["page_hash"].nunique()) if len(page_views) > 0 and "page_hash" in page_views else 0

        profile_update_count = int(df_history["is_profile_update"].sum()) if "is_profile_update" in df_history else 0
        support_ticket_count = int(df_history["is_support_ticket"].sum()) if "is_support_ticket" in df_history else 0
        document_upload_count = int(df_history["is_document_upload"].sum()) if "is_document_upload" in df_history else 0

        kyc_change_count = int(df_history["is_kyc_change"].sum()) if "is_kyc_change" in df_history else 0
        password_change_count = int(df_history["is_password_change"].sum()) if "is_password_change" in df_history else 0

        doc_uploads = df_history[df_history.get("is_document_upload", 0) == 1] if "is_document_upload" in df_history else pd.DataFrame()
        if len(doc_uploads) > 0 and "upload_success" in doc_uploads:
            document_upload_fail_rate = 1 - doc_uploads["upload_success"].mean()
        else:
            document_upload_fail_rate = 0

        tickets = df_history[df_history.get("is_support_ticket", 0) == 1] if "is_support_ticket" in df_history else pd.DataFrame()
        if len(tickets) > 0 and "ticket_priority_score" in tickets:
            avg_ticket_priority = tickets["ticket_priority_score"].mean()
        else:
            avg_ticket_priority = 0

        # Session duration stats (logout events)
        logouts = df_history[df_history.get("is_logout", 0) == 1] if "is_logout" in df_history else pd.DataFrame()
        if len(logouts) > 0 and "session_duration" in logouts:
            sd = logouts["session_duration"].astype(float)
            avg_session_duration = sd.mean()
            std_session_duration = sd.std() if len(sd) > 1 else 0
        else:
            avg_session_duration = 0
            std_session_duration = 0

        # Temporal
        current_ts = parse_timestamp(current_event.get("timestamp"))
        last_ts = timestamps.max().to_pydatetime() if len(timestamps) > 0 else None
        if last_ts is not None:
            time_since_last_event = max(0.0, (current_ts - last_ts).total_seconds())
        else:
            time_since_last_event = 0

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

            # Portal behavior
            "page_view_count_24h": page_view_count,
            "unique_pages_24h": unique_pages,
            "profile_update_count_24h": profile_update_count,
            "support_ticket_count_24h": support_ticket_count,
            "document_upload_count_24h": document_upload_count,
            "document_upload_fail_rate_24h": document_upload_fail_rate,
            "avg_ticket_priority_24h": avg_ticket_priority,

            # Account changes
            "kyc_change_count_24h": kyc_change_count,
            "password_change_count_24h": password_change_count,

            # Account access
            "login_count_24h": login_count,
            "login_fail_count_24h": login_fail_count,
            "failed_login_rate": failed_login_rate,
            "consecutive_login_failures": consecutive_login_failures,

            # Session
            "avg_session_duration_24h": avg_session_duration,
            "std_session_duration_24h": std_session_duration,

            # Temporal
            "time_since_last_event": time_since_last_event,
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

        # Portal / access anomalies
        deviations["rapid_navigation"] = 1 if (rolling.get("page_view_count_24h", 0) > 30 and rolling["min_inter_event_time"] < 2) else 0
        deviations["page_diversity_spike"] = 1 if rolling.get("unique_pages_24h", 0) > 20 else 0
        deviations["document_upload_spike"] = 1 if rolling.get("document_upload_count_24h", 0) > 10 else 0
        deviations["support_ticket_spike"] = 1 if rolling.get("support_ticket_count_24h", 0) > 5 else 0
        deviations["account_change_rush"] = 1 if (
            rolling.get("profile_update_count_24h", 0)
            + rolling.get("kyc_change_count_24h", 0)
            + rolling.get("password_change_count_24h", 0)
        ) > 5 else 0

        deviations["login_bruteforce"] = 1 if (
            rolling.get("consecutive_login_failures", 0) >= 3
            or rolling.get("login_fail_count_24h", 0) >= 6
            or rolling.get("failed_login_rate", 0) >= 0.4
        ) else 0

        # Session duration deviation (only meaningful when a session_duration is present)
        if rolling.get("std_session_duration_24h", 0) > 0 and current.get("session_duration", 0) > 0:
            deviations["session_duration_zscore"] = (
                current["session_duration"] - rolling.get("avg_session_duration_24h", 0)
            ) / rolling["std_session_duration_24h"]
        else:
            deviations["session_duration_zscore"] = 0

        # Dormancy -> withdrawal pattern
        deviations["dormancy_withdrawal"] = 1 if (
            current.get("is_withdrawal", 0) == 1
            and rolling.get("time_since_last_event", 0) > (7 * 24 * 3600)
            and current.get("amount", 0) > max(5000.0, rolling.get("avg_amount_24h", 0) * 3)
        ) else 0

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

        # Global hub signals (shared IP/device across users)
        ip_shared_users, device_shared_users = self.entity_hub.get_shared_counts(
            user_id=user_id,
            ip_hash=base_features.get("ip_hash"),
            device_hash=base_features.get("device_hash"),
            now=base_features.get("timestamp"),
        )

        hub_features = {
            "ip_shared_users_24h": float(ip_shared_users),
            "device_shared_users_24h": float(device_shared_users),
            "ip_hub_behavior": 1.0 if ip_shared_users >= 5 else 0.0,
            "device_hub_behavior": 1.0 if device_shared_users >= 3 else 0.0,
        }

        # Update stores with current event
        if update_store:
            self.feature_store.add_event(user_id, base_features)
            self.entity_hub.update(
                user_id=user_id,
                ip_hash=base_features.get("ip_hash"),
                device_hash=base_features.get("device_hash"),
                now=base_features.get("timestamp"),
            )

        # Combine all features
        all_features = {**base_features, **rolling_features, **deviation_features, **hub_features}

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
            "event_hour", "is_night",
            "login_success", "login_failure",
            "ticket_priority_score", "upload_success",
            "is_page_view", "is_profile_update", "is_support_ticket", "is_document_upload", "is_logout",
            "is_deposit", "is_withdrawal", "is_trade", "is_login",
            "is_kyc_change", "is_password_change", "payment_crypto", "is_buy",

            # Rolling features
            "event_count_24h", "unique_ips_24h", "unique_devices_24h",
            "total_amount_24h", "total_lot_size_24h", "trade_count_24h",
            "deposit_count_24h", "withdrawal_count_24h",
            "avg_amount_24h", "std_amount_24h",
            "avg_lot_size_24h", "std_lot_size_24h",
            "avg_inter_event_time", "min_inter_event_time",
            "ip_change_rate", "device_change_rate",

            # Portal rolling
            "page_view_count_24h", "unique_pages_24h",
            "profile_update_count_24h", "support_ticket_count_24h",
            "document_upload_count_24h", "document_upload_fail_rate_24h",
            "avg_ticket_priority_24h",
            "kyc_change_count_24h", "password_change_count_24h",

            # Access rolling
            "login_count_24h", "login_fail_count_24h", "failed_login_rate", "consecutive_login_failures",

            # Session/temporal rolling
            "avg_session_duration_24h", "std_session_duration_24h",
            "time_since_last_event",

            # Hub features
            "ip_shared_users_24h", "device_shared_users_24h",

            # Deviation features
            "amount_zscore", "lot_size_zscore", "session_duration_zscore",
            "activity_spike", "trade_spike",
            "rapid_ip_switching", "rapid_device_switching",
            "withdrawal_deposit_ratio", "high_velocity",
            "rapid_navigation", "page_diversity_spike",
            "document_upload_spike", "support_ticket_spike",
            "account_change_rush", "login_bruteforce", "dormancy_withdrawal",
            "ip_hub_behavior", "device_hub_behavior",
        ]

        # Filter to columns that exist
        available_cols = [c for c in feature_cols if c in df.columns]

        X = df[available_cols].fillna(0).values

        return X, available_cols

    def reset(self):
        """Reset the feature store(s)."""
        self.feature_store.reset()
        self.entity_hub = EntityHubStore(window_hours=self.lookback_window)


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
