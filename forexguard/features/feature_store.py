"""
Rolling Feature Store for ForexGuard
Maintains sliding window of user events for real-time feature computation.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd


def ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (UTC)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_timestamp(ts) -> datetime:
    """Parse timestamp to timezone-aware datetime."""
    if ts is None:
        return datetime.now(timezone.utc)
    if isinstance(ts, datetime):
        return ensure_utc(ts)
    if isinstance(ts, str):
        parsed = pd.to_datetime(ts)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.to_pydatetime().astimezone(timezone.utc)
    # pandas Timestamp
    if hasattr(ts, 'to_pydatetime'):
        dt = ts.to_pydatetime()
        return ensure_utc(dt)
    return datetime.now(timezone.utc)


class RollingFeatureStore:
    """
    In-memory store for computing rolling window features.
    Maintains recent event history per user for real-time feature engineering.
    """

    def __init__(self, window_hours: int = 24, max_events_per_user: int = 1000):
        """
        Initialize the feature store.

        Args:
            window_hours: Size of the rolling window in hours
            max_events_per_user: Maximum events to retain per user (prevents memory issues)
        """
        self.window_hours = window_hours
        self.max_events_per_user = max_events_per_user
        self.user_events: dict[str, list[dict]] = defaultdict(list)
        self.user_stats: dict[str, dict] = defaultdict(dict)

    def add_event(self, user_id: str, event_features: dict):
        """
        Add an event to the store.

        Args:
            user_id: User identifier
            event_features: Dictionary of extracted features for the event
        """
        # Add timestamp if not present
        if "timestamp" not in event_features:
            event_features["timestamp"] = datetime.now(timezone.utc)

        self.user_events[user_id].append(event_features)

        # Enforce max events limit
        if len(self.user_events[user_id]) > self.max_events_per_user:
            self.user_events[user_id] = self.user_events[user_id][-self.max_events_per_user:]

        # Update user statistics
        self._update_user_stats(user_id)

    def get_user_history(
        self,
        user_id: str,
        window_hours: Optional[int] = None
    ) -> list[dict]:
        """
        Get user's event history within the rolling window.

        Args:
            user_id: User identifier
            window_hours: Optional override for window size

        Returns:
            List of event feature dictionaries within the window
        """
        if user_id not in self.user_events:
            return []

        window = window_hours or self.window_hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=window)

        # Filter events within window
        filtered_events = []
        for event in self.user_events[user_id]:
            event_time = event.get("timestamp")
            event_time = parse_timestamp(event_time)

            # Include events without timestamp or within window
            if event_time is None or event_time >= cutoff_time:
                filtered_events.append(event)

        return filtered_events

    def _update_user_stats(self, user_id: str):
        """Update aggregate statistics for a user."""
        history = self.get_user_history(user_id)

        if not history:
            return

        df = pd.DataFrame(history)

        self.user_stats[user_id] = {
            "total_events": len(history),
            "last_event_time": df["timestamp"].max() if "timestamp" in df else None,
            "unique_ips": df["ip_hash"].nunique() if "ip_hash" in df else 0,
            "unique_devices": df["device_hash"].nunique() if "device_hash" in df else 0,
        }

    def get_user_stats(self, user_id: str) -> dict:
        """Get cached statistics for a user."""
        return self.user_stats.get(user_id, {})

    def get_all_user_ids(self) -> list[str]:
        """Get list of all users in the store."""
        return list(self.user_events.keys())

    def cleanup_old_events(self):
        """Remove events outside the rolling window for all users."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.window_hours)

        for user_id in list(self.user_events.keys()):
            filtered = []
            for event in self.user_events[user_id]:
                event_time = event.get("timestamp")
                event_time = parse_timestamp(event_time)

                if event_time is None or event_time >= cutoff_time:
                    filtered.append(event)

            if filtered:
                self.user_events[user_id] = filtered
            else:
                del self.user_events[user_id]
                if user_id in self.user_stats:
                    del self.user_stats[user_id]

    def reset(self):
        """Clear all stored data."""
        self.user_events.clear()
        self.user_stats.clear()

    def get_memory_usage(self) -> dict:
        """Get approximate memory usage statistics."""
        total_events = sum(len(events) for events in self.user_events.values())
        return {
            "num_users": len(self.user_events),
            "total_events": total_events,
            "avg_events_per_user": total_events / len(self.user_events) if self.user_events else 0
        }


class GlobalFeatureStore:
    """
    Global statistics store for computing population-level baselines.
    Used for detecting deviations from normal behavior across all users.
    """

    def __init__(self, decay_factor: float = 0.99):
        """
        Initialize global store with exponential moving statistics.

        Args:
            decay_factor: Weight for exponential moving average (closer to 1 = slower decay)
        """
        self.decay_factor = decay_factor
        self.stats = {
            "amount_mean": 0.0,
            "amount_var": 0.0,
            "lot_size_mean": 0.0,
            "lot_size_var": 0.0,
            "events_per_user_mean": 0.0,
            "trades_per_user_mean": 0.0,
        }
        self.n_samples = 0

    def update(self, features: dict):
        """Update global statistics with new observation using EMA."""
        self.n_samples += 1
        alpha = 1 - self.decay_factor

        # Update amount statistics
        if features.get("amount", 0) > 0:
            delta = features["amount"] - self.stats["amount_mean"]
            self.stats["amount_mean"] += alpha * delta
            self.stats["amount_var"] = self.decay_factor * (
                self.stats["amount_var"] + alpha * delta * delta
            )

        # Update lot size statistics
        if features.get("lot_size", 0) > 0:
            delta = features["lot_size"] - self.stats["lot_size_mean"]
            self.stats["lot_size_mean"] += alpha * delta
            self.stats["lot_size_var"] = self.decay_factor * (
                self.stats["lot_size_var"] + alpha * delta * delta
            )

    def get_global_stats(self) -> dict:
        """Get current global statistics."""
        return {
            **self.stats,
            "amount_std": self.stats["amount_var"] ** 0.5,
            "lot_size_std": self.stats["lot_size_var"] ** 0.5,
            "n_samples": self.n_samples
        }

    def compute_zscore(self, value: float, stat_name: str) -> float:
        """Compute z-score relative to global statistics."""
        mean = self.stats.get(f"{stat_name}_mean", 0)
        std = self.stats.get(f"{stat_name}_var", 0) ** 0.5

        if std == 0:
            return 0

        return (value - mean) / std


class EntityHubStore:
    """Tracks shared IP/device usage across users within a rolling time window.

    This enables client-portal access anomalies such as:
    - Many users logging in from the same IP (IP hub behavior)
    - Many users using the same device fingerprint
    """

    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours
        # key -> {user_id: last_seen_ts}
        self.ip_users: dict[int, dict[str, datetime]] = defaultdict(dict)
        self.device_users: dict[int, dict[str, datetime]] = defaultdict(dict)

    def _cutoff(self, now: datetime) -> datetime:
        return ensure_utc(now) - timedelta(hours=self.window_hours)

    def _cleanup_map(self, m: dict[int, dict[str, datetime]], cutoff: datetime):
        for key in list(m.keys()):
            users = m[key]
            for uid, ts in list(users.items()):
                if ensure_utc(ts) < cutoff:
                    del users[uid]
            if not users:
                del m[key]

    def get_shared_counts(
        self,
        user_id: str,
        ip_hash: Optional[int],
        device_hash: Optional[int],
        now: datetime,
    ) -> tuple[int, int]:
        """Return (ip_shared_users, device_shared_users) excluding current user."""
        now = ensure_utc(now)
        cutoff = self._cutoff(now)
        self._cleanup_map(self.ip_users, cutoff)
        self._cleanup_map(self.device_users, cutoff)

        ip_shared = 0
        if ip_hash is not None:
            users = self.ip_users.get(ip_hash, {})
            ip_shared = max(0, len(users) - (1 if user_id in users else 0))

        device_shared = 0
        if device_hash is not None:
            users = self.device_users.get(device_hash, {})
            device_shared = max(0, len(users) - (1 if user_id in users else 0))

        return ip_shared, device_shared

    def update(self, user_id: str, ip_hash: Optional[int], device_hash: Optional[int], now: datetime):
        now = ensure_utc(now)
        if ip_hash is not None:
            self.ip_users[ip_hash][user_id] = now
        if device_hash is not None:
            self.device_users[device_hash][user_id] = now
