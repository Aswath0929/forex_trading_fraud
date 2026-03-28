"""
Synthetic Dataset Generator for ForexGuard
Generates ~50,000 events with realistic anomalies for forex brokerage fraud detection.
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# Constants
NUM_USERS = 500
NUM_EVENTS = 50000
ANOMALY_RATE = 0.05  # 5% anomalous events

# IP pools
NORMAL_IP_POOL = [f"192.168.{i}.{j}" for i in range(1, 10) for j in range(1, 255)]
SUSPICIOUS_IP_POOL = [f"10.0.{i}.{j}" for i in range(100, 110) for j in range(1, 255)]
VPN_IP_POOL = [f"185.{i}.{j}.{k}" for i in range(100, 105) for j in range(1, 50) for k in range(1, 50)]

# Device types
DEVICES = ["Windows Desktop", "MacOS Desktop", "iOS Mobile", "Android Mobile", "Linux Desktop"]

# Forex instruments
INSTRUMENTS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "EUR/GBP", "USD/CAD", "NZD/USD"]

# Event types
EVENT_TYPES = ["login", "logout", "deposit", "withdrawal", "trade", "kyc_change", "password_change"]


class UserProfile:
    """Represents a synthetic user with behavioral patterns."""

    def __init__(self, user_id: str, is_anomalous: bool = False):
        self.user_id = user_id
        self.is_anomalous = is_anomalous

        # Normal behavior patterns
        self.primary_ip = random.choice(NORMAL_IP_POOL)
        self.secondary_ip = random.choice(NORMAL_IP_POOL)
        self.primary_device = random.choice(DEVICES)
        self.preferred_instruments = random.sample(INSTRUMENTS, k=random.randint(2, 4))

        # Trading behavior
        self.avg_lot_size = random.uniform(0.1, 2.0)
        self.lot_size_std = self.avg_lot_size * 0.2
        self.avg_trades_per_day = random.randint(1, 20)
        self.avg_session_duration = random.randint(300, 7200)  # 5 min to 2 hours

        # Financial behavior
        self.avg_deposit = random.uniform(500, 10000)
        self.avg_withdrawal = random.uniform(200, 5000)
        self.deposit_frequency_days = random.randint(7, 60)

        # Anomalous users get additional suspicious patterns
        if is_anomalous:
            self.anomaly_type = random.choice([
                "ip_switcher",
                "trade_spiker",
                "deposit_withdrawal_abuser",
                "session_hijacker",
                "multi_device_rapid"
            ])
        else:
            self.anomaly_type = None


def generate_timestamp(base_time: datetime, max_offset_hours: int = 24) -> datetime:
    """Generate a random timestamp within offset from base."""
    offset = timedelta(
        hours=random.randint(0, max_offset_hours),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    return base_time + offset


def generate_normal_event(user: UserProfile, timestamp: datetime) -> dict:
    """Generate a normal (non-anomalous) event."""
    event_type = random.choice(EVENT_TYPES)

    base_event = {
        "event_id": str(uuid.uuid4()),
        "user_id": user.user_id,
        "timestamp": timestamp.isoformat(),
        "event_type": event_type,
        "ip_address": random.choice([user.primary_ip, user.secondary_ip]),
        "device": user.primary_device,
        "is_anomaly": False,
        "anomaly_type": None
    }

    # Add event-specific fields
    if event_type == "login":
        base_event["session_id"] = str(uuid.uuid4())
        base_event["login_success"] = random.random() > 0.02  # 98% success rate

    elif event_type == "logout":
        base_event["session_duration"] = int(np.random.normal(
            user.avg_session_duration,
            user.avg_session_duration * 0.3
        ))

    elif event_type == "deposit":
        base_event["amount"] = round(np.random.normal(user.avg_deposit, user.avg_deposit * 0.3), 2)
        base_event["currency"] = "USD"
        base_event["payment_method"] = random.choice(["bank_transfer", "credit_card", "crypto"])

    elif event_type == "withdrawal":
        base_event["amount"] = round(np.random.normal(user.avg_withdrawal, user.avg_withdrawal * 0.3), 2)
        base_event["currency"] = "USD"
        base_event["payment_method"] = random.choice(["bank_transfer", "crypto"])

    elif event_type == "trade":
        base_event["instrument"] = random.choice(user.preferred_instruments)
        base_event["lot_size"] = round(np.random.normal(user.avg_lot_size, user.lot_size_std), 2)
        base_event["direction"] = random.choice(["buy", "sell"])
        base_event["margin_used"] = round(base_event["lot_size"] * random.uniform(100, 500), 2)
        base_event["leverage"] = random.choice([10, 20, 50, 100, 200])

    elif event_type == "kyc_change":
        base_event["field_changed"] = random.choice(["address", "phone", "email", "id_document"])

    elif event_type == "password_change":
        base_event["change_method"] = random.choice(["user_initiated", "reset_link"])

    return base_event


def generate_anomalous_event(user: UserProfile, timestamp: datetime) -> dict:
    """Generate an anomalous event based on user's anomaly type."""
    event = generate_normal_event(user, timestamp)
    event["is_anomaly"] = True
    event["anomaly_type"] = user.anomaly_type

    if user.anomaly_type == "ip_switcher":
        # Rapid IP changes, often using VPN/proxy IPs
        event["ip_address"] = random.choice(VPN_IP_POOL + SUSPICIOUS_IP_POOL)
        event["device"] = random.choice(DEVICES)  # Device also changes

    elif user.anomaly_type == "trade_spiker":
        # Sudden massive trade volumes
        event["event_type"] = "trade"
        event["instrument"] = random.choice(INSTRUMENTS)
        event["lot_size"] = round(user.avg_lot_size * random.uniform(5, 20), 2)  # 5-20x normal
        event["direction"] = random.choice(["buy", "sell"])
        event["margin_used"] = round(event["lot_size"] * random.uniform(100, 500), 2)
        event["leverage"] = random.choice([100, 200, 500])  # Higher leverage

    elif user.anomaly_type == "deposit_withdrawal_abuser":
        # Rapid deposit-withdrawal cycles (money laundering pattern)
        event["event_type"] = random.choice(["deposit", "withdrawal"])
        event["amount"] = round(random.uniform(10000, 100000), 2)  # Unusually large amounts
        event["currency"] = "USD"
        event["payment_method"] = "crypto"  # Prefer crypto for anonymity

    elif user.anomaly_type == "session_hijacker":
        # Multiple concurrent sessions, impossible travel
        event["ip_address"] = random.choice(SUSPICIOUS_IP_POOL)
        event["device"] = random.choice(DEVICES)
        # Session from different geography within minutes
        event["geo_anomaly"] = True

    elif user.anomaly_type == "multi_device_rapid":
        # Rapid device switching (potential account sharing)
        event["device"] = random.choice([d for d in DEVICES if d != user.primary_device])
        event["ip_address"] = random.choice(NORMAL_IP_POOL + SUSPICIOUS_IP_POOL)

    return event


def generate_dataset(
    num_users: int = NUM_USERS,
    num_events: int = NUM_EVENTS,
    anomaly_rate: float = ANOMALY_RATE,
    start_date: Optional[datetime] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic forex brokerage dataset.

    Args:
        num_users: Number of unique users
        num_events: Total number of events to generate
        anomaly_rate: Fraction of events that are anomalous
        start_date: Starting timestamp for events
        seed: Random seed for reproducibility

    Returns:
        DataFrame with all generated events
    """
    random.seed(seed)
    np.random.seed(seed)

    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)

    logger.info(f"Generating {num_events} events for {num_users} users...")

    # Create user profiles
    num_anomalous_users = int(num_users * anomaly_rate * 2)  # More anomalous users to spread events
    users = []

    for i in range(num_users):
        user_id = f"USER_{str(i).zfill(5)}"
        is_anomalous = i < num_anomalous_users
        users.append(UserProfile(user_id, is_anomalous))

    logger.info(f"Created {num_anomalous_users} anomalous user profiles")

    # Generate events
    events = []
    anomalous_events = 0
    target_anomalous = int(num_events * anomaly_rate)

    for i in range(num_events):
        # Pick a user
        if anomalous_events < target_anomalous and random.random() < anomaly_rate * 1.5:
            # Pick an anomalous user
            user = random.choice([u for u in users if u.is_anomalous])
            timestamp = generate_timestamp(start_date, max_offset_hours=24 * 90)
            event = generate_anomalous_event(user, timestamp)
            anomalous_events += 1
        else:
            # Pick any user for normal event
            user = random.choice(users)
            timestamp = generate_timestamp(start_date, max_offset_hours=24 * 90)
            event = generate_normal_event(user, timestamp)

        events.append(event)

        if (i + 1) % 10000 == 0:
            logger.info(f"Generated {i + 1}/{num_events} events...")

    # Convert to DataFrame
    df = pd.DataFrame(events)

    # Sort by timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Dataset generated: {len(df)} events, {df['is_anomaly'].sum()} anomalies ({df['is_anomaly'].mean()*100:.2f}%)")

    return df


def save_dataset(df: pd.DataFrame, output_dir: str = "data") -> tuple[str, str]:
    """Save dataset to CSV and sample to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save full CSV
    csv_path = output_path / "synthetic_events.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")

    # Save sample JSON (first 100 events)
    json_path = output_path / "sample_events.json"
    sample = df.head(100).to_dict(orient="records")

    # Convert timestamps to strings for JSON
    for event in sample:
        if isinstance(event.get("timestamp"), pd.Timestamp):
            event["timestamp"] = event["timestamp"].isoformat()

    with open(json_path, "w") as f:
        json.dump(sample, f, indent=2, default=str)
    logger.info(f"Saved sample JSON to {json_path}")

    return str(csv_path), str(json_path)


def get_anomaly_statistics(df: pd.DataFrame) -> dict:
    """Get statistics about anomalies in the dataset."""
    stats = {
        "total_events": len(df),
        "total_anomalies": int(df["is_anomaly"].sum()),
        "anomaly_rate": float(df["is_anomaly"].mean()),
        "unique_users": df["user_id"].nunique(),
        "anomalous_users": df[df["is_anomaly"]]["user_id"].nunique(),
        "event_type_distribution": df["event_type"].value_counts().to_dict(),
        "anomaly_type_distribution": df[df["is_anomaly"]]["anomaly_type"].value_counts().to_dict(),
        "date_range": {
            "start": df["timestamp"].min().isoformat(),
            "end": df["timestamp"].max().isoformat()
        }
    }
    return stats


if __name__ == "__main__":
    # Generate and save dataset
    df = generate_dataset()
    csv_path, json_path = save_dataset(df)

    # Print statistics
    stats = get_anomaly_statistics(df)
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total Events: {stats['total_events']:,}")
    print(f"Total Anomalies: {stats['total_anomalies']:,} ({stats['anomaly_rate']*100:.2f}%)")
    print(f"Unique Users: {stats['unique_users']}")
    print(f"Anomalous Users: {stats['anomalous_users']}")
    print(f"\nEvent Type Distribution:")
    for event_type, count in stats["event_type_distribution"].items():
        print(f"  {event_type}: {count:,}")
    print(f"\nAnomaly Type Distribution:")
    for anomaly_type, count in stats["anomaly_type_distribution"].items():
        print(f"  {anomaly_type}: {count:,}")
    print(f"\nDate Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
