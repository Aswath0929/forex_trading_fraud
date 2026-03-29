"""
Synthetic Dataset Generator for ForexGuard
Generates ~50,000 events with realistic per-user sequential histories.

Key fixes over v1:
- Events are generated sequentially per user (not randomly scattered)
- Each user builds up real rolling window history before anomalies appear
- Value ranges are tighter and more realistic for Isolation Forest training
- Anomalous users exhibit gradual behavioral drift, not just single outlier events
- Training distribution will match real inference-time feature distributions
"""

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# ── Constants ────────────────────────────────────────────────────────────────

NUM_USERS        = 500
NUM_EVENTS       = 50_000
ANOMALY_RATE     = 0.05   # 5 % of events are anomalous
SEED             = 42

# Each user gets at least this many warm-up events before any anomaly can fire
MIN_HISTORY_EVENTS = 15

# IP pools
NORMAL_IP_POOL     = [f"192.168.{i}.{j}" for i in range(1, 10)   for j in range(1, 255)]
SUSPICIOUS_IP_POOL = [f"10.0.{i}.{j}"    for i in range(100, 110) for j in range(1, 255)]
VPN_IP_POOL        = [f"185.{i}.{j}.{k}" for i in range(100, 105)
                                          for j in range(1,  50)
                                          for k in range(1,  50)]

DEVICES      = ["Windows Desktop", "MacOS Desktop", "iOS Mobile", "Android Mobile", "Linux Desktop"]
INSTRUMENTS  = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "EUR/GBP", "USD/CAD", "NZD/USD"]

# Client portal pages (for navigation / bot-like behavior)
PAGES = [
    "/login",
    "/dashboard",
    "/profile",
    "/kyc",
    "/documents",
    "/deposit",
    "/withdrawal",
    "/history",
    "/support",
    "/settings",
]

# Event taxonomy (portal + trading)
EVENT_TYPES  = [
    "login", "logout",
    "page_view", "profile_update", "support_ticket", "document_upload",
    "kyc_change", "password_change",
    "deposit", "withdrawal", "trade",
]

TICKET_CATEGORIES = ["kyc", "deposit", "withdrawal", "trading", "technical"]
TICKET_PRIORITIES = ["low", "medium", "high", "urgent"]
DOCUMENT_TYPES = ["id_document", "proof_of_address", "selfie", "bank_statement"]
LEVERAGE_OPTS = [10, 20, 50, 100, 200]


# ── User Profile ─────────────────────────────────────────────────────────────

class UserProfile:
    """
    Realistic user with stable behavioral fingerprint.
    Anomalous users start normal then drift into suspicious patterns
    after MIN_HISTORY_EVENTS to ensure the feature store is populated.
    """

    def __init__(self, user_id: str, is_anomalous: bool = False):
        self.user_id      = user_id
        self.is_anomalous = is_anomalous
        self.event_count  = 0           # tracks how many events this user has had

        # ── Stable identity ──
        # Some users share "hub" IPs (cafes, corporate NATs, fraud farms)
        hub_ip_pool = ["203.0.113.10", "203.0.113.11", "203.0.113.12", "198.51.100.20"]
        if random.random() < 0.05:
            self.primary_ip = random.choice(hub_ip_pool)
        else:
            self.primary_ip = random.choice(NORMAL_IP_POOL)

        self.secondary_ip  = random.choice(NORMAL_IP_POOL)
        self.primary_device = random.choice(DEVICES)
        self.preferred_instruments = random.sample(INSTRUMENTS, k=random.randint(2, 4))

        # ── Normal trading behaviour ──
        # Tighter ranges so the model learns a real "normal" band
        self.avg_lot_size   = random.uniform(0.1, 2.0)
        self.lot_size_std   = self.avg_lot_size * 0.15   # 15 % std — less noisy than before
        self.avg_leverage   = random.choice(LEVERAGE_OPTS)

        # margin_used is derived: lot_size * leverage * pip_value (simplified)
        # We keep it proportional so the model sees consistent relationships
        self.margin_multiplier = random.uniform(80, 120)  # ~100 × lot_size on average

        self.avg_trades_per_day    = random.randint(2, 15)
        self.avg_session_duration  = random.randint(300, 5400)   # 5 min – 1.5 h

        self.avg_deposit    = random.uniform(500, 8000)
        self.avg_withdrawal = random.uniform(200, 4000)

        # ── Anomaly configuration ──
        if is_anomalous:
            self.anomaly_type = random.choice([
                # access / device
                "ip_switcher",
                "session_hijacker",
                "multi_device_rapid",
                "failed_login_bruteforce",

                # portal behavior
                "bot_navigator",
                "doc_upload_spammer",
                "kyc_rush_withdrawal",

                # trading / financial
                "trade_spiker",
                "deposit_withdrawal_abuser",
            ])
        else:
            self.anomaly_type = None

    # ── helpers ──

    def next_timestamp(self, last_ts: datetime) -> datetime:
        """
        Return next event timestamp for this user.
        Trades cluster during market hours; other events are more spread out.
        """
        # Inter-event gap: Poisson-ish, mean gap depends on activity level
        mean_gap_minutes = max(5, int(60 * 24 / max(self.avg_trades_per_day * 3, 1)))
        gap_seconds = int(np.random.exponential(mean_gap_minutes * 60))
        gap_seconds = max(30, min(gap_seconds, 8 * 3600))  # clamp 30 s – 8 h
        return last_ts + timedelta(seconds=gap_seconds)

    def should_be_anomalous(self) -> bool:
        """Only fire anomaly after enough history exists."""
        return self.is_anomalous and self.event_count >= MIN_HISTORY_EVENTS


# ── Event Generators ──────────────────────────────────────────────────────────

def _base_event(user: UserProfile, timestamp: datetime, event_type: str) -> dict:
    return {
        "event_id":   str(uuid.uuid4()),
        "user_id":    user.user_id,
        "timestamp":  timestamp.isoformat(),
        "event_type": event_type,
        "ip_address": random.choice([user.primary_ip, user.secondary_ip]),
        "device":     user.primary_device,
        "is_anomaly": False,
        "anomaly_type": None,
    }


def generate_normal_event(user: UserProfile, timestamp: datetime) -> dict:
    """Generate one realistic normal event for a user."""

    # Weighted selection to better match typical portal/trading behavior
    event_type = random.choices(
        EVENT_TYPES,
        weights=[
            0.08,  # login
            0.06,  # logout
            0.40,  # page_view
            0.05,  # profile_update
            0.03,  # support_ticket
            0.05,  # document_upload
            0.03,  # kyc_change
            0.02,  # password_change
            0.08,  # deposit
            0.05,  # withdrawal
            0.15,  # trade
        ],
        k=1,
    )[0]

    event = _base_event(user, timestamp, event_type)

    if event_type == "login":
        event["session_id"]    = str(uuid.uuid4())
        event["login_success"] = random.random() > 0.02

    elif event_type == "logout":
        event["session_duration"] = int(np.random.normal(
            user.avg_session_duration, user.avg_session_duration * 0.25
        ))

    elif event_type == "deposit":
        event["amount"]         = round(abs(np.random.normal(user.avg_deposit,    user.avg_deposit    * 0.25)), 2)
        event["currency"]       = "USD"
        event["payment_method"] = random.choice(["bank_transfer", "credit_card", "crypto"])

    elif event_type == "withdrawal":
        event["amount"]         = round(abs(np.random.normal(user.avg_withdrawal, user.avg_withdrawal * 0.25)), 2)
        event["currency"]       = "USD"
        event["payment_method"] = random.choice(["bank_transfer", "crypto"])

    elif event_type == "trade":
        lot = round(abs(np.random.normal(user.avg_lot_size, user.lot_size_std)), 2)
        lot = max(0.01, lot)
        event["instrument"]  = random.choice(user.preferred_instruments)
        event["lot_size"]    = lot
        event["direction"]   = random.choice(["buy", "sell"])
        event["margin_used"] = round(lot * user.margin_multiplier, 2)
        event["leverage"]    = user.avg_leverage

    elif event_type == "kyc_change":
        event["field_changed"] = random.choice(["address", "phone", "email", "id_document", "pep_status"])

    elif event_type == "password_change":
        event["change_method"] = random.choice(["user_initiated", "reset_link"])

    elif event_type == "profile_update":
        event["field_changed"] = random.choice(["address", "phone", "email", "bank_account"])  # reuse field_changed

    elif event_type == "support_ticket":
        event["ticket_category"] = random.choice(TICKET_CATEGORIES)
        event["ticket_priority"] = random.choices(TICKET_PRIORITIES, weights=[0.55, 0.30, 0.12, 0.03], k=1)[0]

    elif event_type == "document_upload":
        event["document_type"] = random.choice(DOCUMENT_TYPES)
        event["upload_success"] = random.random() > 0.05

    elif event_type == "page_view":
        event["page"] = random.choice(PAGES)
        event["action"] = random.choices(["view", "click", "submit"], weights=[0.8, 0.18, 0.02], k=1)[0]

    return event


def generate_anomalous_event(user: UserProfile, timestamp: datetime) -> dict:
    """
    Generate an anomalous event.
    Builds on top of a normal event so base features are always present,
    then applies the suspicious mutation.
    """
    event = generate_normal_event(user, timestamp)
    event["is_anomaly"]   = True
    event["anomaly_type"] = user.anomaly_type

    if user.anomaly_type == "ip_switcher":
        # Rotate through VPN / suspicious IPs rapidly
        event["ip_address"] = random.choice(VPN_IP_POOL + SUSPICIOUS_IP_POOL)
        event["device"]     = random.choice(DEVICES)   # device also changes

    elif user.anomaly_type == "trade_spiker":
        # Sudden massive lot size — 5-15× the user's normal
        event["event_type"]  = "trade"
        lot = round(user.avg_lot_size * random.uniform(5, 15), 2)
        event["lot_size"]    = lot
        event["direction"]   = random.choice(["buy", "sell"])
        event["margin_used"] = round(lot * user.margin_multiplier * random.uniform(1.5, 3), 2)
        event["leverage"]    = random.choice([200, 500])
        event["instrument"]  = random.choice(user.preferred_instruments)

    elif user.anomaly_type == "deposit_withdrawal_abuser":
        # Large rapid deposit→withdrawal cycle (money-laundering pattern)
        event["event_type"]     = random.choice(["deposit", "withdrawal"])
        event["amount"]         = round(random.uniform(15_000, 100_000), 2)
        event["currency"]       = "USD"
        event["payment_method"] = "crypto"

    elif user.anomaly_type == "session_hijacker":
        # Login from impossible location within minutes of last event
        event["ip_address"] = random.choice(SUSPICIOUS_IP_POOL)
        event["device"]     = random.choice([d for d in DEVICES if d != user.primary_device])
        event["geo_anomaly"] = True

    elif user.anomaly_type == "multi_device_rapid":
        # Different device every event + mixed IPs
        event["device"]     = random.choice([d for d in DEVICES if d != user.primary_device])
        event["ip_address"] = random.choice(NORMAL_IP_POOL + SUSPICIOUS_IP_POOL)

    elif user.anomaly_type == "failed_login_bruteforce":
        # Many failed logins; often at unusual hours
        event["event_type"] = "login"
        event["login_success"] = False
        event["ip_address"] = random.choice(VPN_IP_POOL + SUSPICIOUS_IP_POOL)
        event["device"] = random.choice(DEVICES)

    elif user.anomaly_type == "bot_navigator":
        # Bot-like rapid portal navigation
        event["event_type"] = "page_view"
        event["page"] = random.choice(PAGES + [f"/promo/{i}" for i in range(1, 20)])
        event["action"] = random.choice(["view", "click"])  # high volume, low diversity actions

    elif user.anomaly_type == "doc_upload_spammer":
        # Repeated uploads (often failing)
        event["event_type"] = "document_upload"
        event["document_type"] = random.choice(DOCUMENT_TYPES)
        event["upload_success"] = random.random() > 0.4
        event["page"] = "/documents"

    elif user.anomaly_type == "kyc_rush_withdrawal":
        # Rapid KYC/profile changes often preceding withdrawal activity
        event["event_type"] = random.choice(["kyc_change", "profile_update", "document_upload", "withdrawal"])
        if event["event_type"] in ("kyc_change", "profile_update"):
            event["field_changed"] = random.choice(["address", "phone", "email", "bank_account", "id_document"])
        if event["event_type"] == "document_upload":
            event["document_type"] = random.choice(DOCUMENT_TYPES)
            event["upload_success"] = True
            event["page"] = "/documents"
        if event["event_type"] == "withdrawal":
            event["amount"] = round(random.uniform(10_000, 80_000), 2)
            event["currency"] = "USD"
            event["payment_method"] = "crypto"

    return event


# ── Dataset Generator ─────────────────────────────────────────────────────────

def generate_dataset(
    num_users:    int            = NUM_USERS,
    num_events:   int            = NUM_EVENTS,
    anomaly_rate: float          = ANOMALY_RATE,
    start_date:   Optional[datetime] = None,
    seed:         int            = SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic forex dataset with proper per-user sequential histories.

    Strategy
    --------
    1. Assign each user a budget of events proportional to their activity level.
    2. Generate events user-by-user in chronological order so the rolling window
       store sees real history when features are later computed.
    3. Anomalous users only start misbehaving after MIN_HISTORY_EVENTS normal
       events, ensuring the feature pipeline has a baseline to deviate from.
    4. The final DataFrame is sorted by timestamp across all users.
    """
    random.seed(seed)
    np.random.seed(seed)

    if start_date is None:
        start_date = datetime.now(timezone.utc) - timedelta(days=90)

    logger.info(f"Generating {num_events:,} events for {num_users} users …")

    # ── Create user profiles ──
    num_anomalous = int(num_users * anomaly_rate * 2)   # ~10 % of users are anomalous
    users: list[UserProfile] = []
    for i in range(num_users):
        uid = f"USER_{str(i).zfill(5)}"
        users.append(UserProfile(uid, is_anomalous=(i < num_anomalous)))

    logger.info(f"  {num_anomalous} anomalous user profiles, {num_users - num_anomalous} normal")

    # ── Distribute event budget across users ──
    # Weight by avg_trades_per_day so active users get more events
    weights = np.array([u.avg_trades_per_day for u in users], dtype=float)
    weights /= weights.sum()
    event_counts = np.random.multinomial(num_events, weights)
    # Guarantee every user gets at least MIN_HISTORY_EVENTS + 5
    min_events = MIN_HISTORY_EVENTS + 5
    event_counts = np.maximum(event_counts, min_events)

    # ── Generate events per user ──
    all_events: list[dict] = []
    target_anomalous = int(num_events * anomaly_rate)
    total_anomalous  = 0

    # for user, n_events in zip(users, event_counts):
    #     # Random start time within the 90-day window (so users don't all start together)
    #     user_start = start_date + timedelta(
    #         hours=random.randint(0, 24 * 30)   # spread starts over first 30 days
    #     )
    #     ts = user_start

    #     for _ in range(int(n_events)):
    #         ts = user.next_timestamp(ts)

    #         # Decide: normal or anomalous?
    #         quota_remaining = target_anomalous - total_anomalous
    #         anomaly_budget  = quota_remaining / max(1, num_events - len(all_events))

    #         if user.should_be_anomalous() and random.random() < min(anomaly_budget * 3, 0.4):
    #             event = generate_anomalous_event(user, ts)
    #             total_anomalous += 1
    #         else:
    #             event = generate_normal_event(user, ts)

    #         user.event_count += 1
    #         all_events.append(event)
    for user, n_events in zip(users, event_counts):
        user_start = start_date + timedelta(
            hours=random.randint(0, 24 * 30)
        )
        ts = user_start

        for idx in range(int(n_events)):
            ts = user.next_timestamp(ts)

            # For anomalous users past warmup, fire anomaly at fixed 40% rate
            # This ensures we hit ~5% overall (10% anomalous users × 40% = 4% + buffer)
            if user.should_be_anomalous() and random.random() < 0.40:
                event = generate_anomalous_event(user, ts)
                total_anomalous += 1
            else:
                event = generate_normal_event(user, ts)

            user.event_count += 1
            all_events.append(event)

        if len(all_events) % 10_000 < int(n_events):
            logger.info(f"  {len(all_events):,} events generated so far …")

    # ── Build DataFrame ──
    df = pd.DataFrame(all_events)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    actual_anomaly_rate = df["is_anomaly"].mean()
    logger.info(
        f"Dataset ready: {len(df):,} events | "
        f"{df['is_anomaly'].sum():,} anomalies ({actual_anomaly_rate*100:.2f}%) | "
        f"{df['user_id'].nunique()} users"
    )

    return df


# ── Save / Stats ──────────────────────────────────────────────────────────────

def save_dataset(df: pd.DataFrame, output_dir: str = "data") -> tuple[str, str]:
    """Save dataset to CSV and a 100-event sample JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "synthetic_events.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV → {csv_path}")

    json_path = out / "sample_events.json"
    sample = df.head(100).to_dict(orient="records")
    for ev in sample:
        if isinstance(ev.get("timestamp"), pd.Timestamp):
            ev["timestamp"] = ev["timestamp"].isoformat()
    with open(json_path, "w") as f:
        json.dump(sample, f, indent=2, default=str)
    logger.info(f"Saved sample JSON → {json_path}")

    return str(csv_path), str(json_path)


def get_anomaly_statistics(df: pd.DataFrame) -> dict:
    return {
        "total_events":    len(df),
        "total_anomalies": int(df["is_anomaly"].sum()),
        "anomaly_rate":    float(df["is_anomaly"].mean()),
        "unique_users":    df["user_id"].nunique(),
        "anomalous_users": df[df["is_anomaly"]]["user_id"].nunique(),
        "event_type_distribution":  df["event_type"].value_counts().to_dict(),
        "anomaly_type_distribution": (
            df[df["is_anomaly"]]["anomaly_type"].value_counts().to_dict()
        ),
        "date_range": {
            "start": df["timestamp"].min().isoformat(),
            "end":   df["timestamp"].max().isoformat(),
        },
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_dataset()
    csv_path, json_path = save_dataset(df)

    stats = get_anomaly_statistics(df)
    print("\n" + "=" * 55)
    print("DATASET STATISTICS")
    print("=" * 55)
    print(f"Total Events    : {stats['total_events']:,}")
    print(f"Total Anomalies : {stats['total_anomalies']:,}  ({stats['anomaly_rate']*100:.2f}%)")
    print(f"Unique Users    : {stats['unique_users']}")
    print(f"Anomalous Users : {stats['anomalous_users']}")

    print("\nEvent Type Distribution:")
    for k, v in sorted(stats["event_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {k:<20} {v:,}")

    print("\nAnomaly Type Distribution:")
    for k, v in sorted(stats["anomaly_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {k:<30} {v:,}")

    print(f"\nDate Range: {stats['date_range']['start']}  →  {stats['date_range']['end']}")
    print(f"\nFiles saved:\n  {csv_path}\n  {json_path}")