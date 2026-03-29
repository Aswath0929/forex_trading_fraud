#!/usr/bin/env python3
"""
Warmup and Test Script for ForexGuard
Addresses cold-start problem by pre-loading user history before testing.
"""

import random
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_USER_ID = "USER_TEST_001"

# Normal user profile (consistent with training data)
NORMAL_IP = "192.168.1.100"
NORMAL_IP_SECONDARY = "192.168.1.101"
NORMAL_DEVICE = "Windows Desktop"
NORMAL_LOT_SIZE_RANGE = (0.1, 2.0)
NORMAL_LEVERAGE_OPTIONS = [10, 20, 50, 100, 200]
INSTRUMENTS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]


def generate_warmup_events(num_events: int = 25) -> list[dict]:
    """
    Generate realistic normal events for warmup, spread across last 24 hours.
    """
    events = []
    now = datetime.now(timezone.utc)

    # Spread events across last 24 hours
    time_slots = sorted([
        now - timedelta(hours=random.uniform(0.5, 23.5))
        for _ in range(num_events)
    ])

    for i, timestamp in enumerate(time_slots):
        
        event_type = random.choices(
            ["login", "trade", "deposit"],
             weights=[0.15, 0.75, 0.10]
        )[0]

        event = {
            "event_id": f"warmup_{i+1:03d}",
            "user_id": TEST_USER_ID,
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "ip_address": random.choice([NORMAL_IP, NORMAL_IP, NORMAL_IP_SECONDARY]),  # 67% primary IP
            "device": NORMAL_DEVICE,
        }

        # if event_type == "login":
        #     event["login_success"] = True
        #     event["session_duration"] = random.randint(300, 3600)
        if event_type == "login":
            event["login_success"] = True
            # No session_duration here - that belongs on logout events

        elif event_type == "trade":
            lot_size = round(random.uniform(*NORMAL_LOT_SIZE_RANGE), 2)
            event["instrument"] = random.choice(INSTRUMENTS)
            event["lot_size"] = lot_size
            event["direction"] = random.choice(["buy", "sell"])
            event["margin_used"] = round(lot_size * random.uniform(100, 500), 2)
            event["leverage"] = random.choice(NORMAL_LEVERAGE_OPTIONS)

        elif event_type == "deposit":
            event["amount"] = round(random.uniform(500, 5000), 2)
            event["currency"] = "USD"
            event["payment_method"] = random.choice(["bank_transfer", "credit_card"])

        events.append(event)

    return events


def generate_test_events() -> list[dict]:
    """
    Generate 3 test cases: normal, borderline, and suspicious.
    """
    now = datetime.now(timezone.utc)

    # Test Case 1: Normal trade
    normal_trade = {
        "event_id": "test_normal_001",
        "user_id": TEST_USER_ID,
        "timestamp": now.isoformat(),
        "event_type": "trade",
        "ip_address": NORMAL_IP,
        "device": NORMAL_DEVICE,
        "instrument": "EUR/USD",
        "lot_size": 0.5,  # Normal lot size
        "direction": "buy",
        "margin_used": 250.0,
        "leverage": 100,
    }

    # Test Case 2: Borderline trade (slightly elevated)
    borderline_trade = {
        "event_id": "test_borderline_001",
        "user_id": TEST_USER_ID,
        "timestamp": (now + timedelta(seconds=30)).isoformat(),
        "event_type": "trade",
        "ip_address": NORMAL_IP,
        "device": NORMAL_DEVICE,
        "instrument": "GBP/USD",
        "lot_size": 3.5,  # Slightly above normal range
        "direction": "sell",
        "margin_used": 1400.0,
        "leverage": 200,
    }

    # Test Case 3: Suspicious trade (multiple red flags)
    suspicious_trade = {
        "event_id": "test_suspicious_001",
        "user_id": TEST_USER_ID,
        "timestamp": (now + timedelta(seconds=60)).isoformat(),
        "event_type": "trade",
        "ip_address": "10.0.100.55",  # Suspicious IP range
        "device": "iOS Mobile",  # Different device
        "instrument": "USD/JPY",
        "lot_size": 15.0,  # 10x normal (suspicious spike)
        "direction": "buy",
        "margin_used": 7500.0,  # High margin
        "leverage": 500,  # Extremely high leverage
    }

    return [
        ("Normal", normal_trade),
        ("Borderline", borderline_trade),
        ("Suspicious", suspicious_trade),
    ]


def score_event(event: dict) -> Optional[dict]:
    """Send event to /score endpoint and return response."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/score",
            json=event,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  [ERROR] {response.status_code}: {response.text[:100]}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] Cannot connect to {API_BASE_URL}")
        print("  Make sure the API is running: python -m api.main")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def check_health() -> bool:
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_summary_table(results: list[tuple]):
    """Print results as a formatted table."""
    print("\n" + "-" * 90)
    print(f"{'Event ID':<25} {'Score':>8} {'Anomaly':>10} {'Severity':>12} {'Top Feature':<30}")
    print("-" * 90)

    for label, event_id, score_data in results:
        if score_data:
            score = score_data.get("anomaly_score", 0)
            is_anomaly = "YES" if score_data.get("is_anomaly") else "no"
            severity = score_data.get("severity", "unknown")

            # Get top contributing feature
            contributions = score_data.get("feature_contributions", [])
            if contributions:
                top_feature = contributions[0].get("feature_name", "N/A")
            else:
                top_feature = "N/A"

            print(f"{event_id:<25} {score:>8.3f} {is_anomaly:>10} {severity:>12} {top_feature:<30}")
        else:
            print(f"{event_id:<25} {'ERROR':>8} {'-':>10} {'-':>12} {'-':<30}")

    print("-" * 90)


def main():
    print_header("ForexGuard Warmup & Test Script")
    print(f"API: {API_BASE_URL}")
    print(f"Test User: {TEST_USER_ID}")

    # Step 0: Health check
    print("\n[Step 0] Checking API health...")
    if not check_health():
        print("  [FAILED] API is not running!")
        print("\n  Start the API first:")
        print("    cd forexguard")
        print("    python -m api.main")
        return
    print("  [OK] API is healthy")

    # Step 1: Warmup - Pre-load user history
    print_header("Step 1: Warmup - Pre-loading User History")

    warmup_events = generate_warmup_events(num_events=25)
    print(f"Sending {len(warmup_events)} warmup events for {TEST_USER_ID}...")
    print(f"  Time range: last 24 hours")
    print(f"  Event types: login, trade, deposit")
    print(f"  Normal lot sizes: {NORMAL_LOT_SIZE_RANGE}")
    print()

    warmup_success = 0
    warmup_failed = 0

    for i, event in enumerate(warmup_events):
        result = score_event(event)
        if result:
            warmup_success += 1
            score = result.get("anomaly_score", 0)
            status = "ANOMALY" if result.get("is_anomaly") else "normal"
            # Only print every 5th event to reduce noise
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [{i+1:2d}/{len(warmup_events)}] {event['event_id']}: score={score:.3f} ({status})")
        else:
            warmup_failed += 1
            if warmup_failed == 1:
                print("  Stopping warmup due to errors...")
                break

    print(f"\n  Warmup complete: {warmup_success} succeeded, {warmup_failed} failed")

    if warmup_failed > 0:
        print("  [WARNING] Some warmup events failed. Results may be affected.")

    # Step 2: Test - Score test cases
    print_header("Step 2: Testing - Scoring Test Cases")

    test_cases = generate_test_events()
    print(f"Scoring {len(test_cases)} test events...\n")

    results = []

    for label, event in test_cases:
        print(f"Test Case: {label}")
        print(f"  Event ID: {event['event_id']}")
        print(f"  Lot Size: {event.get('lot_size', 'N/A')}")
        print(f"  IP: {event['ip_address']}")
        print(f"  Device: {event['device']}")
        print(f"  Leverage: {event.get('leverage', 'N/A')}")

        result = score_event(event)

        if result:
            score = result.get("anomaly_score", 0)
            is_anomaly = result.get("is_anomaly", False)
            severity = result.get("severity", "unknown")
            explanation = result.get("explanation", "")

            print(f"  --> Score: {score:.3f}")
            print(f"  --> Anomaly: {'YES' if is_anomaly else 'No'}")
            print(f"  --> Severity: {severity}")
            print(f"  --> Explanation: {explanation[:80]}...")

            results.append((label, event['event_id'], result))
        else:
            results.append((label, event['event_id'], None))

        print()

    # Step 3: Summary
    print_header("Step 3: Summary")
    print_summary_table(results)

    # Analysis
    print("\nAnalysis:")
    for label, event_id, score_data in results:
        if score_data:
            score = score_data.get("anomaly_score", 0)
            is_anomaly = score_data.get("is_anomaly", False)

            if label == "Normal":
                if is_anomaly:
                    print(f"  [!] {label}: Expected normal, got ANOMALY (score: {score:.3f})")
                    print("      Cold-start issue may still be present.")
                else:
                    print(f"  [OK] {label}: Correctly identified as normal (score: {score:.3f})")

            elif label == "Borderline":
                print(f"  [?] {label}: Score {score:.3f} ({'ANOMALY' if is_anomaly else 'normal'})")
                print("      Borderline cases are expected to be near threshold.")

            elif label == "Suspicious":
                if is_anomaly:
                    print(f"  [OK] {label}: Correctly flagged as ANOMALY (score: {score:.3f})")
                else:
                    print(f"  [!] {label}: Expected anomaly, got normal (score: {score:.3f})")
                    print("      Model may need retuning.")

    print("\n" + "=" * 70)
    print("  Test Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
