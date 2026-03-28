#!/usr/bin/env python3
"""
ForexGuard Demo Script
Demonstrates the complete anomaly detection pipeline.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from loguru import logger

from data.generate_synthetic import generate_dataset, get_anomaly_statistics
from features.feature_pipeline import FeaturePipeline, create_training_features
from models.isolation_forest import IsolationForestDetector
from models.model_registry import get_registry
from explainability.explainer import AnomalyExplainer


def demo_training():
    """Demonstrate model training."""
    print("\n" + "="*60)
    print("STEP 1: DATA GENERATION & TRAINING")
    print("="*60)

    # Generate small dataset for demo
    print("\nGenerating synthetic dataset (5000 events)...")
    df = generate_dataset(num_users=100, num_events=5000, seed=42)

    stats = get_anomaly_statistics(df)
    print(f"  - Total events: {stats['total_events']}")
    print(f"  - Anomalies: {stats['total_anomalies']} ({stats['anomaly_rate']*100:.1f}%)")
    print(f"  - Users: {stats['unique_users']}")

    # Extract features
    print("\nExtracting features...")
    X, feature_names, feature_df = create_training_features(df)
    print(f"  - Feature matrix: {X.shape}")
    print(f"  - Features: {len(feature_names)}")

    # Train model
    print("\nTraining Isolation Forest...")
    detector = IsolationForestDetector(contamination=0.05, n_estimators=100)
    detector.fit(X, feature_names)

    # Evaluate
    y_true = df["is_anomaly"].values
    predictions = detector.predict(X)
    scores = detector.predict_proba(X)

    # Confusion matrix
    tp = ((predictions == -1) & (y_true == 1)).sum()
    fp = ((predictions == -1) & (y_true == 0)).sum()
    fn = ((predictions == 1) & (y_true == 1)).sum()
    tn = ((predictions == 1) & (y_true == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\nResults:")
    print(f"  - Precision: {precision:.1%}")
    print(f"  - Recall: {recall:.1%}")
    print(f"  - True Positives: {tp}")
    print(f"  - False Positives: {fp}")

    # Save model
    registry = get_registry("models/saved")
    version = registry.register(
        model=detector,
        model_name="demo",
        model_type="isolation_forest",
        metrics={"precision": precision, "recall": recall}
    )
    print(f"\nModel saved: isolation_forest/demo {version}")

    return detector, X, feature_names, df


def demo_scoring(detector, X, feature_names, df):
    """Demonstrate scoring and explanations."""
    print("\n" + "="*60)
    print("STEP 2: SCORING & EXPLANATIONS")
    print("="*60)

    # Initialize pipeline and explainer
    pipeline = FeaturePipeline()
    explainer = AnomalyExplainer(detector, pipeline, use_shap=False)

    # Find some anomalies and normal events
    anomaly_indices = df[df["is_anomaly"] == True].index[:3]
    normal_indices = df[df["is_anomaly"] == False].index[:2]

    print("\n--- Anomalous Events ---")
    for idx in anomaly_indices:
        event = df.iloc[idx].to_dict()
        score, contributions, explanation = explainer.explain(event)

        print(f"\nEvent: {event.get('event_id', idx)} | User: {event['user_id']}")
        print(f"  Type: {event['event_type']}")
        print(f"  Score: {score:.3f} {'[ANOMALY]' if score > 0.5 else ''}")
        print(f"  Explanation: {explanation}")

        if contributions:
            print("  Top contributors:")
            for name, value, contrib, desc in contributions[:3]:
                print(f"    - {name}: {value:.2f} (contribution: {contrib:.3f})")

    print("\n--- Normal Events ---")
    for idx in normal_indices:
        event = df.iloc[idx].to_dict()
        score, contributions, explanation = explainer.explain(event)

        print(f"\nEvent: {event.get('event_id', idx)} | User: {event['user_id']}")
        print(f"  Type: {event['event_type']}")
        print(f"  Score: {score:.3f}")
        print(f"  Explanation: {explanation}")


def demo_streaming():
    """Demonstrate streaming simulation."""
    print("\n" + "="*60)
    print("STEP 3: STREAMING SIMULATION")
    print("="*60)

    from streaming.simulator import StreamSimulator, StreamConfig

    config = StreamConfig(
        events_per_second=100,  # Fast for demo
        anomaly_rate=0.05,
        num_users=50,
        seed=123
    )

    simulator = StreamSimulator(config)

    print("\nGenerating 100 streaming events...")
    events = list(simulator.stream_sync(max_events=100))

    stats = simulator.get_stats()
    print(f"  - Total events: {stats['total_events']}")
    print(f"  - Anomalies: {stats['anomaly_count']} ({stats['anomaly_rate']*100:.1f}%)")

    # Show sample events
    print("\nSample events:")
    for i, event in enumerate(events[:3]):
        print(f"  {i+1}. User {event['user_id']} | {event['event_type']} | Anomaly: {event.get('is_anomaly', False)}")


def demo_alerts():
    """Demonstrate alert generation."""
    print("\n" + "="*60)
    print("STEP 4: ALERT GENERATION")
    print("="*60)

    from api.alerts import AlertGenerator
    from api.schemas import AnomalyScore, SeverityLevel, FeatureContribution

    generator = AlertGenerator(alert_threshold=0.5)

    # Create sample anomaly scores
    sample_scores = [
        AnomalyScore(
            event_id="evt_001",
            user_id="USER_00001",
            anomaly_score=0.92,
            is_anomaly=True,
            threshold=0.5,
            model_type="isolation_forest",
            severity=SeverityLevel.CRITICAL,
            feature_contributions=[
                FeatureContribution(
                    feature_name="ip_change_rate",
                    value=5.2,
                    contribution=0.4,
                    description="Rapid IP changes"
                ),
                FeatureContribution(
                    feature_name="lot_size_zscore",
                    value=3.1,
                    contribution=0.3,
                    description="Unusual lot sizes"
                )
            ],
            explanation="Critical anomaly detected due to rapid IP changes and unusual lot sizes."
        ),
        AnomalyScore(
            event_id="evt_002",
            user_id="USER_00025",
            anomaly_score=0.78,
            is_anomaly=True,
            threshold=0.5,
            model_type="isolation_forest",
            severity=SeverityLevel.HIGH,
            feature_contributions=[
                FeatureContribution(
                    feature_name="amount_zscore",
                    value=4.5,
                    contribution=0.5,
                    description="Unusual withdrawal amount"
                )
            ],
            explanation="High-risk anomaly detected due to unusual withdrawal amount."
        )
    ]

    print("\nGenerating alerts from anomaly scores...")
    for score in sample_scores:
        alert = generator.generate_alert(score)
        if alert:
            print(f"\n[{alert.severity.value.upper()}] {alert.title}")
            print(f"  User: {alert.user_id}")
            print(f"  Score: {alert.anomaly_score:.2f}")
            print(f"  Description: {alert.description[:100]}...")
            print(f"  Action: {alert.recommended_action}")

    summary = generator.get_alert_summary()
    print(f"\nAlert Summary:")
    print(f"  - Total: {summary['total']}")
    print(f"  - By severity: {summary['by_severity']}")


def demo_api_usage():
    """Show example API usage."""
    print("\n" + "="*60)
    print("STEP 5: API USAGE EXAMPLES")
    print("="*60)

    print("""
To start the API server:

    python -m api.main

Or with uvicorn:

    uvicorn api.main:app --reload --port 8000

Example API calls:

1. Health Check:
   curl http://localhost:8000/health

2. Score Single Event:
   curl -X POST http://localhost:8000/score \\
     -H "Content-Type: application/json" \\
     -d '{
       "event_id": "evt_001",
       "user_id": "USER_00001",
       "timestamp": "2024-01-15T10:30:00Z",
       "event_type": "trade",
       "ip_address": "192.168.1.100",
       "device": "Windows Desktop",
       "lot_size": 10.0
     }'

3. Get Alerts:
   curl http://localhost:8000/alerts?limit=10

4. Interactive Docs:
   http://localhost:8000/docs
""")


def main():
    """Run full demo."""
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*15 + "ForexGuard Demo" + " "*27 + "║")
    print("║" + " "*10 + "Real-Time Anomaly Detection" + " "*20 + "║")
    print("╚" + "═"*58 + "╝")

    try:
        # Step 1: Train
        detector, X, feature_names, df = demo_training()

        # Step 2: Score
        demo_scoring(detector, X, feature_names, df)

        # Step 3: Streaming
        demo_streaming()

        # Step 4: Alerts
        demo_alerts()

        # Step 5: API Usage
        demo_api_usage()

        print("\n" + "="*60)
        print("DEMO COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run full training: python train.py")
        print("  2. Start API: python -m api.main")
        print("  3. Try the docs: http://localhost:8000/docs")
        print("")

    except Exception as e:
        logger.error(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
