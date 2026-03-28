#!/usr/bin/env python3
"""
Training Script for ForexGuard
Generates synthetic data, trains anomaly detection models, and saves them to the registry.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.generate_synthetic import generate_dataset, get_anomaly_statistics
from features.feature_pipeline import FeaturePipeline, create_training_features
from models.isolation_forest import IsolationForestDetector
from models.lstm_autoencoder import LSTMAutoencoderDetector
from models.model_registry import get_registry


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    """Evaluate model performance."""
    # Convert predictions to binary (1 = anomaly)
    if y_pred.min() == -1:
        # Isolation Forest format: -1 = anomaly, 1 = normal
        y_pred_binary = (y_pred == -1).astype(int)
    else:
        y_pred_binary = y_pred

    # Calculate metrics
    tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
    fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
    fn = ((y_pred_binary == 0) & (y_true == 1)).sum()
    tn = ((y_pred_binary == 0) & (y_true == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true)

    # Calculate AUC-like metric using anomaly scores
    # Higher scores should correspond to anomalies
    anomaly_scores_positive = scores[y_true == 1]
    anomaly_scores_negative = scores[y_true == 0]

    if len(anomaly_scores_positive) > 0 and len(anomaly_scores_negative) > 0:
        # Approximate AUC: proportion of anomalies with higher scores than normal
        auc_approx = np.mean([
            np.mean(anomaly_scores_positive > threshold)
            for threshold in np.percentile(anomaly_scores_negative, [25, 50, 75])
        ])
    else:
        auc_approx = 0.5

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "auc_approx": round(auc_approx, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }


def train_isolation_forest(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    contamination: float = 0.05
) -> tuple[IsolationForestDetector, dict]:
    """Train Isolation Forest model."""
    logger.info("Training Isolation Forest model...")

    detector = IsolationForestDetector(
        contamination=contamination,
        n_estimators=200,
        random_state=42
    )

    detector.fit(X, feature_names)

    # Evaluate
    predictions = detector.predict(X)
    scores = detector.predict_proba(X)

    metrics = evaluate_model(y, predictions, scores)
    logger.info(f"Isolation Forest - Precision: {metrics['precision']:.2%}, Recall: {metrics['recall']:.2%}, F1: {metrics['f1_score']:.2%}")

    return detector, metrics


def train_lstm_autoencoder(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    seq_len: int = 10,
    epochs: int = 30
) -> tuple[LSTMAutoencoderDetector, dict]:
    """Train LSTM Autoencoder model."""
    logger.info("Training LSTM Autoencoder model...")

    detector = LSTMAutoencoderDetector(
        input_dim=X.shape[1],
        seq_len=seq_len,
        hidden_dim=64,
        latent_dim=32,
        threshold_percentile=95.0
    )

    # Train on normal data only (unsupervised)
    X_normal = X[y == 0]
    detector.fit(X_normal, feature_names=feature_names, epochs=epochs)

    # Evaluate on all data
    scores = detector.score_samples(X)
    predictions = detector.predict(X)

    # Adjust y to match scores length (LSTM produces fewer outputs due to sequencing)
    y_adjusted = y[:len(scores)]

    metrics = evaluate_model(y_adjusted, predictions, scores)
    logger.info(f"LSTM Autoencoder - Precision: {metrics['precision']:.2%}, Recall: {metrics['recall']:.2%}, F1: {metrics['f1_score']:.2%}")

    return detector, metrics


def main():
    parser = argparse.ArgumentParser(description="Train ForexGuard anomaly detection models")
    parser.add_argument("--num-events", type=int, default=50000, help="Number of events to generate")
    parser.add_argument("--num-users", type=int, default=500, help="Number of users")
    parser.add_argument("--anomaly-rate", type=float, default=0.05, help="Anomaly rate (0-1)")
    parser.add_argument("--model", choices=["isolation_forest", "lstm", "both"], default="both", help="Model to train")
    parser.add_argument("--epochs", type=int, default=30, help="LSTM training epochs")
    parser.add_argument("--output-dir", type=str, default="models/saved", help="Output directory for models")
    parser.add_argument("--save-data", action="store_true", help="Save generated dataset to CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("ForexGuard Model Training")
    logger.info("="*60)

    # Generate synthetic data
    logger.info(f"\nGenerating synthetic dataset with {args.num_events} events...")
    df = generate_dataset(
        num_users=args.num_users,
        num_events=args.num_events,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed
    )

    # Print statistics
    stats = get_anomaly_statistics(df)
    logger.info(f"Dataset: {stats['total_events']} events, {stats['total_anomalies']} anomalies ({stats['anomaly_rate']*100:.2f}%)")

    # Save data if requested
    if args.save_data:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        csv_path = data_dir / "synthetic_events.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved dataset to {csv_path}")

    # Extract features
    logger.info("\nExtracting features...")
    X, feature_names, feature_df = create_training_features(df)

    # Get labels
    y = df["is_anomaly"].astype(int).values

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of features: {len(feature_names)}")

    # Initialize registry
    registry = get_registry(args.output_dir)

    # Train Isolation Forest
    if args.model in ["isolation_forest", "both"]:
        logger.info("\n" + "-"*40)
        if_detector, if_metrics = train_isolation_forest(
            X, y, feature_names,
            contamination=args.anomaly_rate
        )

        # Register model
        version = registry.register(
            model=if_detector,
            model_name="default",
            model_type="isolation_forest",
            description=f"Trained on {args.num_events} events with {args.anomaly_rate*100:.0f}% anomaly rate",
            metrics=if_metrics,
            set_active=True
        )
        logger.info(f"Registered Isolation Forest model version: {version}")

    # Train LSTM Autoencoder
    if args.model in ["lstm", "both"]:
        logger.info("\n" + "-"*40)
        lstm_detector, lstm_metrics = train_lstm_autoencoder(
            X, y, feature_names,
            epochs=args.epochs
        )

        # Register model
        version = registry.register(
            model=lstm_detector,
            model_name="default",
            model_type="lstm_autoencoder",
            description=f"Trained on {args.num_events} events with {args.epochs} epochs",
            metrics=lstm_metrics,
            set_active=(args.model == "lstm")  # Only set active if LSTM is sole model
        )
        logger.info(f"Registered LSTM Autoencoder model version: {version}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)

    models = registry.list_models()
    for m in models:
        status = "[ACTIVE]" if m["is_active"] else ""
        logger.info(f"  - {m['type']}/{m['name']} v{m['latest_version']} {status}")

    logger.info(f"\nModels saved to: {args.output_dir}")
    logger.info("Run the API with: python -m api.main")


if __name__ == "__main__":
    main()
