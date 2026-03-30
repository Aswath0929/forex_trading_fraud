"""
Isolation Forest Model for ForexGuard
Baseline unsupervised anomaly detection using sklearn's Isolation Forest.

WHY ISOLATION FOREST:
1. No labeled data required - perfect for unsupervised anomaly detection
2. Efficient on high-dimensional data - O(n log n) complexity
3. Handles mixed feature types well
4. Provides interpretable anomaly scores (-1 to 1)
5. Works well with the 5% anomaly rate in our dataset
6. Fast inference - suitable for real-time scoring
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Optional
import joblib
from pathlib import Path
from loguru import logger


class IsolationForestDetector:
    """
    Isolation Forest based anomaly detector.
    Detects anomalies by isolating observations through random partitioning.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        max_samples: str = "auto",
        random_state: int = 42
    ):
        """
        Initialize the detector.

        Args:
            contamination: Expected proportion of anomalies (0.05 = 5%)
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
            warm_start=False
        )
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted = False

        # Calibration stats (learned at fit time) for mapping raw scores -> 0..1 anomaly score
        # IsolationForest.score_samples: lower = more anomalous
        self.raw_score_mean: Optional[float] = None
        self.raw_score_std: Optional[float] = None
        self.raw_score_threshold: Optional[float] = None
        self.calibration_scale: float = 5.0

    def fit(self, X: np.ndarray, feature_names: Optional[list[str]] = None):
        """
        Fit the model on training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Names of features for explainability
        """
        logger.info(f"Fitting Isolation Forest on {X.shape[0]} samples with {X.shape[1]} features")

        # Store feature names
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Handle NaN and infinite values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit isolation forest
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Learn calibration stats from training distribution so that
        # anomaly_score ~= 0.5 around the model's contamination cut.
        raw = self.model.score_samples(X_scaled)
        self.raw_score_mean = float(np.mean(raw))
        self.raw_score_std = float(np.std(raw)) if float(np.std(raw)) > 0 else 1.0
        self.raw_score_threshold = float(np.quantile(raw, self.contamination))

        logger.info(
            "Isolation Forest training complete | raw_mean={:.4f} raw_std={:.4f} raw_thresh(q={:.3f})={:.4f}",
            self.raw_score_mean,
            self.raw_score_std,
            self.contamination,
            self.raw_score_threshold,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Args:
            X: Feature matrix

        Returns:
            Array of labels: 1 for normal, -1 for anomaly
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for samples.

        Args:
            X: Feature matrix

        Returns:
            Array of anomaly scores. Lower (more negative) = more anomalous.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X)

        # score_samples returns negative scores for anomalies
        return self.model.score_samples(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly probability scores (0 to 1).

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities where higher = more likely anomaly
        """
        # Get raw scores: lower = more anomalous
        raw_scores = self.score_samples(X)

        # If we have calibration stats, map relative to the learned threshold.
        # We want: raw == threshold -> 0.5, raw < threshold -> >0.5 (anomalous)
        if self.raw_score_threshold is not None:
            std = float(self.raw_score_std or 1.0)
            z = (float(self.raw_score_threshold) - raw_scores) / (std + 1e-9)
            proba = 1 / (1 + np.exp(-self.calibration_scale * z))
            return proba

        # Fallback (older models): monotonic mapping (may be poorly calibrated)
        z = -raw_scores
        proba = 1 / (1 + np.exp(-10 * z))
        return proba

    def get_feature_importances(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Compute approximate feature importances for anomaly detection.
        Uses permutation-based importance estimation.

        Args:
            X: Feature matrix

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X)

        # Baseline scores
        base_scores = self.model.score_samples(X_scaled)

        importances = {}

        for i, feature_name in enumerate(self.feature_names):
            # Permute feature
            X_permuted = X_scaled.copy()
            np.random.shuffle(X_permuted[:, i])

            # Score with permuted feature
            permuted_scores = self.model.score_samples(X_permuted)

            # Importance = change in score variance
            importance = np.abs(base_scores - permuted_scores).mean()
            importances[feature_name] = importance

        # Normalize
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances

    def get_anomaly_contributions(self, x: np.ndarray) -> dict[str, float]:
        """
        Get feature contributions to anomaly score for a single sample.
        Useful for explaining why a specific sample was flagged.

        Args:
            x: Single sample feature vector (1D array)

        Returns:
            Dictionary mapping feature names to contribution scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        x_scaled = self.scaler.transform(x)

        # Baseline score
        base_score = self.model.score_samples(x_scaled)[0]

        contributions = {}

        for i, feature_name in enumerate(self.feature_names):
            # Zero out this feature (use mean = 0 after scaling)
            x_modified = x_scaled.copy()
            x_modified[0, i] = 0

            # Score without this feature
            modified_score = self.model.score_samples(x_modified)[0]

            # Contribution = how much removing this feature changes the score
            # Negative contribution means feature makes sample more anomalous
            contributions[feature_name] = base_score - modified_score

        return contributions

    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "is_fitted": self.is_fitted,
            "raw_score_mean": self.raw_score_mean,
            "raw_score_std": self.raw_score_std,
            "raw_score_threshold": self.raw_score_threshold,
            "calibration_scale": self.calibration_scale,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        """Load model from disk."""
        model_data = joblib.load(path)

        detector = cls(
            contamination=model_data["contamination"],
            n_estimators=model_data["n_estimators"]
        )
        detector.model = model_data["model"]
        detector.scaler = model_data["scaler"]
        detector.feature_names = model_data["feature_names"]
        detector.is_fitted = model_data["is_fitted"]

        detector.raw_score_mean = model_data.get("raw_score_mean")
        detector.raw_score_std = model_data.get("raw_score_std")
        detector.raw_score_threshold = model_data.get("raw_score_threshold")
        detector.calibration_scale = float(model_data.get("calibration_scale", detector.calibration_scale))

        logger.info(f"Model loaded from {path}")
        return detector


def train_isolation_forest(
    X: np.ndarray,
    feature_names: list[str],
    contamination: float = 0.05,
    save_path: Optional[str] = None
) -> IsolationForestDetector:
    """
    Train an Isolation Forest detector.

    Args:
        X: Feature matrix
        feature_names: List of feature names
        contamination: Expected anomaly rate
        save_path: Optional path to save the trained model

    Returns:
        Trained detector
    """
    detector = IsolationForestDetector(contamination=contamination)
    detector.fit(X, feature_names)

    if save_path:
        detector.save(save_path)

    return detector


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)

    # Generate normal data
    n_normal = 950
    X_normal = np.random.randn(n_normal, 10)

    # Generate anomalies (different distribution)
    n_anomaly = 50
    X_anomaly = np.random.randn(n_anomaly, 10) * 3 + 5

    # Combine
    X = np.vstack([X_normal, X_anomaly])
    y_true = np.array([0] * n_normal + [1] * n_anomaly)

    feature_names = [f"feature_{i}" for i in range(10)]

    # Train model
    detector = IsolationForestDetector(contamination=0.05)
    detector.fit(X, feature_names)

    # Get predictions
    scores = detector.predict_proba(X)
    predictions = detector.predict(X)

    # Evaluate
    anomalies_detected = (predictions == -1)
    true_positives = anomalies_detected[y_true == 1].sum()
    false_positives = anomalies_detected[y_true == 0].sum()

    print(f"True Positives: {true_positives}/{n_anomaly}")
    print(f"False Positives: {false_positives}/{n_normal}")
    print(f"Precision: {true_positives / (true_positives + false_positives):.2%}")
    print(f"Recall: {true_positives / n_anomaly:.2%}")

    # Feature contributions for an anomaly
    anomaly_idx = np.where(y_true == 1)[0][0]
    contributions = detector.get_anomaly_contributions(X[anomaly_idx])
    print(f"\nTop contributing features for anomaly:")
    for name, score in sorted(contributions.items(), key=lambda x: x[1])[:5]:
        print(f"  {name}: {score:.4f}")
