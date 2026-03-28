"""
Anomaly Explainer for ForexGuard
Provides SHAP-based and rule-based explanations for anomaly scores.
"""

import numpy as np
from typing import Optional, Union
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Using rule-based explanations only.")


class AnomalyExplainer:
    """
    Provides explainable anomaly scores using SHAP values
    and rule-based heuristics.
    """

    # Feature description templates
    FEATURE_DESCRIPTIONS = {
        "ip_change_count": "IP address changes ({value:.0f} changes)",
        "ip_change_rate": "IP change rate ({value:.2f}/hour)",
        "unique_ips": "Unique IPs used ({value:.0f} IPs)",
        "ip_entropy": "IP diversity score ({value:.2f})",
        "trade_volume_zscore": "Trade volume deviation ({value:.2f} std from mean)",
        "lot_size_zscore": "Lot size deviation ({value:.2f} std from mean)",
        "trade_count_zscore": "Trade frequency deviation ({value:.2f} std from mean)",
        "deposit_amount_zscore": "Deposit amount deviation ({value:.2f} std from mean)",
        "withdrawal_amount_zscore": "Withdrawal amount deviation ({value:.2f} std from mean)",
        "deposit_withdrawal_ratio": "Deposit/withdrawal ratio ({value:.2f})",
        "transaction_velocity": "Transaction velocity ({value:.2f}/hour)",
        "session_duration_zscore": "Session duration deviation ({value:.2f} std from mean)",
        "login_failure_rate": "Login failure rate ({value:.1%})",
        "device_change_rate": "Device change rate ({value:.2f}/day)",
        "unique_devices": "Unique devices used ({value:.0f} devices)",
        "time_since_last_event": "Time since last event ({value:.1f} seconds)",
        "event_frequency": "Event frequency ({value:.2f}/hour)",
        "margin_utilization": "Margin utilization ({value:.1%})",
        "leverage_zscore": "Leverage deviation ({value:.2f} std from mean)"
    }

    # Rule-based explanation thresholds
    ANOMALY_RULES = {
        "ip_change_count": {"threshold": 3, "msg": "unusual IP switching pattern"},
        "ip_change_rate": {"threshold": 2.0, "msg": "rapid IP changes"},
        "trade_volume_zscore": {"threshold": 2.0, "msg": "abnormal trade volume"},
        "lot_size_zscore": {"threshold": 2.5, "msg": "unusual lot sizes"},
        "deposit_amount_zscore": {"threshold": 2.0, "msg": "unusual deposit amount"},
        "withdrawal_amount_zscore": {"threshold": 2.0, "msg": "unusual withdrawal amount"},
        "transaction_velocity": {"threshold": 5.0, "msg": "high transaction frequency"},
        "device_change_rate": {"threshold": 3.0, "msg": "frequent device changes"},
        "login_failure_rate": {"threshold": 0.3, "msg": "high login failure rate"},
        "margin_utilization": {"threshold": 0.9, "msg": "high margin utilization"}
    }

    def __init__(self, model, feature_pipeline, use_shap: bool = True):
        """
        Initialize explainer.

        Args:
            model: Trained anomaly detection model
            feature_pipeline: Feature extraction pipeline
            use_shap: Whether to use SHAP for explanations
        """
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.use_shap = use_shap and SHAP_AVAILABLE
        self.shap_explainer = None

        if self.use_shap:
            self._init_shap_explainer()

    def _init_shap_explainer(self) -> None:
        """Initialize SHAP explainer for the model."""
        try:
            # For tree-based models (Isolation Forest)
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'estimators_'):
                self.shap_explainer = shap.TreeExplainer(self.model.model)
                logger.info("Initialized SHAP TreeExplainer for Isolation Forest")
            else:
                # Fallback to KernelExplainer for other models
                logger.info("Using rule-based explanations (SHAP not compatible with model type)")
                self.use_shap = False
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.use_shap = False

    def explain(self, event: dict) -> tuple[float, list[tuple], str]:
        """
        Generate anomaly score with explanation.

        Args:
            event: Event dictionary

        Returns:
            Tuple of (anomaly_score, feature_contributions, explanation_text)
        """
        # Extract features using the pipeline
        feature_dict = self.feature_pipeline.transform_event(event, update_store=True)

        # Convert to DataFrame and get model features
        import pandas as pd
        df = pd.DataFrame([feature_dict])
        features, feature_names = self.feature_pipeline.get_model_features(df)
        features = features[0]  # Get first row as 1D array

        # Get anomaly score (predict_proba returns array, get single value)
        score_array = self.model.predict_proba(features.reshape(1, -1))
        score = float(score_array[0])

        # Get feature contributions
        if self.use_shap and self.shap_explainer:
            contributions = self._get_shap_contributions(features, feature_names)
        else:
            contributions = self._get_rule_contributions(features, feature_names)

        # Generate explanation text
        explanation = self._generate_explanation(score, contributions)

        return score, contributions, explanation

    def _get_shap_contributions(
        self,
        features: np.ndarray,
        feature_names: list[str]
    ) -> list[tuple]:
        """Get SHAP-based feature contributions."""
        try:
            shap_values = self.shap_explainer.shap_values(features.reshape(1, -1))

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_values = shap_values.flatten()

            contributions = []
            for i, (name, value, shap_val) in enumerate(zip(feature_names, features.flatten(), shap_values)):
                desc = self._get_feature_description(name, value)
                contributions.append((name, float(value), float(abs(shap_val)), desc))

            # Sort by absolute contribution
            contributions.sort(key=lambda x: x[2], reverse=True)

            return contributions

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return self._get_rule_contributions(features, feature_names)

    def _get_rule_contributions(
        self,
        features: np.ndarray,
        feature_names: list[str]
    ) -> list[tuple]:
        """Get rule-based feature contributions."""
        contributions = []

        for name, value in zip(feature_names, features.flatten()):
            # Calculate contribution based on rules
            contribution = self._calculate_rule_contribution(name, value)
            desc = self._get_feature_description(name, value)
            contributions.append((name, float(value), contribution, desc))

        # Sort by contribution
        contributions.sort(key=lambda x: x[2], reverse=True)

        return contributions

    def _calculate_rule_contribution(self, feature_name: str, value: float) -> float:
        """Calculate rule-based contribution score for a feature."""
        # Check if we have a rule for this feature
        for rule_name, rule in self.ANOMALY_RULES.items():
            if rule_name in feature_name.lower():
                threshold = rule["threshold"]
                if abs(value) > threshold:
                    # Normalize contribution to 0-1 scale
                    return min(1.0, abs(value) / (threshold * 3))

        # For z-score features, use standard thresholds
        if "zscore" in feature_name.lower():
            return min(1.0, abs(value) / 3.0)

        # Default: normalize based on value magnitude
        return min(1.0, abs(value) / 10.0)

    def _get_feature_description(self, feature_name: str, value: float) -> str:
        """Get human-readable description for a feature."""
        # Check for exact match
        if feature_name in self.FEATURE_DESCRIPTIONS:
            return self.FEATURE_DESCRIPTIONS[feature_name].format(value=value)

        # Check for partial match
        for key, template in self.FEATURE_DESCRIPTIONS.items():
            if key in feature_name.lower():
                return template.format(value=value)

        # Default description
        return f"{feature_name}: {value:.2f}"

    def _generate_explanation(
        self,
        score: float,
        contributions: list[tuple]
    ) -> str:
        """Generate human-readable explanation text."""
        if score < 0.5:
            return "Activity appears normal with no significant anomalies detected."

        # Get top contributing factors
        top_factors = []

        for name, value, contrib, desc in contributions[:5]:
            if contrib > 0.1:  # Only include significant contributors
                # Find matching rule message
                for rule_name, rule in self.ANOMALY_RULES.items():
                    if rule_name in name.lower() and abs(value) > rule["threshold"]:
                        top_factors.append(rule["msg"])
                        break
                else:
                    # Default message based on feature name
                    if "zscore" in name.lower() and abs(value) > 2:
                        feature_base = name.replace("_zscore", "").replace("_", " ")
                        top_factors.append(f"unusual {feature_base}")

        # Remove duplicates while preserving order
        seen = set()
        unique_factors = []
        for f in top_factors:
            if f not in seen:
                seen.add(f)
                unique_factors.append(f)

        if not unique_factors:
            unique_factors = ["deviation from normal behavioral patterns"]

        # Build explanation
        if score >= 0.95:
            severity = "Critical anomaly"
        elif score >= 0.85:
            severity = "High-risk anomaly"
        elif score >= 0.70:
            severity = "Moderate anomaly"
        else:
            severity = "Low-risk anomaly"

        factors_text = ", ".join(unique_factors[:3])
        if len(unique_factors) > 3:
            factors_text += f", and {len(unique_factors) - 3} other factor(s)"

        return f"{severity} detected due to {factors_text}."

    def explain_batch(
        self,
        events: list[dict]
    ) -> list[tuple[float, list[tuple], str]]:
        """Explain a batch of events."""
        return [self.explain(event) for event in events]


class RuleBasedExplainer:
    """
    Lightweight rule-based explainer for when SHAP is not available
    or for quick explanations.
    """

    RULES = [
        {
            "name": "ip_switching",
            "condition": lambda f: f.get("ip_change_count", 0) > 3,
            "message": "unusual IP switching pattern",
            "weight": 0.3
        },
        {
            "name": "trade_spike",
            "condition": lambda f: abs(f.get("trade_volume_zscore", 0)) > 2,
            "message": "abnormal trade volume spike",
            "weight": 0.25
        },
        {
            "name": "deposit_anomaly",
            "condition": lambda f: abs(f.get("deposit_amount_zscore", 0)) > 2,
            "message": "unusual deposit amount",
            "weight": 0.2
        },
        {
            "name": "withdrawal_anomaly",
            "condition": lambda f: abs(f.get("withdrawal_amount_zscore", 0)) > 2,
            "message": "unusual withdrawal amount",
            "weight": 0.2
        },
        {
            "name": "device_switching",
            "condition": lambda f: f.get("device_change_rate", 0) > 3,
            "message": "frequent device changes",
            "weight": 0.15
        },
        {
            "name": "session_anomaly",
            "condition": lambda f: abs(f.get("session_duration_zscore", 0)) > 2,
            "message": "unusual session duration",
            "weight": 0.1
        },
        {
            "name": "high_margin",
            "condition": lambda f: f.get("margin_utilization", 0) > 0.9,
            "message": "high margin utilization",
            "weight": 0.15
        }
    ]

    def explain(self, features: dict) -> tuple[float, list[str]]:
        """
        Generate rule-based explanation.

        Args:
            features: Dictionary of feature values

        Returns:
            Tuple of (anomaly_score, list of triggered rule messages)
        """
        triggered = []
        total_weight = 0

        for rule in self.RULES:
            try:
                if rule["condition"](features):
                    triggered.append(rule["message"])
                    total_weight += rule["weight"]
            except (KeyError, TypeError):
                continue

        # Normalize score to 0-1
        score = min(1.0, total_weight)

        return score, triggered
