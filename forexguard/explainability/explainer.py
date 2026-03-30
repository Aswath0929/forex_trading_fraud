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
        # Access / identity
        "ip_change_rate": "IP change rate ({value:.2f})",
        "unique_ips": "Unique IPs used ({value:.0f})",
        "device_change_rate": "Device change rate ({value:.2f})",
        "unique_devices": "Unique devices used ({value:.0f})",
        "failed_login_rate": "Login failure rate ({value:.1%})",
        "consecutive_login_failures": "Consecutive login failures ({value:.0f})",
        "event_hour": "Event hour ({value:.0f})",
        "is_night": "Night-time activity flag ({value:.0f})",

        # Hub / shared entity signals
        "ip_shared_users_24h": "Other users seen on same IP in 24h ({value:.0f})",
        "device_shared_users_24h": "Other users seen on same device in 24h ({value:.0f})",

        # Trading / payments
        "lot_size_zscore": "Lot size deviation ({value:.2f} std from mean)",
        "amount_zscore": "Amount deviation ({value:.2f} std from mean)",
        "withdrawal_deposit_ratio": "Withdrawal/deposit ratio ({value:.2f})",

        # Portal behavior
        "page_view_count": "Portal page views in 24h ({value:.0f})",
        "unique_pages": "Unique portal pages in 24h ({value:.0f})",
        "support_ticket_count": "Support tickets in 24h ({value:.0f})",
        "document_upload_count": "Document uploads in 24h ({value:.0f})",
        "document_upload_fail_rate": "Document upload failure rate ({value:.1%})",
        "avg_ticket_priority": "Average ticket priority ({value:.2f})",

        # Session / temporal
        "session_duration_zscore": "Session duration deviation ({value:.2f} std from mean)",
        "time_since_last_event": "Time since last event ({value:.1f} seconds)",
    }

    # Rule-based explanation thresholds
    ANOMALY_RULES = {
        # Access / identity
        "rapid_ip_switching": {"threshold": 0.5, "msg": "unusual IP switching pattern"},
        "rapid_device_switching": {"threshold": 0.5, "msg": "frequent device switching"},
        "failed_login_rate": {"threshold": 0.3, "msg": "high login failure rate"},
        "consecutive_login_failures": {"threshold": 3, "msg": "multiple consecutive failed logins"},
        "login_bruteforce": {"threshold": 0.5, "msg": "possible brute-force login behavior"},
        "is_night": {"threshold": 0.5, "msg": "unusual login/activity time (night hours)"},

        # Hub / shared entity signals
        "ip_shared_users_24h": {"threshold": 5, "msg": "many users sharing the same IP (IP hub)"},
        "device_shared_users_24h": {"threshold": 3, "msg": "many users sharing the same device fingerprint"},
        "ip_hub_behavior": {"threshold": 0.5, "msg": "IP hub behavior"},
        "device_hub_behavior": {"threshold": 0.5, "msg": "device hub behavior"},

        # Portal behavior
        "rapid_navigation": {"threshold": 0.5, "msg": "bot-like rapid navigation"},
        "page_diversity_spike": {"threshold": 0.5, "msg": "unusually diverse portal navigation"},
        "document_upload_spike": {"threshold": 0.5, "msg": "document upload spike"},
        "support_ticket_spike": {"threshold": 0.5, "msg": "support ticket spike"},
        "account_change_rush": {"threshold": 0.5, "msg": "rapid account/profile changes"},

        # Financial / trading
        "amount_zscore": {"threshold": 2.0, "msg": "unusual transaction amount"},
        "lot_size_zscore": {"threshold": 2.5, "msg": "unusual lot sizes"},
        "withdrawal_deposit_ratio": {"threshold": 3.0, "msg": "withdrawal-heavy pattern"},
        "dormancy_withdrawal": {"threshold": 0.5, "msg": "large withdrawal after dormancy"},
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

    def explain_shap(self, event: dict, top_k: int = 10) -> dict:
        """Return SHAP (or rule-based fallback) attributions for a single event.

        This is meant to be consumed by the UI (Swagger /docs).
        """
        feature_dict = self.feature_pipeline.transform_event(event, update_store=True)

        import pandas as pd
        df = pd.DataFrame([feature_dict])

        expected = getattr(self.model, "feature_names", None) or None
        X, feature_names = self.feature_pipeline.get_model_features(df, expected_features=expected)
        x = X[0]

        score = float(self.model.predict_proba(x.reshape(1, -1))[0])

        # Default to rule-based output
        base_value = None
        method = "rule"
        attributions: list[dict] = []

        if self.use_shap and self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(x.reshape(1, -1))
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_values = np.asarray(shap_values).reshape(-1)

                ev = getattr(self.shap_explainer, "expected_value", None)
                if ev is not None:
                    ev_arr = np.asarray(ev).reshape(-1)
                    base_value = float(ev_arr[0]) if ev_arr.size else None

                method = "shap"
                for name, val, sv in zip(feature_names, x.reshape(-1), shap_values):
                    desc = self._get_feature_description(name, float(val))
                    attributions.append({
                        "feature_name": name,
                        "value": float(val),
                        "shap_value": float(sv),
                        "abs_contribution": float(abs(sv)),
                        "description": desc,
                    })

                attributions.sort(key=lambda d: d["abs_contribution"], reverse=True)
                attributions = attributions[: max(1, int(top_k))]

                return {
                    "anomaly_score": score,
                    "model_type": getattr(self.model, "__class__", type(self.model)).__name__,
                    "method": method,
                    "base_value": base_value,
                    "top_features": attributions,
                }

            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")

        # Rule-based fallback using existing contribution logic
        contribs = self._get_rule_contributions(x, feature_names)
        for name, val, contrib, desc in contribs[: max(1, int(top_k))]:
            attributions.append({
                "feature_name": name,
                "value": float(val),
                "shap_value": None,
                "abs_contribution": float(contrib),
                "description": desc,
            })

        return {
            "anomaly_score": score,
            "model_type": getattr(self.model, "__class__", type(self.model)).__name__,
            "method": method,
            "base_value": base_value,
            "top_features": attributions,
        }

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

        expected = getattr(self.model, "feature_names", None) or None
        features, feature_names = self.feature_pipeline.get_model_features(df, expected_features=expected)
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
            "condition": lambda f: f.get("rapid_ip_switching", 0) == 1,
            "message": "unusual IP switching pattern",
            "weight": 0.25,
        },
        {
            "name": "device_switching",
            "condition": lambda f: f.get("rapid_device_switching", 0) == 1,
            "message": "frequent device switching",
            "weight": 0.2,
        },
        {
            "name": "login_bruteforce",
            "condition": lambda f: f.get("login_bruteforce", 0) == 1,
            "message": "possible brute-force login behavior",
            "weight": 0.25,
        },
        {
            "name": "ip_hub",
            "condition": lambda f: f.get("ip_hub_behavior", 0) == 1,
            "message": "many users sharing the same IP (IP hub)",
            "weight": 0.2,
        },
        {
            "name": "rapid_navigation",
            "condition": lambda f: f.get("rapid_navigation", 0) == 1,
            "message": "bot-like rapid navigation",
            "weight": 0.15,
        },
        {
            "name": "document_upload_spike",
            "condition": lambda f: f.get("document_upload_spike", 0) == 1,
            "message": "document upload spike",
            "weight": 0.1,
        },
        {
            "name": "session_anomaly",
            "condition": lambda f: abs(f.get("session_duration_zscore", 0)) > 2,
            "message": "unusual session duration",
            "weight": 0.1,
        },
        {
            "name": "amount_anomaly",
            "condition": lambda f: abs(f.get("amount_zscore", 0)) > 2,
            "message": "unusual transaction amount",
            "weight": 0.2,
        },
        {
            "name": "lot_size_anomaly",
            "condition": lambda f: abs(f.get("lot_size_zscore", 0)) > 2.5,
            "message": "unusual lot sizes",
            "weight": 0.2,
        },
        {
            "name": "dormancy_withdrawal",
            "condition": lambda f: f.get("dormancy_withdrawal", 0) == 1,
            "message": "large withdrawal after dormancy",
            "weight": 0.2,
        },
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
