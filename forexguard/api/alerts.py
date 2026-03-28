"""
Alert Generation for ForexGuard
Generates human-readable alerts with severity levels and recommended actions.
"""

import uuid
from datetime import datetime
from typing import Optional

from loguru import logger

from .schemas import Alert, AnomalyScore, SeverityLevel


class AlertGenerator:
    """
    Generates alerts from anomaly scores with severity classification
    and human-readable descriptions.
    """

    # Severity thresholds
    SEVERITY_THRESHOLDS = {
        SeverityLevel.CRITICAL: 0.95,
        SeverityLevel.HIGH: 0.85,
        SeverityLevel.MEDIUM: 0.70,
        SeverityLevel.LOW: 0.50
    }

    # Alert templates by anomaly pattern
    ALERT_TEMPLATES = {
        "ip_switching": {
            "title": "Suspicious IP Switching Detected",
            "description": "User {user_id} has been accessing the platform from multiple unusual IP addresses in a short time period. This may indicate account compromise or VPN/proxy usage for fraud.",
            "action": "Review recent login history and verify user identity. Consider requiring additional authentication."
        },
        "trade_spike": {
            "title": "Abnormal Trading Volume Detected",
            "description": "User {user_id} has shown a significant spike in trading activity ({spike_factor:.1f}x normal volume). This may indicate automated trading or potential market manipulation.",
            "action": "Review trading patterns and positions. Consider setting temporary trading limits."
        },
        "deposit_withdrawal": {
            "title": "Suspicious Deposit/Withdrawal Pattern",
            "description": "User {user_id} has exhibited unusual deposit-withdrawal behavior with {amount_total:,.2f} USD in rapid transactions. This may indicate money laundering or bonus abuse.",
            "action": "Flag for compliance review. Verify source of funds and withdrawal destinations."
        },
        "session_anomaly": {
            "title": "Session Anomaly Detected",
            "description": "User {user_id} shows impossible session patterns (concurrent sessions from distant locations). This may indicate session hijacking or credential sharing.",
            "action": "Terminate all active sessions and require password reset. Enable 2FA if not active."
        },
        "device_switching": {
            "title": "Rapid Device Switching Detected",
            "description": "User {user_id} has accessed the platform from {device_count} different devices in a short period. This may indicate account sharing or compromise.",
            "action": "Review device history and verify user identity. Consider device binding."
        },
        "general": {
            "title": "Anomalous Activity Detected",
            "description": "User {user_id} has been flagged for unusual behavior patterns that deviate significantly from their normal activity profile.",
            "action": "Review user activity history and investigate flagged events."
        }
    }

    def __init__(self, alert_threshold: float = 0.5):
        """
        Initialize alert generator.

        Args:
            alert_threshold: Minimum anomaly score to generate alert
        """
        self.alert_threshold = alert_threshold
        self.alert_history: list[Alert] = []

    def classify_severity(self, score: float) -> SeverityLevel:
        """Classify severity based on anomaly score."""
        for severity, threshold in self.SEVERITY_THRESHOLDS.items():
            if score >= threshold:
                return severity
        return SeverityLevel.LOW

    def detect_anomaly_pattern(self, score: AnomalyScore) -> str:
        """
        Detect the likely anomaly pattern from feature contributions.

        Args:
            score: Anomaly score with feature contributions

        Returns:
            Detected pattern name
        """
        if not score.feature_contributions:
            return "general"

        # Map feature names to patterns
        pattern_indicators = {
            "ip_switching": ["ip_change", "unique_ips", "ip_entropy", "geo_distance"],
            "trade_spike": ["trade_volume", "lot_size", "trade_count", "margin"],
            "deposit_withdrawal": ["deposit", "withdrawal", "transaction", "amount"],
            "session_anomaly": ["session", "concurrent", "duration", "login"],
            "device_switching": ["device", "unique_device", "device_change"]
        }

        # Score each pattern based on feature contributions
        pattern_scores = {pattern: 0.0 for pattern in pattern_indicators}

        for contrib in score.feature_contributions:
            feature_lower = contrib.feature_name.lower()
            for pattern, indicators in pattern_indicators.items():
                if any(ind in feature_lower for ind in indicators):
                    pattern_scores[pattern] += abs(contrib.contribution)

        # Return pattern with highest score
        if max(pattern_scores.values()) > 0:
            return max(pattern_scores, key=pattern_scores.get)

        return "general"

    def generate_alert(
        self,
        score: AnomalyScore,
        context: Optional[dict] = None
    ) -> Optional[Alert]:
        """
        Generate an alert from an anomaly score.

        Args:
            score: Anomaly score result
            context: Optional additional context

        Returns:
            Alert if score exceeds threshold, None otherwise
        """
        if score.anomaly_score < self.alert_threshold:
            return None

        context = context or {}

        # Detect pattern
        pattern = self.detect_anomaly_pattern(score)
        template = self.ALERT_TEMPLATES.get(pattern, self.ALERT_TEMPLATES["general"])

        # Build description context
        desc_context = {
            "user_id": score.user_id,
            "spike_factor": context.get("spike_factor", 5.0),
            "amount_total": context.get("amount_total", 50000.0),
            "device_count": context.get("device_count", 5),
            **context
        }

        # Generate alert
        alert = Alert(
            alert_id=f"alert_{uuid.uuid4().hex[:12]}",
            event_id=score.event_id,
            user_id=score.user_id,
            severity=score.severity,
            anomaly_score=score.anomaly_score,
            title=template["title"],
            description=template["description"].format(**desc_context),
            recommended_action=template["action"],
            created_at=datetime.utcnow()
        )

        self.alert_history.append(alert)

        logger.info(f"Generated {alert.severity.value} alert for user {score.user_id}: {alert.title}")

        return alert

    def generate_batch_alerts(
        self,
        scores: list[AnomalyScore],
        context: Optional[dict] = None
    ) -> list[Alert]:
        """Generate alerts for a batch of scores."""
        alerts = []
        for score in scores:
            alert = self.generate_alert(score, context)
            if alert:
                alerts.append(alert)
        return alerts

    def get_alert_summary(self) -> dict:
        """Get summary of generated alerts."""
        if not self.alert_history:
            return {"total": 0, "by_severity": {}}

        by_severity = {}
        for alert in self.alert_history:
            sev = alert.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total": len(self.alert_history),
            "by_severity": by_severity,
            "unacknowledged": sum(1 for a in self.alert_history if not a.acknowledged)
        }

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow()
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        return False

    def get_recent_alerts(
        self,
        limit: int = 100,
        severity: Optional[SeverityLevel] = None,
        user_id: Optional[str] = None
    ) -> list[Alert]:
        """Get recent alerts with optional filtering."""
        alerts = self.alert_history

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if user_id:
            alerts = [a for a in alerts if a.user_id == user_id]

        # Sort by creation time (newest first)
        alerts = sorted(alerts, key=lambda a: a.created_at, reverse=True)

        return alerts[:limit]


# Global alert generator instance
_alert_generator: Optional[AlertGenerator] = None


def get_alert_generator(threshold: float = 0.5) -> AlertGenerator:
    """Get or create global alert generator."""
    global _alert_generator
    if _alert_generator is None:
        _alert_generator = AlertGenerator(alert_threshold=threshold)
    return _alert_generator
