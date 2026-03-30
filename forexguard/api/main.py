"""
FastAPI Application for ForexGuard
Real-time anomaly detection API for forex brokerage fraud detection.
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    EventInput,
    AnomalyScore,
    BatchEventInput,
    BatchAnomalyScore,
    Alert,
    SeverityLevel,
    HealthResponse,
    ModelInfo,
    FeatureContribution,
    ShapExplanation,
    ShapFeatureContribution,
)
from api.alerts import AlertGenerator, get_alert_generator
from features.feature_pipeline import FeaturePipeline
from features.feature_store import GlobalFeatureStore
from models.model_registry import get_registry, ModelRegistry
from explainability.explainer import AnomalyExplainer


def _parse_model_id(model_id: str) -> tuple[str, str, str]:
    """Parse model_id of the form: <model_type>_<model_name>_<version>.

    Note: model_type itself contains underscores and version is like v_YYYYMMDD_HHMMSS,
    so we parse from the right using the "_v_" marker and then match known types.
    """
    if "_v_" not in model_id:
        raise ValueError("Invalid model_id format")

    left, version_suffix = model_id.rsplit("_v_", 1)
    version = f"v_{version_suffix}"

    known_types = ["lstm_autoencoder", "isolation_forest"]
    for t in sorted(known_types, key=len, reverse=True):
        prefix = t + "_"
        if left.startswith(prefix):
            name = left[len(prefix):]
            if not name:
                raise ValueError("Invalid model_id (missing model name)")
            return t, name, version

    raise ValueError("Invalid model_id (unknown model type)")


# Global instances
feature_pipeline: Optional[FeaturePipeline] = None
feature_store: Optional[GlobalFeatureStore] = None
model_registry: Optional[ModelRegistry] = None
explainer: Optional[AnomalyExplainer] = None
alert_generator: Optional[AlertGenerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global feature_pipeline, feature_store, model_registry, explainer, alert_generator

    logger.info("Starting ForexGuard API...")

    # Initialize components
    feature_pipeline = FeaturePipeline()
    feature_store = GlobalFeatureStore()
    model_registry = get_registry("models/saved")
    threshold = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))
    alert_generator = get_alert_generator(threshold=threshold)

    # Try to load model
    try:
        model = model_registry.load_active()
        explainer = AnomalyExplainer(model, feature_pipeline)
        logger.info(f"Loaded model: {model_registry.get_active_model_id()}")
    except ValueError as e:
        logger.warning(f"No model loaded: {e}. Train a model first.")
        explainer = None

    yield

    logger.info("Shutting down ForexGuard API...")


# Create FastAPI app
app = FastAPI(
    title="ForexGuard API",
    description="Real-time anomaly detection for forex brokerage fraud prevention",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _score_to_severity(score: float) -> SeverityLevel:
    """Convert anomaly score to severity level."""
    if score >= 0.95:
        return SeverityLevel.CRITICAL
    elif score >= 0.85:
        return SeverityLevel.HIGH
    elif score >= 0.70:
        return SeverityLevel.MEDIUM
    else:
        return SeverityLevel.LOW


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": "ForexGuard API",
        "version": "1.0.0",
        "description": "Real-time anomaly detection for forex brokerage fraud prevention",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    model_loaded = explainer is not None
    model_type = None

    if model_loaded and model_registry:
        try:
            model_id = model_registry.get_active_model_id()
            if model_id:
                model_type = model_id.split("_")[0]
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_type=model_type,
        version="1.0.0"
    )


@app.post("/score", response_model=AnomalyScore, tags=["Scoring"])
async def score_event(
    event: EventInput,
    background_tasks: BackgroundTasks,
    generate_alert: bool = Query(True, description="Whether to generate alert for anomalies")
):
    """
    Score a single event for anomalies.

    Returns anomaly score with feature contributions and human-readable explanation.
    """
    if explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    try:
        start_time = time.time()

        # Convert event to dict
        event_dict = event.model_dump()

        # Get anomaly score with explanation
        score, contributions, explanation = explainer.explain(event_dict)

        # Determine if anomaly
        threshold = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))
        is_anomaly = score >= threshold
        severity = _score_to_severity(score)

        # Build feature contributions
        feature_contribs = [
            FeatureContribution(
                feature_name=name,
                value=value,
                contribution=contrib,
                description=desc
            )
            for name, value, contrib, desc in contributions[:10]  # Top 10 contributors
        ]

        result = AnomalyScore(
            event_id=event.event_id,
            user_id=event.user_id,
            anomaly_score=round(score, 4),
            is_anomaly=is_anomaly,
            threshold=threshold,
            model_type=model_registry.get_active_model_id() or "unknown",
            severity=severity,
            feature_contributions=feature_contribs,
            explanation=explanation,
            timestamp=datetime.now(timezone.utc)
        )

        # Generate alert in background if anomaly
        if generate_alert and is_anomaly and alert_generator:
            background_tasks.add_task(
                alert_generator.generate_alert,
                result
            )

        processing_time = (time.time() - start_time) * 1000
        logger.debug(f"Scored event {event.event_id} in {processing_time:.2f}ms: score={score:.4f}")

        return result

    except Exception as e:
        logger.error(f"Error scoring event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/shap", response_model=ShapExplanation, tags=["Explainability"])
async def explain_shap(
    event: EventInput,
    top_k: int = Query(10, ge=1, le=50, description="Number of top contributing features to return")
):
    """Explain a single event using SHAP if available (otherwise rule-based fallback)."""
    if explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")

    try:
        event_dict = event.model_dump()
        payload = explainer.explain_shap(event_dict, top_k=top_k)

        top_features = [ShapFeatureContribution(**d) for d in payload.get("top_features", [])]

        return ShapExplanation(
            event_id=event.event_id,
            user_id=event.user_id,
            anomaly_score=round(float(payload.get("anomaly_score", 0.0)), 4),
            model_type=(model_registry.get_active_model_id() or "unknown") if model_registry else "unknown",
            method=str(payload.get("method", "rule")),
            base_value=payload.get("base_value"),
            top_features=top_features,
        )

    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/batch", response_model=BatchAnomalyScore, tags=["Scoring"])
async def score_batch(
    batch: BatchEventInput,
    background_tasks: BackgroundTasks,
    generate_alerts: bool = Query(True, description="Whether to generate alerts for anomalies")
):
    """
    Score a batch of events for anomalies.

    More efficient than scoring individually for large batches.
    """
    if explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    try:
        start_time = time.time()
        scores = []
        anomalies_found = 0

        for event in batch.events:
            event_dict = event.model_dump()

            # Get anomaly score with explanation
            score, contributions, explanation = explainer.explain(event_dict)

            threshold = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))
            is_anomaly = score >= threshold
            severity = _score_to_severity(score)

            if is_anomaly:
                anomalies_found += 1

            feature_contribs = [
                FeatureContribution(
                    feature_name=name,
                    value=value,
                    contribution=contrib,
                    description=desc
                )
                for name, value, contrib, desc in contributions[:5]  # Top 5 for batch
            ]

            result = AnomalyScore(
                event_id=event.event_id,
                user_id=event.user_id,
                anomaly_score=round(score, 4),
                is_anomaly=is_anomaly,
                threshold=threshold,
                model_type=model_registry.get_active_model_id() or "unknown",
                severity=severity,
                feature_contributions=feature_contribs,
                explanation=explanation,
                timestamp=datetime.now(timezone.utc)
            )
            scores.append(result)

            # Generate alert if anomaly
            if generate_alerts and is_anomaly and alert_generator:
                background_tasks.add_task(
                    alert_generator.generate_alert,
                    result
                )

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"Scored batch of {len(batch.events)} events in {processing_time:.2f}ms: {anomalies_found} anomalies")

        return BatchAnomalyScore(
            scores=scores,
            total_events=len(batch.events),
            anomalies_found=anomalies_found,
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"Error scoring batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts", response_model=list[Alert], tags=["Alerts"])
async def get_alerts(
    limit: int = Query(100, ge=1, le=1000),
    severity: Optional[SeverityLevel] = Query(None),
    user_id: Optional[str] = Query(None)
):
    """Get recent alerts with optional filtering."""
    if alert_generator is None:
        return []

    return alert_generator.get_recent_alerts(
        limit=limit,
        severity=severity,
        user_id=user_id
    )


@app.post("/alerts/{alert_id}/acknowledge", tags=["Alerts"])
async def acknowledge_alert(alert_id: str, acknowledged_by: str = Query(...)):
    """Acknowledge an alert."""
    if alert_generator is None:
        raise HTTPException(status_code=404, detail="Alert system not initialized")

    success = alert_generator.acknowledge_alert(alert_id, acknowledged_by)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"status": "acknowledged", "alert_id": alert_id}


@app.get("/alerts/summary", tags=["Alerts"])
async def get_alert_summary():
    """Get summary of alerts."""
    if alert_generator is None:
        return {"total": 0, "by_severity": {}, "unacknowledged": 0}

    return alert_generator.get_alert_summary()


@app.get("/models", response_model=list[ModelInfo], tags=["Models"])
async def list_models():
    """List all registered models."""
    if model_registry is None:
        return []

    models = model_registry.list_models()
    active_id = model_registry.get_active_model_id()

    result = []
    for m in models:
        model_id = f"{m['type']}_{m['name']}_{m['latest_version']}"
        result.append(ModelInfo(
            model_id=model_id,
            model_type=m["type"],
            version=m["latest_version"] or "unknown",
            created_at=datetime.fromisoformat(m["created_at"]) if m["created_at"] else datetime.now(timezone.utc),
            metrics={},
            is_active=m["is_active"]
        ))
    return result


@app.post("/models/{model_id}/activate", tags=["Models"])
async def activate_model(model_id: str):
    """Activate a specific model."""
    global explainer

    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")

    try:
        model_type, model_name, version = _parse_model_id(model_id)
        model_registry.set_active(model_type, model_name, version)
        model = model_registry.load(model_type, model_name, version)
        explainer = AnomalyExplainer(model, feature_pipeline)

        return {"status": "activated", "model_id": model_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
