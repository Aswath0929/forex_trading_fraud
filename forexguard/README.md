# ForexGuard

**Real-Time User/Trader Anomaly Detection Engine for Forex Brokerages**

ForexGuard is a production-grade system for detecting suspicious user behavior in forex brokerage platforms using unsupervised machine learning on streaming data.

## Architecture

```
                                    ForexGuard Architecture

    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
    │   Event Stream  │────▶│  Feature Engine  │────▶│   ML Models     │
    │   (Kafka/API)   │     │  (Rolling Stats) │     │ (IF/LSTM-AE)    │
    └─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                              │
    ┌─────────────────┐     ┌──────────────────┐              │
    │   Alert System  │◀────│   Explainer      │◀─────────────┘
    │  (Kafka/WebUI)  │     │  (SHAP/Rules)    │
    └─────────────────┘     └──────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **Feature Pipeline** | Extracts 30+ behavioral features including rolling statistics, z-scores, and deviation metrics |
| **Isolation Forest** | Baseline unsupervised anomaly detector - fast, interpretable, O(n log n) |
| **LSTM Autoencoder** | Deep learning model for sequential pattern anomalies via reconstruction error |
| **Explainer** | SHAP-based and rule-based explanations for model decisions |
| **Alert Generator** | Human-readable alerts with severity levels and recommended actions |
| **REST API** | FastAPI endpoints for scoring, alerts, and model management |
| **Stream Processor** | Async event processing with optional Kafka integration |

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-org/forexguard.git
cd forexguard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Generate synthetic data and train both models
python train.py

# Train only Isolation Forest (faster)
python train.py --model isolation_forest --num-events 20000

# Train LSTM with more epochs
python train.py --model lstm --epochs 50
```

### 3. Start the API

```bash
# Run the API server
python -m api.main

# Or with uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Score Events

```bash
# Health check
curl http://localhost:8000/health

# Score a single event
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "evt_001",
    "user_id": "USER_00001",
    "timestamp": "2024-01-15T10:30:00Z",
    "event_type": "trade",
    "ip_address": "192.168.1.100",
    "device": "Windows Desktop",
    "instrument": "EUR/USD",
    "lot_size": 1.5,
    "direction": "buy",
    "margin_used": 750.0,
    "leverage": 100
  }'

# Explain the same event (shows in Swagger UI at /docs too)
curl -X POST "http://localhost:8000/explain/shap?top_k=10" \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "evt_001",
    "user_id": "USER_00001",
    "timestamp": "2024-01-15T10:30:00Z",
    "event_type": "trade",
    "ip_address": "192.168.1.100",
    "device": "Windows Desktop",
    "instrument": "EUR/USD",
    "lot_size": 1.5,
    "direction": "buy",
    "margin_used": 750.0,
    "leverage": 100
  }'
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model status |
| `/score` | POST | Score single event for anomalies |
| `/explain/shap` | POST | Top feature attributions (SHAP if available, else rule-based) |
| `/score/batch` | POST | Score multiple events |
| `/alerts` | GET | Get recent alerts |
| `/alerts/{id}/acknowledge` | POST | Acknowledge an alert |
| `/alerts/summary` | GET | Get alert statistics |
| `/models` | GET | List registered models |
| `/models/{id}/activate` | POST | Activate a specific model |

### Score Response

```json
{
  "event_id": "evt_001",
  "user_id": "USER_00001",
  "anomaly_score": 0.85,
  "is_anomaly": true,
  "threshold": 0.5,
  "model_type": "isolation_forest",
  "severity": "high",
  "feature_contributions": [
    {
      "feature_name": "lot_size_zscore",
      "value": 3.2,
      "contribution": 0.35,
      "description": "Lot size deviation (3.20 std from mean)"
    }
  ],
  "explanation": "High-risk anomaly detected due to unusual lot sizes, abnormal trade volume.",
  "timestamp": "2024-01-15T10:30:05Z"
}
```

## Feature Engineering

ForexGuard extracts 30+ features in three categories:

### Base Features
- Transaction amounts, lot sizes, margin usage
- Leverage levels, session durations
- Event type encodings (trade, deposit, withdrawal, etc.)

### Rolling Window Features (24h)
- Event counts, trade counts, deposit/withdrawal counts
- Unique IPs and devices
- Amount statistics (mean, std)
- Inter-event time gaps
- IP/device change rates

### Deviation Features
- Z-scores for amounts, lot sizes
- Activity spikes (>50 events/24h)
- Rapid switching flags
- Withdrawal/deposit ratios

## Models

### Isolation Forest (Recommended for Production)

**Why Isolation Forest?**
- No labeled data required (unsupervised)
- Fast training and inference - O(n log n)
- Handles mixed feature types well
- Interpretable anomaly scores
- Suitable for 5% anomaly rate scenarios

```python
from models.isolation_forest import IsolationForestDetector

detector = IsolationForestDetector(contamination=0.05, n_estimators=200)
detector.fit(X, feature_names)
scores = detector.predict_proba(X_test)
```

### LSTM Autoencoder (For Sequential Patterns)

**Why LSTM Autoencoder?**
- Captures temporal patterns in behavior sequences
- Learns complex non-linear relationships
- Reconstruction error as anomaly score
- Detects subtle pattern deviations

```python
from models.lstm_autoencoder import LSTMAutoencoderDetector

detector = LSTMAutoencoderDetector(input_dim=30, seq_len=10)
detector.fit(X_normal, epochs=50)  # Train on normal data only
scores = detector.score_samples(X_test)
```

## Streaming Simulation

### Without Kafka (Async Simulation)

```python
from streaming.simulator import StreamSimulator, StreamConfig
from streaming.processor import StreamProcessor

# Configure simulation
config = StreamConfig(
    events_per_second=10,
    anomaly_rate=0.05,
    num_users=100
)

# Run simulation
simulator = StreamSimulator(config)
processor = StreamProcessor()

async for event in simulator.stream(max_events=1000):
    result = await processor.score_event(event)
    if result.is_anomaly:
        print(f"ALERT: {result.explanation}")
```

### With Kafka

Kafka flow is:

`forex-events` topic → `streaming.kafka_worker` → `/score` API → `forex-alerts` topic

```bash
# 1) Start Kafka (and optional UI) via docker compose profile
docker-compose --profile kafka up -d

# 2) Start the ForexGuard API (in another terminal)
python -m api.main

# 3) Start the Kafka worker (consumes events, publishes alerts)
python -m streaming.kafka_worker

# 4) (Optional) Publish simulated events into Kafka
python -m streaming.kafka_producer --max-events 1000

# Env overrides (optional)
#   API_URL=http://localhost:8000
#   KAFKA_BOOTSTRAP=localhost:9092
#   KAFKA_INPUT_TOPIC=forex-events
#   KAFKA_OUTPUT_TOPIC=forex-alerts
#   KAFKA_GROUP_ID=forexguard-processor
```

## Deployment

### Docker

```bash
# Build image
docker build -t forexguard .

# Run API
docker run -p 8000:8000 forexguard

# Train model inside container
docker run -v $(pwd)/models:/app/models forexguard python train.py
```

### Docker Compose

```bash
# API only
docker-compose up -d api

# With Kafka
docker-compose --profile kafka up -d
```

### Cloud Deployment

#### Render
1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

#### HuggingFace Spaces
1. Create new Space with Docker SDK
2. Copy Dockerfile and code
3. Set environment variables

#### AWS (ECS/Fargate)
1. Push image to ECR
2. Create task definition with health check
3. Deploy to Fargate cluster

## Anomaly Types Detected

| Type | Description | Key Features |
|------|-------------|--------------|
| **IP Switching** | Rapid changes between IPs/VPNs | `unique_ips_24h`, `ip_change_rate` |
| **IP/Device Hub** | Many accounts share same IP/device | `ip_shared_users_24h`, `device_shared_users_24h`, `ip_hub_behavior` |
| **Login Brute Force** | Failed login streaks / high failure rate | `consecutive_login_failures`, `failed_login_rate`, `login_bruteforce` |
| **Bot-like Navigation** | Rapid portal navigation bursts | `page_view_count_24h`, `min_inter_event_time`, `rapid_navigation` |
| **Trade Spikes** | Sudden volume increases | `lot_size_zscore`, `trade_count_24h` |
| **Deposit/Withdrawal Abuse** | Money laundering patterns | `withdrawal_deposit_ratio`, `amount_zscore`, `dormancy_withdrawal` |
| **Session Hijacking** | Suspicious access switching | `unique_devices_24h`, `min_inter_event_time`, `rapid_device_switching` |

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | API server port |
| `LOG_LEVEL` | INFO | Logging level |
| `MODEL_PATH` | models/saved | Model storage path |
| `ANOMALY_THRESHOLD` | 0.5 | Score threshold for anomaly flagging + alerts |
| `ALERT_THRESHOLD` | 0.5 | (Stream processor) Threshold for publishing alerts (if used) |
| `KAFKA_BOOTSTRAP` | localhost:9092 | Kafka servers |
| `KAFKA_INPUT_TOPIC` | forex-events | Kafka topic for raw events |
| `KAFKA_OUTPUT_TOPIC` | forex-alerts | Kafka topic for alerts |
| `KAFKA_GROUP_ID` | forexguard-processor | Kafka consumer group id |
| `API_URL` | http://localhost:8000 | API base URL used by Kafka worker |

## Trade-offs & Design Decisions

### Model Selection
- **Isolation Forest** chosen as primary model for speed and interpretability
- **LSTM** available for scenarios requiring sequential pattern detection
- Both are unsupervised - no labeled fraud data required

### Feature Store
- In-memory rolling window (24h default)
- Trade-off: Memory usage vs. feature richness
- Max 1000 events per user to prevent memory issues

### Real-time vs Batch
- API supports both single-event and batch scoring
- Batch is 3-5x faster for bulk processing
- Single-event for true real-time (sub-100ms latency)

### Explainability
- SHAP for tree-based models (when available)
- Rule-based fallback for LSTM and when SHAP fails
- Always produces human-readable explanations

## Project Structure

```
forexguard/
├── api/
│   ├── main.py           # FastAPI application
│   ├── schemas.py        # Pydantic models
│   └── alerts.py         # Alert generation
├── data/
│   └── generate_synthetic.py  # Data generator
├── features/
│   ├── feature_pipeline.py    # Feature extraction
│   └── feature_store.py       # Rolling window store
├── models/
│   ├── isolation_forest.py    # IF detector
│   ├── lstm_autoencoder.py    # LSTM-AE detector
│   └── model_registry.py      # Model versioning
├── explainability/
│   └── explainer.py      # SHAP/rule explanations
├── streaming/
│   ├── simulator.py      # Event stream simulation
│   └── processor.py      # Stream processing
├── train.py              # Training script
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Testing

```bash
# Run tests
pytest tests/ -v

# Test API endpoints
pytest tests/test_api.py -v

# Test with coverage
pytest --cov=. tests/
```

## Performance

Benchmarks on synthetic 50K event dataset:

| Metric | Isolation Forest | LSTM Autoencoder |
|--------|-----------------|------------------|
| Training Time | ~5 seconds | ~2 minutes |
| Inference (single) | <5ms | <10ms |
| Inference (batch 100) | <50ms | <100ms |
| Precision | ~75% | ~70% |
| Recall | ~80% | ~75% |

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

---

Built with FastAPI, PyTorch, scikit-learn, and SHAP.

cd forexguard
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn api.main:app --reload