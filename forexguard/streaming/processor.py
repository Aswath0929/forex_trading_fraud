"""
Stream Processor for ForexGuard
Processes streaming events through the anomaly detection pipeline.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque

import aiohttp
from loguru import logger

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.info("Kafka not available. Using in-memory processing only.")


@dataclass
class ProcessorConfig:
    """Configuration for stream processor."""
    api_url: str = "http://localhost:8000"
    batch_size: int = 10
    batch_timeout: float = 1.0
    max_concurrent: int = 5
    alert_threshold: float = 0.5

    # Kafka settings (optional)
    kafka_bootstrap: str = "localhost:9092"
    kafka_input_topic: str = "forex-events"
    kafka_output_topic: str = "forex-alerts"
    kafka_group_id: str = "forexguard-processor"


@dataclass
class ProcessingResult:
    """Result of processing an event."""
    event_id: str
    user_id: str
    anomaly_score: float
    is_anomaly: bool
    explanation: str
    processing_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class StreamProcessor:
    """
    Processes streaming events through the ForexGuard API.
    Supports both direct API calls and Kafka integration.
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """
        Initialize the stream processor.

        Args:
            config: Processor configuration
        """
        self.config = config or ProcessorConfig()
        self.results: deque[ProcessingResult] = deque(maxlen=10000)
        self.alert_callbacks: list[Callable[[ProcessingResult], None]] = []

        self._session: Optional[aiohttp.ClientSession] = None
        self._kafka_producer: Optional[Any] = None
        self._kafka_consumer: Optional[Any] = None
        self._running = False

        # Stats
        self.events_processed = 0
        self.anomalies_detected = 0
        self.errors = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def score_event(self, event: dict) -> Optional[ProcessingResult]:
        """
        Score a single event via the API.

        Args:
            event: Event dictionary

        Returns:
            Processing result or None if failed
        """
        start_time = datetime.utcnow()

        try:
            session = await self._get_session()
            url = f"{self.config.api_url}/score"

            async with session.post(url, json=event, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()

                    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

                    result = ProcessingResult(
                        event_id=data["event_id"],
                        user_id=data["user_id"],
                        anomaly_score=data["anomaly_score"],
                        is_anomaly=data["is_anomaly"],
                        explanation=data["explanation"],
                        processing_time_ms=processing_time
                    )

                    self.events_processed += 1
                    if result.is_anomaly:
                        self.anomalies_detected += 1
                        await self._handle_anomaly(result)

                    return result

                else:
                    error_text = await response.text()
                    logger.error(f"API error: {response.status} - {error_text}")
                    self.errors += 1
                    return None

        except asyncio.TimeoutError:
            logger.error("API request timed out")
            self.errors += 1
            return None
        except Exception as e:
            logger.error(f"Error scoring event: {e}")
            self.errors += 1
            return None

    async def score_batch(self, events: list[dict]) -> list[ProcessingResult]:
        """
        Score a batch of events via the API.

        Args:
            events: List of event dictionaries

        Returns:
            List of processing results
        """
        start_time = datetime.utcnow()

        try:
            session = await self._get_session()
            url = f"{self.config.api_url}/score/batch"

            async with session.post(url, json={"events": events}, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

                    results = []
                    for score in data["scores"]:
                        result = ProcessingResult(
                            event_id=score["event_id"],
                            user_id=score["user_id"],
                            anomaly_score=score["anomaly_score"],
                            is_anomaly=score["is_anomaly"],
                            explanation=score["explanation"],
                            processing_time_ms=processing_time / len(events)
                        )
                        results.append(result)

                        self.events_processed += 1
                        if result.is_anomaly:
                            self.anomalies_detected += 1
                            await self._handle_anomaly(result)

                    return results
                else:
                    self.errors += len(events)
                    return []

        except Exception as e:
            logger.error(f"Error scoring batch: {e}")
            self.errors += len(events)
            return []

    async def _handle_anomaly(self, result: ProcessingResult):
        """Handle detected anomaly."""
        self.results.append(result)

        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        # Publish to Kafka if available
        if self._kafka_producer:
            await self._publish_alert(result)

    async def _publish_alert(self, result: ProcessingResult):
        """Publish alert to Kafka."""
        if not self._kafka_producer:
            return

        alert = {
            "event_id": result.event_id,
            "user_id": result.user_id,
            "anomaly_score": result.anomaly_score,
            "explanation": result.explanation,
            "timestamp": result.timestamp.isoformat()
        }

        try:
            await self._kafka_producer.send_and_wait(
                self.config.kafka_output_topic,
                json.dumps(alert).encode("utf-8")
            )
        except Exception as e:
            logger.error(f"Failed to publish alert to Kafka: {e}")

    def on_alert(self, callback: Callable[[ProcessingResult], None]):
        """Register an alert callback."""
        self.alert_callbacks.append(callback)

    async def process_stream(
        self,
        events: list[dict],
        use_batching: bool = True
    ) -> list[ProcessingResult]:
        """
        Process a stream of events.

        Args:
            events: List of events to process
            use_batching: Whether to use batch processing

        Returns:
            List of all processing results
        """
        results = []

        if use_batching:
            # Process in batches
            for i in range(0, len(events), self.config.batch_size):
                batch = events[i:i + self.config.batch_size]
                batch_results = await self.score_batch(batch)
                results.extend(batch_results)

                if len(results) % 100 == 0:
                    logger.info(f"Processed {len(results)}/{len(events)} events")
        else:
            # Process individually with concurrency
            semaphore = asyncio.Semaphore(self.config.max_concurrent)

            async def process_one(event):
                async with semaphore:
                    return await self.score_event(event)

            tasks = [process_one(event) for event in events]
            results = await asyncio.gather(*tasks)
            results = [r for r in results if r is not None]

        return results

    async def start_kafka_consumer(self):
        """Start consuming from Kafka topic."""
        if not KAFKA_AVAILABLE:
            logger.error("Kafka is not available. Install aiokafka.")
            return

        self._kafka_consumer = AIOKafkaConsumer(
            self.config.kafka_input_topic,
            bootstrap_servers=self.config.kafka_bootstrap,
            group_id=self.config.kafka_group_id,
            auto_offset_reset="latest"
        )

        self._kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.config.kafka_bootstrap
        )

        await self._kafka_consumer.start()
        await self._kafka_producer.start()

        logger.info(f"Started Kafka consumer on topic: {self.config.kafka_input_topic}")

        self._running = True
        try:
            async for msg in self._kafka_consumer:
                if not self._running:
                    break

                try:
                    event = json.loads(msg.value.decode("utf-8"))
                    await self.score_event(event)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON in Kafka message")
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")

        finally:
            await self._kafka_consumer.stop()
            await self._kafka_producer.stop()

    async def stop(self):
        """Stop the processor."""
        self._running = False

        if self._session:
            await self._session.close()

        if self._kafka_consumer:
            await self._kafka_consumer.stop()
        if self._kafka_producer:
            await self._kafka_producer.stop()

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "events_processed": self.events_processed,
            "anomalies_detected": self.anomalies_detected,
            "anomaly_rate": self.anomalies_detected / max(1, self.events_processed),
            "errors": self.errors,
            "recent_anomalies": len(self.results)
        }


class KafkaAlertProducer:
    """
    Standalone Kafka producer for pushing alerts.
    Use this when you want to integrate alerts into an existing Kafka pipeline.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "forex-alerts"
    ):
        """
        Initialize Kafka alert producer.

        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Topic to publish alerts to
        """
        if not KAFKA_AVAILABLE:
            raise ImportError("aiokafka is required for Kafka integration")

        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self._producer: Optional[AIOKafkaProducer] = None

    async def start(self):
        """Start the producer."""
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        await self._producer.start()
        logger.info(f"Kafka producer started for topic: {self.topic}")

    async def send_alert(self, alert: dict):
        """Send an alert to Kafka."""
        if not self._producer:
            raise RuntimeError("Producer not started. Call start() first.")

        await self._producer.send_and_wait(self.topic, alert)

    async def stop(self):
        """Stop the producer."""
        if self._producer:
            await self._producer.stop()


if __name__ == "__main__":
    # Demo: Process a batch of simulated events

    async def demo():
        from simulator import StreamSimulator, StreamConfig

        # Create simulator
        sim_config = StreamConfig(events_per_second=100, num_users=50)
        simulator = StreamSimulator(sim_config)

        # Generate some events
        events = [simulator.generate_event() for _ in range(100)]

        # Create processor
        proc_config = ProcessorConfig(api_url="http://localhost:8000")
        processor = StreamProcessor(proc_config)

        # Register alert callback
        def on_alert(result):
            print(f"ALERT: User {result.user_id} - Score: {result.anomaly_score:.2f}")
            print(f"       {result.explanation}")

        processor.on_alert(on_alert)

        # Note: This requires the API to be running
        print("Processing events (requires API to be running)...")
        try:
            results = await processor.process_stream(events)
            print(f"\nProcessed {len(results)} events")
            print(f"Stats: {processor.get_stats()}")
        except aiohttp.ClientConnectorError:
            print("API not running. Start the API first with: python -m api.main")
        finally:
            await processor.stop()

    asyncio.run(demo())
