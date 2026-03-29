"""Kafka producer for ForexGuard demo.

Publishes simulated events to the Kafka input topic.

Run (from forexguard/):
  python -m streaming.kafka_producer --max-events 500

Env vars:
  KAFKA_BOOTSTRAP, KAFKA_INPUT_TOPIC
"""

import argparse
import asyncio
import json
import os

from loguru import logger

try:
    from aiokafka import AIOKafkaProducer
except ImportError as e:  # pragma: no cover
    raise SystemExit("aiokafka is required for Kafka producer. Install requirements.txt") from e

from .simulator import StreamSimulator, StreamConfig


async def run(max_events: int, eps: float, anomaly_rate: float, num_users: int, seed: int):
    bootstrap = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    topic = os.getenv("KAFKA_INPUT_TOPIC", "forex-events")

    sim = StreamSimulator(
        StreamConfig(
            events_per_second=eps,
            anomaly_rate=anomaly_rate,
            num_users=num_users,
            seed=seed,
        )
    )

    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
    await producer.start()

    try:
        sent = 0
        async for event in sim.stream(max_events=max_events):
            payload = json.dumps(event, default=str).encode("utf-8")
            await producer.send_and_wait(topic, payload)
            sent += 1

            if sent % 100 == 0:
                logger.info("Sent {} events to topic '{}'", sent, topic)

        logger.info("Done. Sent {} events", sent)

    finally:
        await producer.stop()


def main():
    parser = argparse.ArgumentParser(description="Publish simulated ForexGuard events to Kafka")
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--eps", type=float, default=50.0, help="Events per second")
    parser.add_argument("--anomaly-rate", type=float, default=0.05)
    parser.add_argument("--num-users", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    asyncio.run(run(args.max_events, args.eps, args.anomaly_rate, args.num_users, args.seed))


if __name__ == "__main__":
    main()
