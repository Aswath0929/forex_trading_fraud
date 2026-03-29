"""Kafka worker for ForexGuard.

Consumes raw events from Kafka, scores them through the ForexGuard API, and publishes
anomaly alerts back to Kafka.

Run (from forexguard/):
  python -m streaming.kafka_worker

Env vars:
  API_URL, KAFKA_BOOTSTRAP, KAFKA_INPUT_TOPIC, KAFKA_OUTPUT_TOPIC, KAFKA_GROUP_ID
"""

import asyncio
from loguru import logger

from .processor import StreamProcessor, ProcessorConfig


async def main():
    cfg = ProcessorConfig()
    logger.info(
        "Starting Kafka worker | api_url={} | bootstrap={} | in={} | out={} | group={}",
        cfg.api_url,
        cfg.kafka_bootstrap,
        cfg.kafka_input_topic,
        cfg.kafka_output_topic,
        cfg.kafka_group_id,
    )

    processor = StreamProcessor(cfg)

    try:
        await processor.start_kafka_consumer()
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        await processor.stop()
        logger.info("Kafka worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
