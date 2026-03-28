"""
Stream Simulator for ForexGuard
Simulates real-time event streaming for testing and demonstration.
"""

import asyncio
import random
import time
from datetime import datetime
from typing import Optional, Callable, AsyncIterator
from dataclasses import dataclass

from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_synthetic import (
    UserProfile, generate_normal_event, generate_anomalous_event,
    NORMAL_IP_POOL, DEVICES, INSTRUMENTS
)


@dataclass
class StreamConfig:
    """Configuration for stream simulation."""
    events_per_second: float = 10.0
    anomaly_rate: float = 0.05
    num_users: int = 100
    burst_probability: float = 0.02
    burst_size: int = 20
    seed: int = 42


class StreamSimulator:
    """
    Simulates a real-time stream of forex brokerage events.
    Supports both synchronous and async iteration.
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        """
        Initialize the stream simulator.

        Args:
            config: Stream configuration
        """
        self.config = config or StreamConfig()
        random.seed(self.config.seed)

        # Create user pool
        self.users = self._create_users()
        self.anomalous_users = [u for u in self.users if u.is_anomalous]
        self.normal_users = [u for u in self.users if not u.is_anomalous]

        self.running = False
        self.event_count = 0
        self.anomaly_count = 0

    def _create_users(self) -> list[UserProfile]:
        """Create pool of simulated users."""
        users = []
        num_anomalous = int(self.config.num_users * self.config.anomaly_rate * 2)

        for i in range(self.config.num_users):
            user_id = f"USER_{str(i).zfill(5)}"
            is_anomalous = i < num_anomalous
            users.append(UserProfile(user_id, is_anomalous))

        return users

    def generate_event(self) -> dict:
        """Generate a single event."""
        timestamp = datetime.now()

        # Decide if this should be an anomaly
        is_anomaly = random.random() < self.config.anomaly_rate

        if is_anomaly and self.anomalous_users:
            user = random.choice(self.anomalous_users)
            event = generate_anomalous_event(user, timestamp)
            self.anomaly_count += 1
        else:
            user = random.choice(self.users)
            event = generate_normal_event(user, timestamp)

        self.event_count += 1
        return event

    def generate_burst(self) -> list[dict]:
        """Generate a burst of events (simulates sudden activity spike)."""
        events = []
        # Pick a user for focused burst
        user = random.choice(self.anomalous_users if self.anomalous_users else self.users)

        for _ in range(self.config.burst_size):
            timestamp = datetime.now()
            if user.is_anomalous:
                event = generate_anomalous_event(user, timestamp)
                self.anomaly_count += 1
            else:
                event = generate_normal_event(user, timestamp)
            self.event_count += 1
            events.append(event)

        return events

    async def stream(self, max_events: Optional[int] = None) -> AsyncIterator[dict]:
        """
        Async generator that yields events at configured rate.

        Args:
            max_events: Maximum events to generate (None = infinite)

        Yields:
            Event dictionaries
        """
        self.running = True
        interval = 1.0 / self.config.events_per_second

        event_num = 0
        while self.running and (max_events is None or event_num < max_events):
            # Check for burst
            if random.random() < self.config.burst_probability:
                logger.debug(f"Generating burst of {self.config.burst_size} events")
                for event in self.generate_burst():
                    yield event
                    event_num += 1
                    if max_events and event_num >= max_events:
                        break
            else:
                yield self.generate_event()
                event_num += 1

            await asyncio.sleep(interval)

        self.running = False

    def stream_sync(self, max_events: Optional[int] = None):
        """
        Synchronous generator that yields events at configured rate.

        Args:
            max_events: Maximum events to generate (None = infinite)

        Yields:
            Event dictionaries
        """
        self.running = True
        interval = 1.0 / self.config.events_per_second

        event_num = 0
        while self.running and (max_events is None or event_num < max_events):
            # Check for burst
            if random.random() < self.config.burst_probability:
                for event in self.generate_burst():
                    yield event
                    event_num += 1
                    if max_events and event_num >= max_events:
                        break
            else:
                yield self.generate_event()
                event_num += 1

            time.sleep(interval)

        self.running = False

    def stop(self):
        """Stop the stream."""
        self.running = False

    def get_stats(self) -> dict:
        """Get streaming statistics."""
        return {
            "total_events": self.event_count,
            "anomaly_count": self.anomaly_count,
            "anomaly_rate": self.anomaly_count / max(1, self.event_count),
            "num_users": len(self.users),
            "running": self.running
        }


async def run_simulation(
    config: Optional[StreamConfig] = None,
    max_events: int = 1000,
    callback: Optional[Callable[[dict], None]] = None
) -> list[dict]:
    """
    Run a streaming simulation.

    Args:
        config: Stream configuration
        max_events: Maximum events to process
        callback: Optional callback for each event

    Returns:
        List of all generated events
    """
    simulator = StreamSimulator(config)
    events = []

    logger.info(f"Starting stream simulation (max_events={max_events})...")

    async for event in simulator.stream(max_events):
        events.append(event)

        if callback:
            callback(event)

        if len(events) % 100 == 0:
            stats = simulator.get_stats()
            logger.info(f"Processed {stats['total_events']} events, {stats['anomaly_count']} anomalies")

    stats = simulator.get_stats()
    logger.info(f"Simulation complete: {stats['total_events']} events, {stats['anomaly_rate']*100:.2f}% anomaly rate")

    return events


if __name__ == "__main__":
    # Demo: run a short simulation
    async def demo():
        config = StreamConfig(
            events_per_second=50,
            anomaly_rate=0.05,
            num_users=50,
            seed=42
        )

        events = await run_simulation(config, max_events=500)

        print(f"\nGenerated {len(events)} events")
        anomalies = [e for e in events if e.get("is_anomaly")]
        print(f"Anomalies: {len(anomalies)} ({len(anomalies)/len(events)*100:.1f}%)")

    asyncio.run(demo())
