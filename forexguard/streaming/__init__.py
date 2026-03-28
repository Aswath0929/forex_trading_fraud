"""Streaming module for real-time event processing."""

from .simulator import StreamSimulator, run_simulation
from .processor import StreamProcessor

__all__ = ["StreamSimulator", "StreamProcessor", "run_simulation"]
