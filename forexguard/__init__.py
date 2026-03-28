"""
ForexGuard - Real-Time Anomaly Detection for Forex Brokerages

A production-grade system for detecting suspicious user behavior
using unsupervised machine learning on streaming data.
"""

__version__ = "0.1.0"
__author__ = "ForexGuard Team"

from .config import get_config, Config

__all__ = ["get_config", "Config", "__version__"]
