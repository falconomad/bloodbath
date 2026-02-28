"""Compatibility wrapper for Alpaca-related data/trading hooks.

The runtime currently routes market-data access through src.api.data_fetcher.
"""

from src.api.data_fetcher import get_alpaca_snapshot_features

__all__ = ["get_alpaca_snapshot_features"]
