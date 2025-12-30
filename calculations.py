#!/usr/bin/env python3

"""
Calculations Module - Candle Processing and State Management
Handles calculation state tracking and determines which candles are new vs already processed
"""

import argparse  # For command line argument parsing
import json  # For JSON parsing

from datetime import datetime  # For timestamp handling

# Import related modules
from candle_processing import normalize_candles  # For normalizing raw candle data
from logger import log_error  # For error logging
from state import (  # State management functions
    get_candle_record,  # Get stored candle data for a symbol
    get_calculation_state,  # Get calculation state (timestamps, counts)
    get_numpy_candle_snapshot,  # Get NumPy array snapshot (not used here)
    update_calculation_state,  # Update calculation state
)


def get_normalized_data(index_name):
    """
    Get normalized data for a specific index
    
    Args:
        index_name: Index name - "NIFTY", "BANKNIFTY", or "SENSEX"
    
    Returns:
        dict: Normalized arrays for all options of that index merged together
    
    Usage:
        normalized = get_normalized_data("NIFTY")
    
    Note: This is a wrapper that delegates to combined_normalization module
    """
    from combined_normalization import get_normalized_index_data  # Lazy import
    return get_normalized_index_data(index_name)


def normalize_all():
    """
    Normalize all index options (NIFTY, BANKNIFTY, SENSEX) together
    
    Returns:
        dict: Normalized data for all indices
    
    When to call: After all fetch cycles are complete during market hours
    
    Usage:
        all_normalized = normalize_all()
    
    Note: CPU-intensive operation, should be called strategically
    """
    from combined_normalization import normalize_all_index_options  # Lazy import
    return normalize_all_index_options()


def _read_candle_payload(
    trading_symbol,
    *,
    record=None,
):
    """
    Fetch stored candle payload for a trading symbol
    
    Args:
        trading_symbol: Symbol to fetch data for (e.g., "NIFTY")
        record: Optional pre-fetched record to use instead of state lookup
    
    Returns:
        dict or None: Candle payload, or None if not found
    
    Process:
        1. If record is provided, return it directly
        2. Otherwise fetch from state
        3. Parse JSON string if needed
        4. Handle parse errors gracefully
    """
    if record is not None:
        return record
    payload = get_candle_record(trading_symbol)
    if payload is None:
        return None
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    return payload


def _extract_rows(payload):
    rows = payload.get("candles")
    if rows:
        return rows
    raw = payload.get("data", {}).get("data", {}).get("candles", [])
    if raw:
        return normalize_candles(raw)
    return []


def _to_epoch_ms(timestamp_str):
    dt = datetime.strptime(timestamp_str, ISO_TIMESTAMP_FORMAT)
    return int(dt.timestamp() * 1000)


def _filter_new_candles(
    rows, last_processed_ms
):
    """Return only candles that are newer than the stored marker."""
    if last_processed_ms is None:
        return rows
    fresh = []
    for row in rows:
        time_str = row.get("Time")
        if not time_str:
            continue
        try:
            ts_ms = _to_epoch_ms(time_str)
        except ValueError:
            continue
        if ts_ms <= last_processed_ms:
            continue
        fresh.append(row)
    return fresh


def run_calculation(
    trading_symbol,
    *,
    record=None,
):
    """
    Entry point invoked after each candle fetch.
    Normalization DISABLED - use combined_normalization.py when needed.
    """
    calc_state = get_calculation_state(trading_symbol)

    # NORMALIZATION DISABLED - आप combined_normalization.py use करोगे जब जरूरत हो
    # from combined_normalization import normalize_all_index_options
    # normalize_all_index_options()

    payload = _read_candle_payload(trading_symbol, record=record)
    if payload is None:
        log_error(f"[calculations] No candle payload found for {trading_symbol}")
        return

    rows = _extract_rows(payload)
    last_processed_ms = calc_state.get("last_timestamp_ms")

    fresh_rows = _filter_new_candles(rows, last_processed_ms)
    if not fresh_rows:
        if rows:
            latest_time = rows[-1]["Time"]
            try:
                latest_ts_ms = _to_epoch_ms(latest_time)
            except ValueError:
                latest_ts_ms = last_processed_ms or 0
            update_calculation_state(
                trading_symbol,
                last_timestamp_ms=latest_ts_ms,
                last_timestamp=latest_time,
                processed_count=calc_state.get("processed_count", 0),
            )
        return

    candle_count = len(fresh_rows)
    latest_time = fresh_rows[-1]["Time"]
    latest_ts_ms = _to_epoch_ms(latest_time)

    processed_total = calc_state.get("processed_count", 0) + candle_count

    update_calculation_state(
        trading_symbol,
        last_timestamp_ms=latest_ts_ms,
        last_timestamp=latest_time,
        processed_count=processed_total,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run calculations for a stored candle payload."
    )
    parser.add_argument(
        "trading_symbol",
        help="Trading symbol whose candle payload should be processed "
        "(e.g. NIFTY or RELIANCE).",
    )
    args = parser.parse_args()

    run_calculation(args.trading_symbol)


__all__ = ["run_calculation"]


if __name__ == "__main__":
    main()
ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
