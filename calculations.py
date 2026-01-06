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
    if record is not None:  # If record already provided
        return record  # Return directly (skip state lookup)
    payload = get_candle_record(trading_symbol)  # Fetch from state
    if payload is None:  # If no data found
        return None
    if isinstance(payload, str):  # If payload is JSON string
        try:
            return json.loads(payload)  # Parse JSON to dict
        except json.JSONDecodeError:  # If JSON parsing fails
            return None  # Return None (invalid JSON)
    return payload  # Return dict directly


def _extract_rows(payload):
    """
    Extract candle rows from payload
    
    Args:
        payload: Candle data payload (dict)
    
    Returns:
        list: List of candle rows (dicts)
    
    Handles two payload formats:
        1. Direct: {"candles": [...]}
        2. Nested: {"data": {"data": {"candles": [...]}}}
    """
    rows = payload.get("candles")  # Try direct "candles" key
    if rows:  # If found and not empty
        return rows
    # Try nested structure
    raw = payload.get("data", {}).get("data", {}).get("candles", [])
    if raw:  # If found raw candles
        return normalize_candles(raw)  # Normalize and return
    return []  # Return empty list if no candles found


# ISO timestamp format constant for parsing
# Example: "2025-01-01T12:00:00+0530"
ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def _to_epoch_ms(timestamp_str):
    """
    Convert ISO timestamp string to epoch milliseconds
    
    Args:
        timestamp_str: ISO format timestamp string
    
    Returns:
        int: Epoch milliseconds (e.g., 1735718400000)
    
    Used for: Timestamp comparison in numeric format
    """
    dt = datetime.strptime(timestamp_str, ISO_TIMESTAMP_FORMAT)  # Parse ISO string
    return int(dt.timestamp() * 1000)  # Convert to milliseconds


def _filter_new_candles(
    rows, last_processed_ms
):
    """
    Filter to return only candles newer than last processed timestamp
    
    Args:
        rows: All candle rows
        last_processed_ms: Last processed timestamp in epoch milliseconds
    
    Returns:
        list: Only candles with timestamp > last_processed_ms
    
    Purpose: Avoid reprocessing already processed candles (deduplication)
    """
    if last_processed_ms is None:  # If first run (no previous timestamp)
        return rows  # Return all rows (all are new)
    fresh = []  # Collect fresh candles
    for row in rows:  # Iterate through all rows
        time_str = row.get("Time")  # Get timestamp string
        if not time_str:  # If Time field missing
            continue  # Skip this row
        try:
            ts_ms = _to_epoch_ms(time_str)  # Convert to milliseconds
        except ValueError:  # If parsing fails (invalid format)
            continue  # Skip this row
        if ts_ms <= last_processed_ms:  # If already processed
            continue  # Skip this row
        fresh.append(row)  # Add to fresh candles list
    return fresh  # Return only new candles


def run_calculation(
    trading_symbol,
    *,
    record=None,
):
    """
    Main calculation entry point - invoked after each candle fetch
    
    Args:
        trading_symbol: Symbol to process (e.g., "NIFTY")
        record: Optional pre-fetched candle record
    
    Process:
        1. Get current calculation state (last timestamp, count)
        2. Read candle payload
        3. Extract and filter new candles only
        4. Update state with new timestamp and count
    
    Note: Direct normalization is DISABLED here
          Normalization runs separately in background (market_fetcher.py)
          This design improves performance
    """
    calc_state = get_calculation_state(trading_symbol)  # Get current state

    # NORMALIZATION DISABLED - runs separately in background
    # from combined_normalization import normalize_all_index_options
    # normalize_all_index_options()

    payload = _read_candle_payload(trading_symbol, record=record)  # Get payload
    if payload is None:  # If no payload found
        log_error(f"[calculations] No candle payload found for {trading_symbol}")
        return  # Exit early

    rows = _extract_rows(payload)  # Extract candle rows
    last_processed_ms = calc_state.get("last_timestamp_ms")  # Get last timestamp

    fresh_rows = _filter_new_candles(rows, last_processed_ms)  # Filter new only
    if not fresh_rows:  # If no new candles
        if rows:  # But rows exist (all old)
            latest_time = rows[-1]["Time"]  # Get latest timestamp
            try:
                latest_ts_ms = _to_epoch_ms(latest_time)  # Convert to ms
            except ValueError:  # If parsing fails
                latest_ts_ms = last_processed_ms or 0  # Use previous or 0
            # Update state with latest timestamp (count stays same)
            update_calculation_state(
                trading_symbol,
                last_timestamp_ms=latest_ts_ms,
                last_timestamp=latest_time,
                processed_count=calc_state.get("processed_count", 0),
            )
        return  # Exit (no new data to process)

    # New candles exist - process them
    candle_count = len(fresh_rows)  # Count of new candles
    latest_time = fresh_rows[-1]["Time"]  # Latest timestamp (ISO)
    latest_ts_ms = _to_epoch_ms(latest_time)  # Convert to milliseconds

    # Calculate total processed count (previous + current)
    processed_total = calc_state.get("processed_count", 0) + candle_count

    # Update state with new values
    update_calculation_state(
        trading_symbol,
        last_timestamp_ms=latest_ts_ms,  # Latest timestamp in milliseconds
        last_timestamp=latest_time,  # Latest timestamp in ISO format
        processed_count=processed_total,  # Total candles processed
    )


def main():
    """
    CLI entry point for manual calculation execution
    
    Usage:
        python calculations.py NIFTY
        python calculations.py BANKNIFTY
    
    Process:
        Parse command line arguments and run calculation for specified symbol
    """
    parser = argparse.ArgumentParser(
        description="Run calculations for a stored candle payload."
    )
    parser.add_argument(
        "trading_symbol",
        help="Trading symbol whose candle payload should be processed "
        "(e.g. NIFTY or RELIANCE).",
    )
    args = parser.parse_args()  # Parse command line arguments

    run_calculation(args.trading_symbol)  # Run calculation


# Export public functions
__all__ = ["run_calculation"]


# Run main if executed directly (not imported as module)
if __name__ == "__main__":
    main()

# Define ISO timestamp format at module level (used by _to_epoch_ms)
ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
