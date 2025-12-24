#!/usr/bin/env python3

import argparse
import json

from datetime import datetime

from candle_processing import normalize_candles
from logger import log_error
from state import (
    get_candle_record,
    get_calculation_state,
    get_numpy_candle_snapshot,
    update_calculation_state,
)


def get_normalized_data(index_name):
    """
    Get normalized data for an index (NIFTY, BANKNIFTY, SENSEX).
    Uses combined_normalization for all options merged together.
    
    Args:
        index_name: "NIFTY", "BANKNIFTY", or "SENSEX"
    
    Returns:
        dict with normalized arrays for all options of that index
    """
    from combined_normalization import get_normalized_index_data
    return get_normalized_index_data(index_name)


def normalize_all():
    """
    Normalize all index options (NIFTY, BANKNIFTY, SENSEX).
    Call this after all fetches are complete.
    
    Returns:
        dict with normalized data for all indices
    """
    from combined_normalization import normalize_all_index_options
    return normalize_all_index_options()


def _read_candle_payload(
    trading_symbol,
    *,
    record=None,
):
    """Fetch the stored candle JSON for the given trading symbol."""
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
