#!/usr/bin/env python3
# Shebang line - Python 3 से script run करने के लिए

"""
CALCULATIONS MODULE (कैलकुलेशन मॉड्यूल)
=========================================

Purpose: Candle data processing और calculation state management
यह module fetched candle data को process करता है और track करता है कि कौनसे candles already processed हैं

Key Responsibilities:
1. New candles को identify करना (duplicates avoid करने के लिए)
2. Calculation state maintain करना (last processed timestamp, count)
3. Combined normalization को trigger करना (currently disabled)

Flow:
market_fetcher.py → run_calculation() → candle processing → state update

Note: Direct normalization यहाँ disabled है, separate normalize_all_index_options() use करो
"""

import argparse  # Command line arguments parse करने के लिए
import json  # JSON data parse करने के लिए

from datetime import datetime  # Timestamp parsing के लिए

# Related modules से functions import
from candle_processing import normalize_candles  # Candles को normalize format में convert करने के लिए
from logger import log_error  # Error logging के लिए
from state import (  # State management functions
    get_candle_record,  # Trading symbol के लिए stored candle record get करने के लिए
    get_calculation_state,  # Calculation state (last timestamp, count) get करने के लिए
    get_numpy_candle_snapshot,  # NumPy array snapshot get करने के लिए (not used here)
    update_calculation_state,  # Calculation state update करने के लिए
)


def get_normalized_data(index_name):
    """
    Index (NIFTY, BANKNIFTY, SENSEX) के लिए normalized data get करो
    
    Purpose: Combined normalization results को fetch करना (सभी options merged)
    
    Args:
        index_name: "NIFTY", "BANKNIFTY", या "SENSEX"
    
    Returns:
        dict: Normalized arrays सभी options के लिए merged
              Contains: time_seconds, normalized columns (iv_diff_cumsum, oi_diff_cumsum, etc.)
    
    Note: यह wrapper function है जो actually combined_normalization module call करती है
    
    Example:
        normalized = get_normalized_data("NIFTY")
        # normalized = {
        #     "DEC24_24000CE_iv_diff_cumsum": [...],
        #     "DEC24_24000PE_iv_diff_cumsum": [...],
        #     ...
        # }
    """
    from combined_normalization import get_normalized_index_data  # Dynamic import (lazy loading)
    return get_normalized_index_data(index_name)  # Combined normalization से data fetch करो


def normalize_all():
    """
    सभी index options (NIFTY, BANKNIFTY, SENSEX) को normalize करो
    
    Purpose: सभी fetches complete होने के बाद एक साथ normalization run करना
    
    Returns:
        dict: सभी indices के लिए normalized data
              Structure: {
                  "NIFTY": {...normalized data...},
                  "BANKNIFTY": {...normalized data...},
                  "SENSEX": {...normalized data...}
              }
    
    When to call: Market hours के दौरान हर fetch cycle के बाद (market_fetcher.py से)
    
    Note: यह CPU-intensive operation है, इसलिए carefully schedule करो
    
    Example:
        all_normalized = normalize_all()
        nifty_data = all_normalized["NIFTY"]
    """
    from combined_normalization import normalize_all_index_options  # Dynamic import
    return normalize_all_index_options()  # सभी indices को normalize करो


def _read_candle_payload(
    trading_symbol,
    *,
    record=None,
):
    """
    Trading symbol के लिए stored candle JSON fetch करो
    
    Purpose: State से candle data retrieve करना (या directly provided record use करना)
    
    Args:
        trading_symbol: Trading symbol (e.g., "NIFTY", "BANKNIFTY")
        record: Optional - directly provided record (state check skip करने के लिए)
    
    Returns:
        dict or None: Candle payload (parsed JSON), या None अगर not found
    
    Process:
        1. अगर record already provided है तो directly return करो
        2. नहीं तो state से get_candle_record() call करो
        3. अगर payload string है तो JSON parse करो
        4. Parse errors handle करो
    
    Example:
        payload = _read_candle_payload("NIFTY")
        if payload:
            candles = payload.get("candles", [])
    """
    if record is not None:  # अगर record already function parameter में दिया गया है
        return record  # Directly return करो (state lookup skip)
    
    payload = get_candle_record(trading_symbol)  # State से stored payload fetch करो
    if payload is None:  # अगर कुछ stored नहीं है
        return None  # None return करो
    
    if isinstance(payload, str):  # अगर payload string format में है (JSON string)
        try:
            return json.loads(payload)  # JSON string को dict में parse करो
        except json.JSONDecodeError:  # अगर parsing fail हो जाए (invalid JSON)
            return None  # None return करो (error silently handle)
    
    return payload  # अगर already dict है तो directly return करो


def _extract_rows(payload):
    """
    Payload से candle rows extract करो (और normalize करो if needed)
    
    Purpose: Different payload formats को handle करके consistent candles list return करना
    
    Args:
        payload: Candle data payload (dict)
    
    Returns:
        list: Candle rows (list of dicts)
              Each row: {"Time": "...", "Open": ..., "High": ..., ...}
    
    Payload formats handled:
        1. Direct format: {"candles": [...]}
        2. Nested format: {"data": {"data": {"candles": [...]}}}
    
    Process:
        1. पहले direct "candles" key check करो
        2. अगर मिल गया तो return करो
        3. नहीं तो nested structure में search करो
        4. अगर raw format में candles हैं तो normalize_candles() से process करो
    
    Example:
        payload = {"candles": [{"Time": "...", "Open": 100, ...}, ...]}
        rows = _extract_rows(payload)
        # rows = [{"Time": "...", "Open": 100, ...}, ...]
    """
    rows = payload.get("candles")  # Direct "candles" key से rows get करने की कोशिश
    if rows:  # अगर rows मिल गए (और empty नहीं हैं)
        return rows  # Directly return करो
    
    # Nested structure में search करो: payload["data"]["data"]["candles"]
    raw = payload.get("data", {}).get("data", {}).get("candles", [])
    if raw:  # अगर nested candles मिले
        return normalize_candles(raw)  # Raw candles को normalize format में convert करो
    
    return []  # अगर कहीं भी candles नहीं मिले तो empty list return करो


# ISO timestamp format constant (timestamp parsing के लिए)
# Example: "2025-01-01T12:00:00+0530"
ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def _to_epoch_ms(timestamp_str):
    """
    ISO format timestamp string को epoch milliseconds में convert करो
    
    Purpose: Timestamp comparison के लिए numeric format चाहिए (milliseconds since epoch)
    
    Args:
        timestamp_str: ISO format timestamp (e.g., "2025-01-01T12:00:00+0530")
    
    Returns:
        int: Epoch milliseconds (e.g., 1735718400000)
    
    Process:
        1. ISO string को datetime object में parse करो
        2. Datetime को Unix timestamp में convert करो
        3. Seconds को milliseconds में multiply करो (*1000)
        4. Integer return करो
    
    Example:
        ms = _to_epoch_ms("2025-01-01T12:00:00+0530")
        # ms = 1735718400000
    """
    dt = datetime.strptime(timestamp_str, ISO_TIMESTAMP_FORMAT)  # ISO string से datetime बनाओ
    return int(dt.timestamp() * 1000)  # Timestamp (seconds) को milliseconds में convert करके return करो


def _filter_new_candles(
    rows, last_processed_ms
):
    """
    केवल new candles return करो (duplicates filter out करने के लिए)
    
    Purpose: पहले से processed candles को skip करना (redundant processing avoid करने के लिए)
    
    Args:
        rows: सभी candle rows (list of dicts)
        last_processed_ms: Last processed timestamp (epoch milliseconds), या None if first run
    
    Returns:
        list: Fresh candles (जो last_processed_ms के बाद के हैं)
    
    Process:
        1. अगर last_processed_ms None है (first run) तो सभी rows return करो
        2. हर row का timestamp extract और parse करो
        3. केवल वे rows collect करो जिनका timestamp > last_processed_ms
        4. Parse errors silently handle करो (invalid rows skip)
    
    Why needed: API duplicate data भी return कर सकता है, redundant processing avoid करना चाहिए
    
    Example:
        all_rows = [
            {"Time": "2025-01-01T09:15:00+0530", ...},
            {"Time": "2025-01-01T09:16:00+0530", ...},  # new
            {"Time": "2025-01-01T09:17:00+0530", ...},  # new
        ]
        last_ms = _to_epoch_ms("2025-01-01T09:15:00+0530")
        fresh = _filter_new_candles(all_rows, last_ms)
        # fresh = last 2 rows only
    """
    if last_processed_ms is None:  # अगर यह first run है (कोई previous timestamp नहीं)
        return rows  # सभी rows return करो (सब new हैं)
    
    fresh = []  # Fresh candles collect करने के लिए empty list
    for row in rows:  # हर row को iterate करो
        time_str = row.get("Time")  # Timestamp string extract करो
        if not time_str:  # अगर Time field missing है
            continue  # Skip this row
        try:
            ts_ms = _to_epoch_ms(time_str)  # Timestamp को milliseconds में convert करो
        except ValueError:  # अगर parsing fail हो जाए (invalid format)
            continue  # Skip this row (silently handle error)
        if ts_ms <= last_processed_ms:  # अगर यह candle already processed है
            continue  # Skip this row (old data)
        fresh.append(row)  # Fresh row को list में add करो
    
    return fresh  # सभी fresh candles return करो


def run_calculation(
    trading_symbol,
    *,
    record=None,
):
    """
    Candle fetch के बाद calculation run करो (main entry point)
    
    Purpose: हर candle fetch के बाद invoke होने वाला function जो calculation state update करता है
    
    Args:
        trading_symbol: Trading symbol (e.g., "NIFTY", "BANKNIFTY")
        record: Optional - directly provided candle record (state lookup skip करने के लिए)
    
    Process:
        1. Current calculation state fetch करो (last timestamp, processed count)
        2. Candle payload read करो
        3. Candle rows extract करो
        4. Only fresh (new) candles filter करो
        5. अगर fresh candles हैं तो state update करो
        6. अगर कोई fresh candles नहीं हैं तो भी latest timestamp update करो
    
    Note: NORMALIZATION DISABLED यहाँ पर!
          - पहले यहाँ direct normalization था
          - अब separate normalize_all_index_options() background में run होता है
          - यह design change performance improve करने के लिए किया गया
    
    State Updates:
        - last_timestamp_ms: Latest processed candle का timestamp (epoch ms)
        - last_timestamp: Latest processed candle का ISO string
        - processed_count: Total processed candles count
    
    Called by: market_fetcher.py में हर fetch के बाद
    
    Example:
        # market_fetcher.py से:
        record = {
            "trading_symbol": "NIFTY",
            "candles": [...],
            ...
        }
        run_calculation("NIFTY", record=record)
    """
    # Current calculation state fetch करो (last processed timestamp और count)
    calc_state = get_calculation_state(trading_symbol)

    # === NORMALIZATION DISABLED ===
    # पहले यहाँ direct normalization call था:
    # from combined_normalization import normalize_all_index_options
    # normalize_all_index_options()
    # 
    # अब background में separately run होता है (market_fetcher.py में)
    # Reason: Performance - हर fetch पर heavy normalization avoid करना

    # Candle payload read करो (state से या provided record से)
    payload = _read_candle_payload(trading_symbol, record=record)
    if payload is None:  # अगर कोई payload नहीं मिला
        log_error(f"[calculations] No candle payload found for {trading_symbol}")  # Error log करो
        return  # Early return (nothing to process)

    # Candle rows extract करो (different formats handle करके)
    rows = _extract_rows(payload)
    
    # Last processed timestamp fetch करो (state से)
    last_processed_ms = calc_state.get("last_timestamp_ms")

    # केवल fresh (new) candles filter करो
    fresh_rows = _filter_new_candles(rows, last_processed_ms)
    
    if not fresh_rows:  # अगर कोई fresh candles नहीं हैं
        # Even if no fresh candles, latest timestamp update कर दो (consistency के लिए)
        if rows:  # अगर कोई rows exist करती हैं (भले ही all old)
            latest_time = rows[-1]["Time"]  # Last row का timestamp get करो
            try:
                latest_ts_ms = _to_epoch_ms(latest_time)  # Milliseconds में convert करो
            except ValueError:  # अगर parsing fail हो
                latest_ts_ms = last_processed_ms or 0  # Fallback to previous या 0
            # State update करो (count same रहेगी)
            update_calculation_state(
                trading_symbol,
                last_timestamp_ms=latest_ts_ms,  # Latest timestamp update
                last_timestamp=latest_time,  # ISO string भी update
                processed_count=calc_state.get("processed_count", 0),  # Count same
            )
        return  # Early return (no fresh data to process)

    # Fresh candles exist करती हैं, process करो
    candle_count = len(fresh_rows)  # Fresh candles count
    latest_time = fresh_rows[-1]["Time"]  # Last fresh candle का timestamp (ISO)
    latest_ts_ms = _to_epoch_ms(latest_time)  # Milliseconds में convert करो

    # Total processed count calculate करो (previous + current)
    processed_total = calc_state.get("processed_count", 0) + candle_count

    # State update करो with new values
    update_calculation_state(
        trading_symbol,
        last_timestamp_ms=latest_ts_ms,  # Latest processed timestamp (epoch ms)
        last_timestamp=latest_time,  # Latest processed timestamp (ISO string)
        processed_count=processed_total,  # Total candles processed so far
    )


def main():
    """
    CLI entry point - command line से manually calculation run करने के लिए
    
    Purpose: Testing/debugging के लिए - stored candle payload पर calculation manually trigger करना
    
    Usage:
        python calculations.py NIFTY
        python calculations.py BANKNIFTY
    
    Args (via command line):
        trading_symbol: Trading symbol whose candle payload should be processed
    
    Example:
        $ python calculations.py NIFTY
        # NIFTY के stored candles पर calculation run होगी
    """
    # Command line argument parser setup करो
    parser = argparse.ArgumentParser(
        description="Run calculations for a stored candle payload."  # Script description
    )
    # Trading symbol argument add करो (required positional argument)
    parser.add_argument(
        "trading_symbol",
        help="Trading symbol whose candle payload should be processed "
        "(e.g. NIFTY or RELIANCE).",  # Help text
    )
    args = parser.parse_args()  # Command line arguments parse करो

    run_calculation(args.trading_symbol)  # Calculation run करो provided symbol के लिए


# Export किए जाने वाले public functions
__all__ = ["run_calculation"]


# यह check करता है कि script directly run हो रही है (not imported as module)
if __name__ == "__main__":
    main()  # main() function call करो

# ISO timestamp format को file के end में define किया गया है (hoisting के लिए)
ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
