#!/usr/bin/env python3

"""
FUTURES DATA - Simple EMA Smoothing (NO Normalization)

Strategy:
1. Get raw fut_spot_diff (NOT cumsum) and oi_diff_cumsum
2. Apply simple EMA for smoothing only
3. Return smoothed values (NO scaling/normalization)

Futures: NIFTY FUT, BANKNIFTY FUT + 11 Stock Futures
Columns: fut_spot_diff (EMA smooth), oi_diff_cumsum (EMA smooth)
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from state import get_numpy_candle_snapshot, get_trading_catalog, get_calculation_state, update_calculation_state
from logger import log_info, log_error

# Futures to process
INDEX_FUTURES = ["NIFTY", "BANKNIFTY"]  # SENSEX has no futures

EQUITY_FUTURES = [
    "RELIANCE",
    "TCS", 
    "HDFCBANK",
    "ICICIBANK",
    "AXISBANK",
    "INFY",
    "BHARTIARTL",
    "ITC",
    "M&M",
    "SBIN",
    "LT",
]

ALL_FUTURES = INDEX_FUTURES + EQUITY_FUTURES

# EMA span for smoothing
EMA_SPAN = 5


def _get_futures_from_catalog(catalog: List[dict]) -> Dict[str, str]:
    """
    Get futures trading symbols from catalog.
    Returns: {"NIFTY": "NIFTY25JANFUT", "RELIANCE": "RELIANCE25JANFUT", ...}
    
    Only gets the EARLIEST expiry (current month) for each underlying.
    """
    futures_map = {}
    
    for entry in catalog:
        category = entry.get("category", "")
        if category not in ("index_future", "equity_future"):
            continue
        
        label = entry.get("label", "")
        trading_symbol = entry.get("trading_symbol", "")
        expiry = entry.get("expiry", 0)
        
        if not label or not trading_symbol:
            continue
        
        # Only keep earliest expiry per label
        if label not in futures_map:
            futures_map[label] = {"trading_symbol": trading_symbol, "expiry": expiry}
        else:
            # Keep the one with earlier expiry
            if expiry < futures_map[label]["expiry"]:
                futures_map[label] = {"trading_symbol": trading_symbol, "expiry": expiry}
    
    # Return just trading_symbol
    return {label: info["trading_symbol"] for label, info in futures_map.items()}


def _get_base_time_seconds() -> Optional[np.ndarray]:
    """
    Get time_seconds from NIFTY spot (most reliable base).
    """
    snapshot = get_numpy_candle_snapshot("NIFTY")
    if not snapshot:
        # Fallback to BANKNIFTY
        snapshot = get_numpy_candle_snapshot("BANKNIFTY")
    
    if not snapshot:
        return None
    
    time_seconds = snapshot.get("time_seconds")
    if time_seconds is None:
        return None
    
    size = snapshot.get("size", len(time_seconds))
    return time_seconds[:size]


def _apply_ema_incremental(
    arr: np.ndarray, 
    existing_ema: Optional[np.ndarray], 
    processed_size: int,
    span: int = EMA_SPAN
) -> np.ndarray:
    """
    Apply EMA incrementally - only calculate new values.
    Reuses previous EMA values, calculates only from processed_size onwards.
    
    Args:
        arr: Full raw data array
        existing_ema: Previously calculated EMA (can be None for first run)
        processed_size: How many values were already processed
        span: EMA span for smoothing
    
    Returns:
        Full EMA array (cached + new)
    """
    if arr is None or len(arr) == 0:
        return arr
    
    n = len(arr)
    alpha = 2.0 / (span + 1)
    
    # Initialize output
    out = np.full(n, np.nan, dtype=np.float32)
    
    # Copy existing EMA values
    if existing_ema is not None and processed_size > 0:
        copy_len = min(processed_size, len(existing_ema), n)
        out[:copy_len] = existing_ema[:copy_len]
    
    # Find where to start calculating
    if processed_size > 0 and existing_ema is not None:
        # Start from where we left off
        start_idx = processed_size
        # Get last valid EMA as starting point
        last_ema = None
        for i in range(processed_size - 1, -1, -1):
            if i < len(existing_ema) and not np.isnan(existing_ema[i]):
                last_ema = existing_ema[i]
                break
    else:
        # First run - start from beginning
        start_idx = 0
        last_ema = None
    
    # Calculate EMA for new values only
    for i in range(start_idx, n):
        val = arr[i]
        if np.isnan(val):
            # Forward fill: use last EMA
            out[i] = last_ema if last_ema is not None else np.nan
        else:
            if last_ema is None:
                # First valid value
                out[i] = val
            else:
                # EMA formula: α × current + (1-α) × previous
                out[i] = alpha * val + (1 - alpha) * last_ema
            last_ema = out[i]
    
    return out


def normalize_all_futures() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Process all futures data - Apply EMA smoothing only (NO normalization).
    
    Returns: {
        "NIFTY": {
            "fut_spot_diff": [...],  # EMA smoothed (actual diff, not cumsum)
            "oi_diff_cumsum": [...],  # EMA smoothed cumsum
        },
        "RELIANCE": {...},
        ...
    }
    """
    catalog = get_trading_catalog()
    if not catalog:
        return {}
    
    # Get base time from NIFTY spot
    base_time = _get_base_time_seconds()
    if base_time is None or len(base_time) == 0:
        log_error("[FuturesData] No base time_seconds available!")
        return {}
    
    n_rows = len(base_time)
    
    # Get futures trading symbols from catalog
    futures_map = _get_futures_from_catalog(catalog)
    
    # Get cached state for incremental EMA
    state_key = "FUTURES_EMA"
    calc_state = get_calculation_state(state_key)
    existing_ema = calc_state.get("ema", {})
    processed_size = calc_state.get("processed_size", 0)
    
    # Check if we have new data
    new_rows = n_rows - processed_size
    
    results = {}
    
    for label in ALL_FUTURES:
        trading_symbol = futures_map.get(label)
        if not trading_symbol:
            continue
        
        snapshot = get_numpy_candle_snapshot(trading_symbol)
        if not snapshot:
            continue
        
        fut_time = snapshot.get("time_seconds")
        if fut_time is None:
            continue
        
        size = snapshot.get("size", len(fut_time))
        fut_time = fut_time[:size]
        
        # Get raw data (NOT cumsum - actual diff values)
        fut_spot_diff_raw = snapshot.get("future_spot_diff")  # Actual diff, not cumsum
        oi_diff_raw = snapshot.get("oi_diff_cumsum")  # Cumsum for OI
        
        if fut_spot_diff_raw is None and oi_diff_raw is None:
            continue
        
        # Align to base time
        time_to_idx = {t: i for i, t in enumerate(base_time)}
        
        # Align fut_spot_diff
        fut_spot_aligned = np.full(n_rows, np.nan, dtype=np.float32)
        if fut_spot_diff_raw is not None:
            fut_spot_diff_raw = fut_spot_diff_raw[:size]
            for i, t in enumerate(fut_time):
                if t in time_to_idx:
                    fut_spot_aligned[time_to_idx[t]] = fut_spot_diff_raw[i]
        
        # Align oi_diff_cumsum
        oi_aligned = np.full(n_rows, np.nan, dtype=np.float32)
        if oi_diff_raw is not None:
            oi_diff_raw = oi_diff_raw[:size]
            for i, t in enumerate(fut_time):
                if t in time_to_idx:
                    oi_aligned[time_to_idx[t]] = oi_diff_raw[i]
        
        # Get existing EMA for this label
        existing_label_ema = existing_ema.get(label, {})
        
        output = {}
        
        # Apply incremental EMA smoothing
        if not np.isnan(fut_spot_aligned).all():
            output["fut_spot_diff"] = _apply_ema_incremental(
                fut_spot_aligned,
                existing_label_ema.get("fut_spot_diff"),
                processed_size
            )
        
        if not np.isnan(oi_aligned).all():
            output["oi_diff_cumsum"] = _apply_ema_incremental(
                oi_aligned,
                existing_label_ema.get("oi_diff_cumsum"),
                processed_size
            )
        
        if output:
            results[label] = output
    
    # Update cache
    if results:
        update_calculation_state(
            state_key,
            ema=results,
            processed_size=n_rows,
        )
        if new_rows > 0:
            log_info(f"[FuturesData] Processed {len(results)} futures, {new_rows} new rows (incremental EMA)")
    
    return results


def get_futures_normalized_data() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Get futures data with EMA smoothing applied.
    """
    return normalize_all_futures()


def get_futures_metadata() -> Dict:
    """
    Get metadata for futures dashboard (fast initial load).
    """
    catalog = get_trading_catalog()
    futures_map = _get_futures_from_catalog(catalog)
    
    available_futures = [label for label in ALL_FUTURES if label in futures_map]
    
    # Get time_seconds from NIFTY spot
    base_time = _get_base_time_seconds()
    time_seconds = base_time.tolist() if base_time is not None else []
    
    return {
        "available_futures": available_futures,
        "time_seconds": time_seconds,
        "index_futures": [f for f in INDEX_FUTURES if f in available_futures],
        "equity_futures": [f for f in EQUITY_FUTURES if f in available_futures],
    }


__all__ = [
    "normalize_all_futures",
    "get_futures_normalized_data",
    "get_futures_metadata",
    "ALL_FUTURES",
    "INDEX_FUTURES",
    "EQUITY_FUTURES",
]
