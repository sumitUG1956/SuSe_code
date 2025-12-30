#!/usr/bin/env python3

"""
FUTURES NORMALIZATION - Same approach as Options

Strategy:
1. FIRST RUN: Use Pandas expanding() - FAST bulk calculation
2. INCREMENTAL: Use NumPy loop - only NEW rows (cached results reused)

Futures: NIFTY FUT, BANKNIFTY FUT + 11 Stock Futures
Columns: fut_spot_diff_cumsum (EMA smooth → IQR normalize), oi_diff_cumsum (Z-score normalize)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from state import get_numpy_candle_snapshot, get_trading_catalog, get_calculation_state, update_calculation_state
from logger import log_info, log_error

# EMA period for smoothing fut_spot_diff before normalization (same as options)
EMA_PERIOD = 12

# Futures to normalize
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


def _calculate_ema(data: np.ndarray, period: int = EMA_PERIOD) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA) for smoothing.
    Applied BEFORE normalization for fut_spot_diff_cumsum.
    """
    n = len(data)
    if n == 0:
        return data
    
    ema = np.zeros(n, dtype=np.float32)
    multiplier = 2.0 / (period + 1)
    
    # Handle NaN - forward fill
    filled = np.copy(data)
    mask = np.isnan(filled)
    if mask.any():
        idx = np.where(~mask, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = filled[idx]
        filled = np.where(np.isnan(filled), 0.0, filled)
    
    # First value = first data point
    ema[0] = filled[0]
    
    # Calculate EMA for rest
    for i in range(1, n):
        ema[i] = (filled[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    return ema.astype(np.float32)


def _normalize_pandas_expanding(series: pd.Series, decimals: int = 2) -> np.ndarray:
    """
    IQR-based normalization using Pandas expanding window.
    Used for fut_spot_diff_cumsum.
    """
    if series.isna().all():
        return np.full(len(series), np.nan, dtype=np.float32)
    
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        return np.full(len(series), np.nan, dtype=np.float32)
    
    first_valid_pos = series.index.get_loc(first_valid_idx)
    result = np.full(len(series), np.nan, dtype=np.float32)
    
    # Extract valid portion
    valid_series = series.iloc[first_valid_pos:]
    valid_filled = valid_series.ffill().fillna(0)
    
    # Expanding stats
    exp_median = valid_filled.expanding(min_periods=1).median()
    exp_q1 = valid_filled.expanding(min_periods=2).quantile(0.25)
    exp_q3 = valid_filled.expanding(min_periods=2).quantile(0.75)
    
    iqr = (exp_q3 - exp_q1)
    
    # Dynamic floor
    abs_median = exp_median.abs()
    fixed_floor = 0.01
    dynamic_floor = (abs_median * 0.1).clip(lower=fixed_floor)
    
    # Normalize
    scaled = (valid_filled - exp_median) / iqr.clip(lower=dynamic_floor)
    scaled = scaled.round(decimals).astype('float32')
    scaled.iloc[0] = 0.0  # First value = 0
    
    result[first_valid_pos:] = scaled.values
    return result


def _zscore_pandas_expanding(series: pd.Series, decimals: int = 2) -> np.ndarray:
    """
    Z-score normalization using Pandas expanding window.
    Used for oi_diff_cumsum.
    """
    if series.isna().all():
        return np.full(len(series), np.nan, dtype=np.float32)
    
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        return np.full(len(series), np.nan, dtype=np.float32)
    
    first_valid_pos = series.index.get_loc(first_valid_idx)
    result = np.full(len(series), np.nan, dtype=np.float32)
    
    valid_series = series.iloc[first_valid_pos:]
    valid_filled = valid_series.ffill().fillna(0)
    
    # Z-score stats
    exp_mean = valid_filled.expanding(min_periods=1).mean()
    exp_std = valid_filled.expanding(min_periods=2).std()
    
    # Dynamic floor for std
    abs_mean = exp_mean.abs()
    dynamic_floor = (abs_mean * 0.1).clip(lower=0.01)
    exp_std = exp_std.clip(lower=dynamic_floor)
    
    zscore = (valid_filled - exp_mean) / exp_std
    zscore = zscore.round(decimals).astype('float32')
    zscore.iloc[0] = 0.0
    
    result[first_valid_pos:] = zscore.values
    return result


def _normalize_incremental_numpy(
    raw: np.ndarray,
    start_index: int,
    existing_norm: np.ndarray,
    decimals: int = 2,
) -> np.ndarray:
    """
    IQR normalization using NumPy for incremental updates.
    """
    n = len(raw)
    valid_mask = np.isfinite(raw)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return np.full(n, np.nan, dtype=np.float32)
    
    first_valid_pos = valid_indices[0]
    out = np.full(n, np.nan, dtype=np.float32)
    
    if start_index > 0 and existing_norm is not None:
        out[:start_index] = existing_norm[:start_index]
    
    effective_start = max(start_index, first_valid_pos)
    
    # Forward fill
    filled = np.copy(raw)
    mask = np.isnan(filled)
    idx = np.where(~mask, np.arange(len(filled)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = filled[idx]
    filled = np.where(np.isnan(filled), 0.0, filled)
    
    for i in range(effective_start, n):
        prefix = filled[first_valid_pos:i+1]
        
        if len(prefix) < 1:
            continue
        
        if len(prefix) == 1:
            out[i] = 0.0
        elif len(prefix) >= 2:
            med = np.median(prefix)
            q1, q3 = np.percentile(prefix, [25, 75])
            raw_iqr = q3 - q1
            dynamic_floor = max(0.01, abs(med) * 0.1)
            iqr = max(raw_iqr, dynamic_floor)
            out[i] = round((filled[i] - med) / iqr, decimals)
    
    return out


def _zscore_incremental_numpy(
    raw: np.ndarray,
    start_index: int,
    existing_norm: np.ndarray,
    decimals: int = 2,
) -> np.ndarray:
    """
    Z-score normalization using NumPy for incremental updates.
    """
    n = len(raw)
    valid_mask = np.isfinite(raw)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return np.full(n, np.nan, dtype=np.float32)
    
    first_valid_pos = valid_indices[0]
    out = np.full(n, np.nan, dtype=np.float32)
    
    if start_index > 0 and existing_norm is not None:
        out[:start_index] = existing_norm[:start_index]
    
    effective_start = max(start_index, first_valid_pos)
    
    filled = np.copy(raw)
    mask = np.isnan(filled)
    idx = np.where(~mask, np.arange(len(filled)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = filled[idx]
    filled = np.where(np.isnan(filled), 0.0, filled)
    
    for i in range(effective_start, n):
        prefix = filled[first_valid_pos:i+1]
        
        if len(prefix) < 1:
            continue
        
        if len(prefix) == 1:
            out[i] = 0.0
        elif len(prefix) >= 2:
            mean = np.mean(prefix)
            raw_std = np.std(prefix, ddof=0)
            dynamic_floor = max(0.01, abs(mean) * 0.1)
            std = max(raw_std, dynamic_floor)
            out[i] = round((filled[i] - mean) / std, decimals)
    
    return out


def _add_ema_columns(norm_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Add EMA smoothed columns for display.
    """
    ema_columns = {}
    
    for col_name, values in norm_data.items():
        if col_name.endswith('_ema') or col_name == 'time_seconds':
            continue
        
        ema_values = _calculate_ema(values)
        ema_columns[f"{col_name}_ema"] = np.round(ema_values, 2).astype(np.float32)
    
    result = dict(norm_data)
    result.update(ema_columns)
    return result


def normalize_all_futures() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Normalize all futures data.
    
    Returns: {
        "NIFTY": {
            "fut_spot_diff_cumsum": [...],
            "fut_spot_diff_cumsum_ema": [...],
            "oi_diff_cumsum": [...],
            "oi_diff_cumsum_ema": [...],
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
        log_error("[FuturesNorm] No base time_seconds available!")
        return {}
    
    n_rows = len(base_time)
    
    # Get futures trading symbols from catalog
    futures_map = _get_futures_from_catalog(catalog)
    
    # Get cached state
    state_key = "FUTURES_COMBINED"
    calc_state = get_calculation_state(state_key)
    existing_norm = calc_state.get("norm", {})
    normalized_size = calc_state.get("normalized_size", 0)
    cache_version = calc_state.get("cache_version", 0)
    
    CURRENT_CACHE_VERSION = 1
    
    new_rows = n_rows - normalized_size
    
    # Force recalculation if cache version changed
    if cache_version < CURRENT_CACHE_VERSION:
        log_info(f"[FuturesNorm] Cache version outdated, forcing recalculation")
        normalized_size = 0
        existing_norm = {}
    
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
        
        # Get raw data
        fut_spot_diff_raw = snapshot.get("future_spot_diff_cumsum")
        oi_diff_raw = snapshot.get("oi_diff_cumsum")
        
        if fut_spot_diff_raw is None and oi_diff_raw is None:
            continue
        
        # Align to base time
        time_to_idx = {t: i for i, t in enumerate(base_time)}
        
        # Align fut_spot_diff_cumsum
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
        
        norm_out = {}
        
        # =====================================================
        # CASE 1: FIRST RUN - Use Pandas
        # =====================================================
        if normalized_size == 0 or label not in existing_norm:
            # FutSpotDiff: EMA smooth THEN IQR normalize
            if not np.isnan(fut_spot_aligned).all():
                fut_spot_smoothed = _calculate_ema(fut_spot_aligned, EMA_PERIOD)
                norm_out["fut_spot_diff_cumsum"] = _normalize_pandas_expanding(
                    pd.Series(fut_spot_smoothed)
                )
            
            # OI: Z-score normalize
            if not np.isnan(oi_aligned).all():
                norm_out["oi_diff_cumsum"] = _zscore_pandas_expanding(
                    pd.Series(oi_aligned)
                )
        
        # =====================================================
        # CASE 2: INCREMENTAL - Use NumPy
        # =====================================================
        elif new_rows > 0:
            existing_label = existing_norm.get(label, {})
            
            # FutSpotDiff
            if not np.isnan(fut_spot_aligned).all():
                fut_spot_smoothed = _calculate_ema(fut_spot_aligned, EMA_PERIOD)
                existing_fut = existing_label.get("fut_spot_diff_cumsum")
                norm_out["fut_spot_diff_cumsum"] = _normalize_incremental_numpy(
                    fut_spot_smoothed, normalized_size, existing_fut
                )
            
            # OI
            if not np.isnan(oi_aligned).all():
                existing_oi = existing_label.get("oi_diff_cumsum")
                norm_out["oi_diff_cumsum"] = _zscore_incremental_numpy(
                    oi_aligned, normalized_size, existing_oi
                )
        
        # =====================================================
        # CASE 3: NO NEW DATA - Return cached
        # =====================================================
        else:
            if label in existing_norm:
                results[label] = existing_norm[label]
                continue
        
        # Store normalized data (no extra EMA - already smoothed before IQR)
        if norm_out:
            results[label] = norm_out
    
    # Store in cache
    if results:
        update_calculation_state(
            state_key,
            norm=results,
            normalized_size=n_rows,
            cache_version=CURRENT_CACHE_VERSION,
        )
        log_info(f"[FuturesNorm] Normalized {len(results)} futures, {n_rows} rows")
    
    return results


def get_futures_normalized_data() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Get normalized futures data. Returns cached data if available.
    """
    state_key = "FUTURES_COMBINED"
    calc_state = get_calculation_state(state_key)
    
    if calc_state.get("norm"):
        return calc_state["norm"]
    
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
