#!/usr/bin/env python3

"""
TRUE EXPANDING Robust Scaler with INCREMENTAL UPDATES.

- 100% ACCURATE: True expanding window (no chunking)
- FAST INCREMENTAL: Only new rows are computed, old results are reused
- FILTERED: Normalize only what you need
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np

try:
    import bottleneck as bn
    HAS_BOTTLENECK = True
except ImportError:
    HAS_BOTTLENECK = False

# CPU cores for parallel processing
NUM_CPU_CORES = os.cpu_count() or 4
NUM_WORKERS = min(NUM_CPU_CORES, 10)

# Default columns from CandleBuffer snapshots we may want to normalize.
DEFAULT_NORMALIZE_COLUMNS = (
    "iv_diff_cumsum",
    "delta_diff_cumsum",
    "vega_diff_cumsum",
    "theta_diff_cumsum",
    "timevalue_diff_cumsum",
    "timevalue_vol_prod_cumsum",
    "oi_diff_cumsum",
    "average_diff_cumsum",
    "average_vol_prod_cumsum",
    "future_spot_diff_cumsum",
)


def _compute_stats(arr: np.ndarray, q_low_pct: float, q_high_pct: float):
    """Compute median and percentiles for an array."""
    valid = arr[np.isfinite(arr)]
    
    if len(valid) >= 2:
        if HAS_BOTTLENECK:
            med = bn.nanmedian(valid)
        else:
            med = np.median(valid)
        q1 = np.percentile(valid, q_low_pct)
        q3 = np.percentile(valid, q_high_pct)
    elif len(valid) == 1:
        med = q1 = q3 = valid[0]
    else:
        med = q1 = q3 = np.nan
    
    return med, q1, q3


def _robust_scale_expanding_true(
    raw: np.ndarray,
    *,
    start_index: int,
    existing_norm: np.ndarray | None,
    cached_stats: Optional[Tuple[float, float, float]],  # (med, q1, q3)
    q_low: float,
    q_high: float,
    iqr_floor: float,
    decimals: int,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    TRUE 100% ACCURATE Expanding window robust scaling.
    
    - Existing normalized values are REUSED (not recalculated!)
    - Only NEW rows are normalized using expanding stats
    - Returns (normalized_array, cached_stats)
    """
    raw = np.asarray(raw, dtype=np.float64)
    n = raw.shape[0]
    
    if n == 0:
        return np.array([], dtype=np.float32), (np.nan, np.nan, np.nan)
    
    start_index = max(0, min(start_index, n))
    q_low_pct = q_low * 100
    q_high_pct = q_high * 100
    
    # Check if we can do INCREMENTAL update (reuse existing normalized data)
    can_increment = (
        existing_norm is not None and 
        existing_norm.shape[0] >= start_index and 
        start_index > 0
    )
    
    if can_increment:
        # INCREMENTAL: Only normalize NEW rows
        out = np.full(n, np.nan, dtype=np.float32)
        out[:start_index] = existing_norm[:start_index]  # REUSE existing!
        
        # Normalize only new rows (from start_index to n)
        for i in range(start_index, n):
            # Use all data up to i for expanding stats
            prefix = raw[:i]
            med, q1, q3 = _compute_stats(prefix, q_low_pct, q_high_pct)
            iqr = max(q3 - q1, iqr_floor)
            
            if np.isfinite(raw[i]) and np.isfinite(med) and iqr > 0:
                out[i] = round((raw[i] - med) / iqr, decimals)
        
        # Cache final stats
        final_stats = _compute_stats(raw, q_low_pct, q_high_pct)
        return out, final_stats
    
    # FULL computation (first time)
    out = np.full(n, np.nan, dtype=np.float32)
    out[0] = 0.0  # First value = 0 (no history)
    
    for i in range(1, n):
        prefix = raw[:i]
        med, q1, q3 = _compute_stats(prefix, q_low_pct, q_high_pct)
        iqr = max(q3 - q1, iqr_floor)
        
        if np.isfinite(raw[i]) and np.isfinite(med) and iqr > 0:
            out[i] = round((raw[i] - med) / iqr, decimals)
    
    final_stats = _compute_stats(raw, q_low_pct, q_high_pct)
    return out, final_stats


def _normalize_single_column(args):
    """Worker function for parallel column normalization."""
    col, raw, start_index, existing, cached_stats, q_low, q_high, iqr_floor, decimals = args
    result, new_stats = _robust_scale_expanding_true(
        raw,
        start_index=start_index,
        existing_norm=existing,
        cached_stats=cached_stats,
        q_low=q_low,
        q_high=q_high,
        iqr_floor=iqr_floor,
        decimals=decimals,
    )
    return col, result, new_stats


def normalize_snapshot(
    snapshot: Dict,
    *,
    columns: Iterable[str] | None = None,
    start_index: int = 0,
    existing_norm: Dict[str, np.ndarray] | None = None,
    cached_stats: Dict[str, Tuple[float, float, float]] | None = None,
    q_low: float = 0.25,
    q_high: float = 0.75,
    iqr_floor: float = 1e-9,
    decimals: int = 2,
) -> Tuple[Dict[str, np.ndarray], int, Dict[str, Tuple[float, float, float]]]:
    """
    TRUE EXPANDING normalization with INCREMENTAL support.
    
    HOW IT WORKS:
    1. First call: Full normalization (slow, one-time)
    2. Next calls: Only new rows normalized (FAST!)
       - Pass start_index = previous row count
       - Pass existing_norm = previous normalized data
       - Old values are REUSED, not recalculated!
    
    Returns (norm_dict, total_rows, cached_stats)
    """
    if snapshot is None:
        return {}, 0, {}

    target_columns = list(columns) if columns is not None else list(DEFAULT_NORMALIZE_COLUMNS)
    existing_norm = existing_norm or {}
    cached_stats = cached_stats or {}

    n = 0
    if "time_seconds" in snapshot:
        n = len(snapshot["time_seconds"])
    else:
        for col in snapshot:
            if hasattr(snapshot[col], "__len__"):
                n = len(snapshot[col])
                break
    if n == 0:
        return {}, 0, {}

    # If already normalized up to current size, return existing
    if start_index >= n and existing_norm:
        return existing_norm, n, cached_stats

    # Prepare work items
    work_items = []
    for col in target_columns:
        raw = snapshot.get(col)
        if raw is None:
            continue
        work_items.append((
            col,
            np.asarray(raw, dtype=np.float64),
            start_index,
            existing_norm.get(col),
            cached_stats.get(col),
            q_low,
            q_high,
            iqr_floor,
            decimals,
        ))
    
    if not work_items:
        return {}, n, {}

    norm_out: Dict[str, np.ndarray] = {}
    stats_out: Dict[str, Tuple[float, float, float]] = {}
    
    # Parallel processing for columns
    if len(work_items) > 1:
        with ThreadPoolExecutor(max_workers=min(NUM_WORKERS, len(work_items))) as executor:
            results = list(executor.map(_normalize_single_column, work_items))
            for col, result, stats in results:
                norm_out[col] = result
                stats_out[col] = stats
    else:
        for item in work_items:
            col, result, stats = _normalize_single_column(item)
            norm_out[col] = result
            stats_out[col] = stats

    return norm_out, n, stats_out


__all__ = [
    "normalize_snapshot",
    "DEFAULT_NORMALIZE_COLUMNS",
]
