#!/usr/bin/env python3

"""
COMBINED INDEX OPTIONS NORMALIZATION - HYBRID APPROACH

Strategy:
1. FIRST RUN: Use Pandas expanding() - FAST bulk calculation
2. INCREMENTAL: Use NumPy loop - only NEW rows (cached results reused)

This is ~100x faster than pure NumPy approach and doesn't need multiple CPU cores!

Only for: NIFTY, BANKNIFTY, SENSEX options
Columns: iv_diff_cumsum, oi_diff_cumsum, timevalue_vol_prod_cumsum, timevalue_diff_cumsum
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from state import get_numpy_candle_snapshot, get_trading_catalog, get_calculation_state, update_calculation_state
from logger import log_info, log_error

# Columns to normalize
NORMALIZE_COLUMNS = (
    "iv_diff_cumsum",
    "oi_diff_cumsum",
    "timevalue_vol_prod_cumsum",
    "timevalue_diff_cumsum",
    "delta_diff_cumsum",
    "theta_diff_cumsum",
    "vega_diff_cumsum",
)

# EMA period for smoothing (12 periods = ~2 min for 10s data)
EMA_PERIOD = 12

# Raw columns (not normalized, just aligned) - REMOVED, will normalize IV instead
RAW_COLUMNS = ()

# Index names
INDEX_NAMES = ("NIFTY", "BANKNIFTY", "SENSEX")


def _parse_trading_symbol(symbol: str) -> Optional[Dict]:
    """
    Parse trading symbol to extract index, strike, type, expiry.
    Example: "NIFTY 24000 CE 26 DEC 24" -> {index: NIFTY, strike: 24000, type: CE, expiry: "26 DEC 24"}
    """
    for index_name in INDEX_NAMES:
        if symbol.startswith(index_name + " "):
            rest = symbol[len(index_name) + 1:]
            parts = rest.split()
            
            if len(parts) >= 5:
                try:
                    strike = int(float(parts[0]))
                    opt_type = parts[1]  # CE or PE
                    expiry = " ".join(parts[2:5])  # "26 DEC 24"
                    
                    if opt_type in ("CE", "PE"):
                        return {
                            "index": index_name,
                            "strike": strike,
                            "type": opt_type,
                            "expiry": expiry,
                            "expiry_short": parts[3] + parts[4][-2:],  # "DEC24"
                        }
                except (ValueError, IndexError):
                    pass
    
    return None


def _get_index_options(catalog: List[dict]) -> Dict[str, List[str]]:
    """
    Group trading symbols by index.
    Returns: {"NIFTY": [symbols...], "BANKNIFTY": [symbols...], "SENSEX": [symbols...]}
    """
    grouped = {name: [] for name in INDEX_NAMES}
    
    for entry in catalog:
        symbol = entry.get("trading_symbol") if isinstance(entry, dict) else entry
        if not symbol:
            continue
        parsed = _parse_trading_symbol(symbol)
        if parsed:
            grouped[parsed["index"]].append(symbol)
    
    return grouped


# Spot symbol mapping for each index
SPOT_SYMBOL_MAP = {
    "NIFTY": "NIFTY",
    "BANKNIFTY": "BANKNIFTY", 
    "SENSEX": "SENSEX",
}


def _get_spot_data(index_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get time_seconds and avg_diff_cumsum from SPOT instrument.
    This is the most reliable base for normalization.
    
    Returns: (time_seconds, avg_diff_cumsum) or (None, None)
    """
    spot_symbol = SPOT_SYMBOL_MAP.get(index_name)
    if not spot_symbol:
        return None, None
    
    snapshot = get_numpy_candle_snapshot(spot_symbol)
    if not snapshot:
        return None, None
    
    time_seconds = snapshot.get("time_seconds")
    avg_diff_cumsum = snapshot.get("avg_diff_cumsum")
    
    if time_seconds is None:
        return None, None
    
    size = snapshot.get("size", len(time_seconds))
    time_seconds = time_seconds[:size]
    
    if avg_diff_cumsum is not None:
        avg_diff_cumsum = avg_diff_cumsum[:size]
    
    return time_seconds, avg_diff_cumsum


def _build_combined_matrix(
    index_name: str,
    options_symbols: List[str],
    base_time_seconds: np.ndarray,
    spot_avg_diff_cumsum: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Build combined matrix for an index.
    
    Args:
        index_name: NIFTY, BANKNIFTY, or SENSEX
        options_symbols: List of option trading symbols
        base_time_seconds: Time array from spot (base for alignment)
        spot_avg_diff_cumsum: Spot's avg_diff_cumsum to include in normalization
    
    Returns:
        - combined_data: {"time_seconds": [...], "SPOT_avg_diff_cumsum": [...], "EXP1_24000CE_iv_diff_cumsum": [...], ...}
        - column_names: list of column names created (including spot column)
    """
    combined_data = {"time_seconds": base_time_seconds}
    column_names = []
    n_rows = len(base_time_seconds)
    
    # Add SPOT's avg_diff_cumsum as first column
    if spot_avg_diff_cumsum is not None and len(spot_avg_diff_cumsum) == n_rows:
        spot_col_name = f"SPOT_{index_name}_avg_diff_cumsum"
        combined_data[spot_col_name] = spot_avg_diff_cumsum.astype(np.float32)
        column_names.append(spot_col_name)
    
    # Create time lookup for fast alignment
    time_to_idx = {t: i for i, t in enumerate(base_time_seconds)}
    
    for symbol in options_symbols:
        parsed = _parse_trading_symbol(symbol)
        if not parsed:
            continue
        
        snapshot = get_numpy_candle_snapshot(symbol)
        if not snapshot:
            continue
        
        opt_time = snapshot.get("time_seconds")
        if opt_time is None:
            continue
        
        size = snapshot.get("size", len(opt_time))
        opt_time = opt_time[:size]
        
        col_prefix = f"{parsed['expiry_short']}_{parsed['strike']}{parsed['type']}"
        
        # Add NORMALIZE_COLUMNS (will be normalized later)
        # IMPORTANT: For cumsum columns, use LINEAR INTERPOLATION to fill gaps
        # This provides smooth transitions and prevents artificial jumps in z-scores
        for col_name in NORMALIZE_COLUMNS:
            col_data = snapshot.get(col_name)
            if col_data is None:
                continue
            
            col_data = col_data[:size]
            
            # Align to base time
            aligned = np.full(n_rows, np.nan, dtype=np.float32)
            for i, t in enumerate(opt_time):
                if t in time_to_idx:
                    aligned[time_to_idx[t]] = col_data[i]
            
            # Linear interpolation ONLY for middle gaps, keep leading NaN
            # This prevents creating artificial data at the start
            pd_series = pd.Series(aligned)
            # interpolate() naturally keeps leading NaN, only fills middle gaps
            # ffill() at the end for any trailing gaps after last valid value
            aligned = pd_series.interpolate(method='linear', limit_direction='forward').ffill().to_numpy(dtype=np.float32)
            
            full_col_name = f"{col_prefix}_{col_name}"
            combined_data[full_col_name] = aligned
            column_names.append(full_col_name)
        
        # Add IV as normalized (direct normalization of raw IV %)
        # This normalizes the actual IV value, not the change
        iv_data = snapshot.get("iv")
        if iv_data is not None:
            iv_data = iv_data[:size]
            
            # Align to base time
            aligned_iv = np.full(n_rows, np.nan, dtype=np.float32)
            for i, t in enumerate(opt_time):
                if t in time_to_idx:
                    aligned_iv[time_to_idx[t]] = iv_data[i]
            
            # Store raw IV values - will be normalized directly (not diff/cumsum)
            full_col_name = f"{col_prefix}_iv"
            combined_data[full_col_name] = aligned_iv
            column_names.append(full_col_name)  # Add to normalize list
    
    return combined_data, column_names


def _normalize_pandas_expanding(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """
    Use Pandas expanding() for FAST bulk normalization.
    This is used for FIRST RUN only.
    
    IMPORTANT: Leading NaN values are preserved and stats are computed
    only from valid (non-NaN) data points.
    """
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
    
    for col in df.columns:
        series = df[col]
        
        # Find first valid index
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is None:
            # All NaN - keep as NaN
            result[col] = np.nan
            continue
        
        # Get position of first valid value
        first_valid_pos = df.index.get_loc(first_valid_idx)
        
        # Leading NaN stays NaN
        if first_valid_pos > 0:
            result.loc[result.index[:first_valid_pos], col] = np.nan
        
        # Extract only valid data portion (from first valid value onwards)
        valid_series = series.iloc[first_valid_pos:]
        
        # Handle any remaining NaN in valid portion with ffill
        valid_filled = valid_series.ffill().fillna(0)
        
        # Calculate expanding stats only on valid data
        exp_median = valid_filled.expanding(min_periods=1).median()
        exp_q1 = valid_filled.expanding(min_periods=2).quantile(0.25)
        exp_q3 = valid_filled.expanding(min_periods=2).quantile(0.75)
        
        iqr = (exp_q3 - exp_q1)
        
        # Dynamic floor: max of fixed floor, or 10% of absolute median
        # This prevents huge normalized values when IQR is tiny but values are changing
        abs_median = exp_median.abs()
        fixed_floor = 0.01
        dynamic_floor = (abs_median * 0.1).clip(lower=fixed_floor)
        
        # Normalize
        scaled = (valid_filled - exp_median) / iqr.clip(lower=dynamic_floor)
        scaled = scaled.round(decimals).astype('float32')
        
        # First value = 0 (no history to normalize against)
        scaled.iloc[0] = 0.0
        
        # Put back in result
        result.loc[result.index[first_valid_pos:], col] = scaled.values
    
    return result


def _normalize_incremental_numpy(
    raw: np.ndarray,
    start_index: int,
    existing_norm: np.ndarray,
    decimals: int = 2,
) -> np.ndarray:
    """
    Use NumPy loop for INCREMENTAL normalization.
    Only calculates NEW rows (start_index to end).
    
    Leading NaN values are preserved - normalization only starts from first valid value.
    """
    n = len(raw)
    
    # Find first valid index in raw data
    valid_mask = np.isfinite(raw)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        # All NaN
        return np.full(n, np.nan, dtype=np.float32)
    
    first_valid_pos = valid_indices[0]
    
    # Create output array
    out = np.full(n, np.nan, dtype=np.float32)
    
    # Copy cached normalized values
    if start_index > 0 and existing_norm is not None:
        out[:start_index] = existing_norm[:start_index]
    
    # If start_index is before first valid, adjust
    effective_start = max(start_index, first_valid_pos)
    
    # Forward fill for computing stats (only valid portion)
    filled = np.copy(raw)
    mask = np.isnan(filled)
    idx = np.where(~mask, np.arange(len(filled)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = filled[idx]
    filled = np.where(np.isnan(filled), 0.0, filled)
    
    # Calculate only NEW rows (from first valid onwards)
    for i in range(effective_start, n):
        # Only use data from first_valid_pos onwards for stats
        prefix = filled[first_valid_pos:i+1]
        
        if len(prefix) < 1:
            continue
        
        if len(prefix) == 1:
            out[i] = 0.0  # First valid value = 0
        elif len(prefix) >= 2:
            med = np.median(prefix)
            q1, q3 = np.percentile(prefix, [25, 75])
            raw_iqr = q3 - q1
            # Dynamic floor: max of fixed floor, or 10% of absolute median
            dynamic_floor = max(0.01, abs(med) * 0.1)
            iqr = max(raw_iqr, dynamic_floor)
            
            out[i] = round((filled[i] - med) / iqr, decimals)
    
    return out


def _zscore_pandas_expanding(series: pd.Series, decimals: int = 2) -> pd.Series:
    """
    Z-score normalization using expanding window.
    zscore = (value - mean) / std
    Used ONLY for OI columns.
    
    IMPORTANT: Leading NaN values are preserved and stats are computed
    only from valid (non-NaN) data points.
    """
    result = pd.Series(index=series.index, dtype=np.float32)
    
    # Find first valid index
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        return result  # All NaN
    
    # Get position of first valid value
    first_valid_pos = series.index.get_loc(first_valid_idx)
    
    # Leading NaN stays NaN
    if first_valid_pos > 0:
        result.loc[result.index[:first_valid_pos]] = np.nan
    
    # Extract only valid data portion
    valid_series = series.iloc[first_valid_pos:]
    
    # Handle any remaining NaN with ffill
    valid_filled = valid_series.ffill().fillna(0)
    
    # Calculate expanding stats
    exp_mean = valid_filled.expanding(min_periods=1).mean()
    exp_std = valid_filled.expanding(min_periods=2).std().clip(lower=1e-9)
    
    # zscore
    zscore = (valid_filled - exp_mean) / exp_std
    zscore = zscore.round(decimals).astype('float32')
    
    # First value = 0
    zscore.iloc[0] = 0.0
    
    # Put back
    result.loc[result.index[first_valid_pos:]] = zscore.values
    
    return result


def _zscore_incremental_numpy(
    raw: np.ndarray,
    start_index: int,
    existing_norm: np.ndarray,
    decimals: int = 2,
) -> np.ndarray:
    """
    Z-score normalization using NumPy for incremental updates.
    Used ONLY for OI columns.
    
    Leading NaN values are preserved - normalization only starts from first valid value.
    """
    n = len(raw)
    
    # Find first valid index
    valid_mask = np.isfinite(raw)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return np.full(n, np.nan, dtype=np.float32)
    
    first_valid_pos = valid_indices[0]
    
    # Create output array with NaN
    out = np.full(n, np.nan, dtype=np.float32)
    
    # Copy cached values
    if start_index > 0 and existing_norm is not None:
        out[:start_index] = existing_norm[:start_index]
    
    effective_start = max(start_index, first_valid_pos)
    
    # Forward fill for computing stats
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
            # Dynamic floor: max of fixed floor, or 10% of absolute mean
            dynamic_floor = max(0.01, abs(mean) * 0.1)
            std = max(raw_std, dynamic_floor)
            
            out[i] = round((filled[i] - mean) / std, decimals)
    
    return out


def _calculate_ema(data: np.ndarray, period: int = EMA_PERIOD) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA) for smoothing.
    
    EMA formula:
    - multiplier = 2 / (period + 1)
    - EMA_today = (value_today * multiplier) + (EMA_yesterday * (1 - multiplier))
    
    Args:
        data: Input numpy array (normalized values)
        period: EMA period (default 12 = ~2 min for 10s data)
    
    Returns:
        EMA smoothed array
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
    
    # Round to 2 decimals
    return np.round(ema, 2).astype(np.float32)


def _add_ema_columns(norm_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Add EMA smoothed columns for all normalized data.
    
    For each column like 'DEC24_24000CE_iv_diff_cumsum',
    creates 'DEC24_24000CE_iv_diff_cumsum_ema'
    """
    ema_columns = {}
    
    for col_name, values in norm_data.items():
        # Skip if already an EMA column or vega_skew
        if col_name.endswith('_ema') or '_vega_skew' in col_name:
            continue
        
        # Calculate EMA
        ema_values = _calculate_ema(values)
        ema_col_name = f"{col_name}_ema"
        ema_columns[ema_col_name] = ema_values
    
    # Merge original + EMA columns
    result = dict(norm_data)
    result.update(ema_columns)
    
    return result


def _add_vega_skew_columns(norm_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate Vega Skew = CE Vega - PE Vega for each strike.
    
    For each strike, finds matching CE and PE vega columns and calculates difference.
    Creates columns like 'DEC24_24000_vega_skew' (without CE/PE suffix)
    """
    import re
    
    # Find all CE vega columns
    ce_vega_cols = {}
    pe_vega_cols = {}
    
    for col_name in norm_data.keys():
        # Match pattern: EXPIRY_STRIKE_CE/PE_vega_diff_cumsum (not _ema)
        match = re.match(r'^([A-Z]{3}\d{2})_(\d+)(CE|PE)_vega_diff_cumsum$', col_name)
        if match:
            expiry = match.group(1)
            strike = match.group(2)
            opt_type = match.group(3)
            key = f"{expiry}_{strike}"
            
            if opt_type == 'CE':
                ce_vega_cols[key] = col_name
            else:
                pe_vega_cols[key] = col_name
    
    # Calculate skew for matching pairs
    skew_columns = {}
    
    for key in ce_vega_cols:
        if key in pe_vega_cols:
            ce_col = ce_vega_cols[key]
            pe_col = pe_vega_cols[key]
            
            ce_values = norm_data[ce_col]
            pe_values = norm_data[pe_col]
            
            # Simple difference: CE Vega - PE Vega
            skew = ce_values - pe_values
            
            # Round to 2 decimals
            skew = np.round(skew, 2).astype(np.float32)
            
            skew_col_name = f"{key}_vega_skew"
            skew_columns[skew_col_name] = skew
            
            # Also create EMA version
            skew_ema = _calculate_ema(skew)
            skew_columns[f"{skew_col_name}_ema"] = skew_ema
    
    # Merge with original data
    result = dict(norm_data)
    result.update(skew_columns)
    
    return result


def normalize_index_options(index_name: str) -> Dict[str, np.ndarray]:
    """
    HYBRID Normalization for all options of a single index.
    
    - Uses SPOT data (time_seconds, avg_diff_cumsum) as base
    - FIRST RUN: Pandas expanding() (fast)
    - INCREMENTAL: NumPy loop (only new rows)
    
    Returns normalized data dict.
    """
    catalog = get_trading_catalog()
    if not catalog:
        return {}
    
    # Get options for this index
    index_options = _get_index_options(catalog)
    options_symbols = index_options.get(index_name, [])
    
    if not options_symbols:
        return {}
    
    # Get SPOT data as base (more reliable than options)
    base_time, spot_avg_diff_cumsum = _get_spot_data(index_name)
    if base_time is None or len(base_time) == 0:
        log_error(f"[Normalization] {index_name}: No SPOT data found!")
        return {}
    
    # Build combined matrix with spot's avg_diff_cumsum
    combined_data, column_names = _build_combined_matrix(
        index_name, options_symbols, base_time, spot_avg_diff_cumsum
    )
    
    if not column_names:
        return {}
    
    # Get existing state (cache)
    state_key = f"{index_name}_COMBINED"
    calc_state = get_calculation_state(state_key)
    existing_norm = calc_state.get("norm", {})
    normalized_size = calc_state.get("normalized_size", 0)
    cache_version = calc_state.get("cache_version", 0)
    
    # Cache version to invalidate old caches (increment when logic changes)
    CURRENT_CACHE_VERSION = 9  # Linear interp for middle gaps only, keep leading NaN
    
    n_rows = len(base_time)
    new_rows = n_rows - normalized_size
    
    # Force recalculation if cache version is old
    if cache_version < CURRENT_CACHE_VERSION:
        log_info(f"[Normalization] {index_name}: Cache version outdated ({cache_version} < {CURRENT_CACHE_VERSION}), forcing recalculation")
        normalized_size = 0
        existing_norm = {}
    
    norm_out = {}
    
    # =====================================================
    # CASE 1: FIRST RUN (no cache) - Use Pandas
    # =====================================================
    if normalized_size == 0 or not existing_norm:
        log_info(f"[Normalization] {index_name}: FIRST RUN - {n_rows} rows, {len(column_names)} columns (Pandas)")
        
        # Separate OI columns (z-score) from others (IQR)
        oi_cols = [c for c in column_names if 'oi_diff' in c]
        other_cols = [c for c in column_names if 'oi_diff' not in c]
        
        # Z-score for OI columns
        for col in oi_cols:
            series = pd.Series(combined_data[col])
            norm_out[col] = _zscore_pandas_expanding(series).values.astype(np.float32)
        
        # IQR normalization for other columns
        if other_cols:
            df_data = {col: combined_data[col] for col in other_cols}
            df = pd.DataFrame(df_data)
            normalized_df = _normalize_pandas_expanding(df)
            for col in other_cols:
                norm_out[col] = normalized_df[col].values.astype(np.float32)
    
    # =====================================================
    # CASE 2: INCREMENTAL - Use NumPy (only new rows)
    # =====================================================
    elif new_rows > 0:
        log_info(f"[Normalization] {index_name}: INCREMENTAL - +{new_rows} new rows (NumPy)")
        
        for col_name in column_names:
            raw = combined_data.get(col_name)
            if raw is None:
                continue
            
            existing = existing_norm.get(col_name)
            is_oi_col = 'oi_diff' in col_name
            
            if existing is not None and len(existing) >= normalized_size:
                # Incremental: only calculate new rows
                if is_oi_col:
                    norm_out[col_name] = _zscore_incremental_numpy(
                        raw, normalized_size, existing
                    )
                else:
                    norm_out[col_name] = _normalize_incremental_numpy(
                        raw, normalized_size, existing
                    )
            else:
                # Column is new, need full calculation
                s = pd.Series(raw)
                if is_oi_col:
                    # Z-score for OI
                    norm_out[col_name] = _zscore_pandas_expanding(s).values.astype(np.float32)
                else:
                    # IQR for others
                    med = s.expanding(min_periods=1).median()
                    q1 = s.expanding(min_periods=2).quantile(0.25)
                    q3 = s.expanding(min_periods=2).quantile(0.75)
                    iqr = (q3 - q1).clip(lower=0.01)  # Reasonable floor
                    normalized = ((s - med.shift(1)) / iqr.shift(1)).round(2).fillna(0)
                    norm_out[col_name] = normalized.values.astype(np.float32)
    
    # =====================================================
    # CASE 3: NO NEW DATA - Return cached
    # =====================================================
    else:
        log_info(f"[Normalization] {index_name}: NO NEW DATA - returning cached ({normalized_size} rows)")
        return existing_norm
    
    # Store results in cache with version
    update_calculation_state(
        state_key,
        norm=norm_out,
        normalized_size=n_rows,
        column_names=column_names,
        cache_version=CURRENT_CACHE_VERSION,
    )
    
    return norm_out


def normalize_all_index_options() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Normalize options for all indices: NIFTY, BANKNIFTY, SENSEX.
    
    Call this AFTER all fetches are complete.
    
    Returns: {"NIFTY": {...}, "BANKNIFTY": {...}, "SENSEX": {...}}
    """
    results = {}
    for index_name in INDEX_NAMES:
        try:
            results[index_name] = normalize_index_options(index_name)
        except Exception as e:
            log_error(f"[Normalization] {index_name} failed: {e}")
            results[index_name] = {}
    
    return results


def get_metadata_only(index_name: str) -> Optional[Dict]:
    """
    Get ONLY metadata (expiries, strikes, spot_price) without heavy normalization.
    Used for fast initial page load.
    
    Returns: {normalized: {}, time_seconds: [], spot_price: float, available_expiries: [], available_strikes: [], ...}
    """
    import re
    
    # Get cached calculation state (already computed in background)
    state_key = f"{index_name}_COMBINED"
    calc_state = get_calculation_state(state_key)
    
    # Extract metadata from cached normalized data
    norm_data = calc_state.get("norm", {})
    
    if not norm_data:
        # Try to get from options symbols directly
        catalog = get_trading_catalog()
        grouped = _get_index_options(catalog)
        options_symbols = grouped.get(index_name, [])
        
        if not options_symbols:
            return None
    
    # Extract available expiries and strikes from column names
    available_expiries = set()
    all_strikes = set()
    
    for col_name in norm_data.keys():
        # Extract expiry: DEC24_24000CE_iv_diff_cumsum -> DEC24
        if "_" in col_name:
            exp = col_name.split("_")[0]
            if exp and len(exp) >= 4 and exp[:3].isalpha():
                available_expiries.add(exp)
        
        # Extract strike
        match = re.match(r'^([A-Z]{3}\d{2})_(\d+)(CE|PE)_', col_name)
        if match:
            all_strikes.add(int(match.group(2)))
    
    available_expiries = sorted(available_expiries)
    available_strikes = sorted(all_strikes)
    
    # Get time_seconds and spot_price from spot snapshot
    spot_snapshot = get_numpy_candle_snapshot(index_name)
    time_seconds = []
    spot_price = None
    
    if spot_snapshot:
        ts = spot_snapshot.get("time_seconds")
        size = spot_snapshot.get("size", len(ts) if ts is not None else 0)
        if ts is not None:
            time_seconds = ts[:size].tolist()
        
        avg = spot_snapshot.get("average")
        if avg is not None and size > 0:
            import numpy as np
            spot_price = float(avg[size - 1]) if not np.isnan(avg[size - 1]) else None
    
    return {
        "index": index_name,
        "normalized": {},  # Empty - no data, just metadata
        "time_seconds": time_seconds,
        "spot_price": spot_price,
        "available_expiries": available_expiries,
        "available_strikes": available_strikes,
        "smooth": True,
        "current_expiry": available_expiries[0] if available_expiries else None,
    }


def get_normalized_index_data(index_name: str) -> Dict[str, np.ndarray]:
    """
    Get normalized data for an index WITH EMA smoothed columns.
    Returns cached data if available, otherwise computes it.
    
    Each original column gets an _ema version for smooth display.
    EMA and skew columns are also cached to avoid recomputation.
    """
    state_key = f"{index_name}_COMBINED"
    calc_state = get_calculation_state(state_key)
    
    # Check if we have cached data WITH EMA columns
    cached_with_ema = calc_state.get("norm_with_ema")
    cached_ema_size = calc_state.get("ema_size", 0)
    
    if calc_state.get("norm"):
        norm_data = calc_state["norm"]
        current_size = len(next(iter(norm_data.values()))) if norm_data else 0
        
        # If EMA cache exists and is same size, return directly
        if cached_with_ema and cached_ema_size == current_size:
            return cached_with_ema
        
        # Otherwise compute EMA and cache it
        result = _add_ema_columns(norm_data)
        result = _add_vega_skew_columns(result)
        
        # Cache the EMA result
        update_calculation_state(
            state_key,
            norm_with_ema=result,
            ema_size=current_size,
        )
        return result
    else:
        norm_data = normalize_index_options(index_name)
        result = _add_ema_columns(norm_data)
        result = _add_vega_skew_columns(result)
        return result


__all__ = [
    "normalize_index_options",
    "normalize_all_index_options",
    "get_normalized_index_data",
    "get_metadata_only",
    "NORMALIZE_COLUMNS",
    "INDEX_NAMES",
    "EMA_PERIOD",
]
