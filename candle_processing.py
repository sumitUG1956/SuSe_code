#!/usr/bin/env python3

from datetime import datetime, date, time, timezone
from typing import Optional
import sys

import numpy as np
import polars as pl
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
CANDLE_COLUMNS = ["Time", "Open", "High", "Low", "Close", "Volume", "OpenInterest"]
NUMERIC_DEFAULTS = {"Volume": 0.0, "OpenInterest": 0.0}
RISK_FREE_RATE = 0.10
SIGMA_LOW = 0.001
SIGMA_HIGH = 5.0
SIGMA_PRECISION = 1e-5
MAX_IV_ITERATIONS = 100
MS_IN_YEAR = 1000 * 60 * 60 * 24 * 365
SQRT_TWO = np.sqrt(2.0)
SPOT_COLUMN = "SpotAverage"
IV_COLUMN = "IV"
DELTA_COLUMN = "Delta"
TIMEVALUE_COLUMN = "TimeValue"
IV_DIFF_COLUMN = "IVDiff"
IV_DIFF_CUMSUM_COLUMN = "IVDiffCumSum"
VEGA_COLUMN = "Vega"
THETA_COLUMN = "Theta"
DELTA_DIFF_COLUMN = "DeltaDiff"
DELTA_DIFF_CUMSUM_COLUMN = "DeltaDiffCumSum"
VEGA_DIFF_COLUMN = "VegaDiff"
VEGA_DIFF_CUMSUM_COLUMN = "VegaDiffCumSum"
THETA_DIFF_COLUMN = "ThetaDiff"
THETA_DIFF_CUMSUM_COLUMN = "ThetaDiffCumSum"
TIMEVALUE_DIFF_COLUMN = "TimeValueDiff"
TIMEVALUE_DIFF_CUMSUM_COLUMN = "TimeValueDiffCumSum"
TIMEVALUE_VOL_PROD_COLUMN = "TimeValueVolProd"
TIMEVALUE_VOL_PROD_CUMSUM_COLUMN = "TimeValueVolProdCumSum"
OI_DIFF_COLUMN = "OpenInterestDiff"
OI_DIFF_CUMSUM_COLUMN = "OpenInterestDiffCumSum"
AVG_DIFF_COLUMN = "AverageDiff"
AVG_DIFF_CUMSUM_COLUMN = "AverageDiffCumSum"
AVG_VOL_PROD_COLUMN = "AverageVolProd"
AVG_VOL_PROD_CUMSUM_COLUMN = "AverageVolProdCumSum"
FUT_SPOT_DIFF_COLUMN = "FutureSpotDiff"
FUT_SPOT_DIFF_CUMSUM_COLUMN = "FutureSpotDiffCumSum"
SPOT_CACHE: dict[str, pl.DataFrame] = {}
_ERF_A1 = 0.254829592
_ERF_A2 = -0.284496736
_ERF_A3 = 1.421413741
_ERF_A4 = -1.453152027
_ERF_A5 = 1.061405429
_ERF_P = 0.3275911



def _empty_frame():
    return pl.DataFrame({column: [] for column in CANDLE_COLUMNS})


def _ensure_frame(candles):
    if not candles:
        return _empty_frame()

    width = len(candles[0])
    cols = CANDLE_COLUMNS[:width]
    df = pl.DataFrame(candles, schema=cols, orient="row")  # type: ignore[arg-type]

    missing = [column for column in CANDLE_COLUMNS if column not in df.columns]
    if missing:
        df = df.with_columns(
            [
                pl.lit(NUMERIC_DEFAULTS.get(column)).alias(column)
                for column in missing
            ]
        )

    df = df.select(CANDLE_COLUMNS)

    df = df.with_columns(
        [
            pl.col("Time").cast(pl.Datetime(time_unit="ms", time_zone="UTC")),
            pl.col("Open").cast(pl.Float32),
            pl.col("High").cast(pl.Float32),
            pl.col("Low").cast(pl.Float32),
            pl.col("Close").cast(pl.Float32),
            pl.col("Volume").cast(pl.Float32).fill_null(NUMERIC_DEFAULTS["Volume"]),
            pl.col("OpenInterest")
            .cast(pl.Float32)
            .fill_null(NUMERIC_DEFAULTS["OpenInterest"]),
            pl.mean_horizontal(["Open", "High", "Low", "Close"]).alias("Average").cast(pl.Float32),
        ]
    )
    
    return df


def _format_time(df, tz):
    return df.with_columns(pl.col("Time").dt.convert_time_zone(tz.key))


def _records_from_frame(df):
    if df.is_empty():
        return []
    formatted = df.with_columns(
        pl.col("Time").dt.strftime("%Y-%m-%dT%H:%M:%S%z").alias("Time")
    )
    return formatted.to_dicts()


def _filter_today(df, today):
    return df.filter(pl.col("Time").dt.date() == pl.lit(today))


def _filter_to_trading_date(df, today):
    if df.is_empty():
        return df, today

    dates = df.select(pl.col("Time").dt.date().alias("Date")).to_series()
    if (dates == today).any():
        trading_date = today
    else:
        trading_date = dates.max()

    filtered = df.filter(pl.col("Time").dt.date() == pl.lit(trading_date))
    return filtered, trading_date


def _cache_key_for_label(meta):
    if not meta:
        return None
    label = meta.get("label")
    if not label:
        return None
    return str(label).upper()


def _is_spot_category(meta):
    if not meta:
        return False
    category = str(meta.get("category", "")).lower()
    return category in {"index_spot", "equity_spot"}


def _is_option_contract(meta):
    if not meta:
        return False
    option_type = str(meta.get("option_type", "")).upper()
    return option_type in {"CE", "PE"}


def _needs_spot_reference(meta):
    if not meta:
        return False
    category = str(meta.get("category", "")).lower()
    if category in {"index_future", "equity_future"}:
        return True
    return _is_option_contract(meta)


def _attach_spot_average(df, instrument_meta):
    label = _cache_key_for_label(instrument_meta)
    if not label:
        return df.with_columns(pl.lit(None).cast(pl.Float32).alias(SPOT_COLUMN))

    if _is_spot_category(instrument_meta):
        cached = (
            df.select(["Time", "Average"])
            .rename({"Average": SPOT_COLUMN})
            .with_columns(pl.col(SPOT_COLUMN).cast(pl.Float32))
        )
        SPOT_CACHE[label] = cached
        return df.with_columns(pl.col("Average").alias(SPOT_COLUMN))

    if not _needs_spot_reference(instrument_meta):
        return df.with_columns(pl.lit(None).cast(pl.Float32).alias(SPOT_COLUMN))

    cached = SPOT_CACHE.get(label)
    if cached is None:
        return df.with_columns(pl.lit(None).cast(pl.Float32).alias(SPOT_COLUMN))

    joined = df.join(cached, on="Time", how="left")
    if SPOT_COLUMN not in joined.columns:
        joined = joined.with_columns(pl.lit(None).cast(pl.Float32).alias(SPOT_COLUMN))
    else:
        joined = joined.with_columns(pl.col(SPOT_COLUMN).cast(pl.Float32))
    return joined


def _attach_future_spot_features(df, is_future):
    """
    Future-Spot difference features add karta hai
    
    Args:
        df: Polars DataFrame
        is_future:  Boolean - agar True to features calculate karo, else None columns add karo
    
    Returns: 
        DataFrame with future-spot diff features
    """
    # Agar future nahi hai to None columns add karo
    if not is_future:
        return df.with_columns([
            pl.lit(None).cast(pl.Float32).alias(FUT_SPOT_DIFF_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(FUT_SPOT_DIFF_CUMSUM_COLUMN),
        ])
    
    # Future hai to actual calculation karo
    # Step 1: Average - Spot ka difference calculate karo
    df = df.with_columns([
        pl.when(
            pl.col("Average").cast(pl.Float32).is_finite() & 
            pl.col(SPOT_COLUMN).cast(pl.Float32).is_finite()
        )
        .then(pl.col("Average").cast(pl.Float32) - pl.col(SPOT_COLUMN).cast(pl.Float32))
        .otherwise(None)
        .cast(pl.Float32)
        .alias(FUT_SPOT_DIFF_COLUMN)
    ])
    
    # Step 2: Cumsum calculate karo (null values ko skip karo)
    df = df.with_columns([
        pl.when(pl.col(FUT_SPOT_DIFF_COLUMN).is_null())
        .then(None)
        .otherwise(
            pl.col(FUT_SPOT_DIFF_COLUMN).fill_null(0.0).cum_sum()  # ✅ cum_sum
        )
        .cast(pl.Float32)
        .alias(FUT_SPOT_DIFF_CUMSUM_COLUMN)
    ])
    
    return df



def _attach_average_features(df):
    """Optimized: Average diff aur cumsum ek batch me calculate karta hai"""
    # Step 1: Average diff calculate karo
    avg_col = pl.col("Average").cast(pl.Float32)
    avg_clean = pl.when(avg_col.is_finite()).then(avg_col).otherwise(None)
    
    df = df.with_columns([
        avg_clean.diff().cast(pl.Float32).alias(AVG_DIFF_COLUMN)
    ])
    
    # Step 2: Cumsum aur products ek saath calculate karo
    vol_col = pl.col("Volume").cast(pl.Float32)
    avg_diff_filled = pl.col(AVG_DIFF_COLUMN).fill_null(0.0)
    
    df = df.with_columns([
        avg_diff_filled.cum_sum().cast(pl.Float32).alias(AVG_DIFF_CUMSUM_COLUMN),
        (avg_diff_filled * vol_col).cast(pl.Float32).alias(AVG_VOL_PROD_COLUMN),
    ])
    
    # Step 3: Product cumsum
    df = df.with_columns([
        pl.col(AVG_VOL_PROD_COLUMN).fill_null(0.0).cum_sum().cast(pl.Float32).alias(AVG_VOL_PROD_CUMSUM_COLUMN)
    ])
    
    return df


def _normalize_expiry_ms(expiry_ms: Optional[int]):
    if expiry_ms is None:
        return None
    try:
        expiry_ms_int = int(expiry_ms)
    except (TypeError, ValueError):
        return None
    expiry_dt = datetime.fromtimestamp(expiry_ms_int / 1000, tz=timezone.utc).astimezone(IST)
    expiry_dt = expiry_dt.replace(hour=15, minute=30, second=0, microsecond=0)
    return int(expiry_dt.astimezone(timezone.utc).timestamp() * 1000)


def _erf(x):
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    abs_x = np.abs(x)
    t = 1.0 / (1.0 + _ERF_P * abs_x)
    polynomial = (
        ((((_ERF_A5 * t + _ERF_A4) * t + _ERF_A3) * t + _ERF_A2) * t + _ERF_A1) * t
    )
    y = 1.0 - polynomial * np.exp(-abs_x * abs_x)
    return sign * y


def _norm_cdf(x):
    return 0.5 * (1.0 + _erf(x / SQRT_TWO))


def _black_76_price(F, K, T, r, sigma, is_call_mask):
    safe_T = np.maximum(T, 1e-12)
    safe_F = np.maximum(F, 1e-12)
    safe_K = np.maximum(K, 1e-12)
    sigma = np.maximum(sigma, SIGMA_LOW)
    sqrt_T = np.sqrt(safe_T)

    d1 = (np.log(safe_F / safe_K) + 0.5 * sigma**2 * safe_T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount = np.exp(-r * safe_T)

    call_price = discount * (safe_F * _norm_cdf(d1) - safe_K * _norm_cdf(d2))
    put_price = discount * (safe_K * _norm_cdf(-d2) - safe_F * _norm_cdf(-d1))
    return np.where(is_call_mask, call_price, put_price)


def _vectorized_black_76_iv(option_price, F, strike_value, T, r, is_call_mask):
    result = np.full(option_price.shape, np.nan, dtype=np.float32)
    valid_mask = (
        np.isfinite(option_price)
        & np.isfinite(F)
        & (option_price > 0)
        & (F > 0)
        & (T > 0)
    )
    if not np.any(valid_mask):
        return result

    valid_idx = np.nonzero(valid_mask)[0]
    option_price = option_price[valid_mask]
    F = F[valid_mask]
    T = T[valid_mask]
    calls = is_call_mask[valid_mask]
    strikes = np.full_like(F, strike_value, dtype=np.float64)

    intrinsic = np.where(calls, np.maximum(0.0, F - strikes), np.maximum(0.0, strikes - F))
    upper = np.where(calls, F, strikes)

    bounds_mask = (option_price >= intrinsic) & (option_price <= upper)
    if not np.any(bounds_mask):
        return result

    eligible_idx = valid_idx[bounds_mask]
    option_price = option_price[bounds_mask]
    F = F[bounds_mask]
    T = T[bounds_mask]
    calls = calls[bounds_mask]
    strikes = strikes[bounds_mask]

    sigma_low = np.full_like(option_price, SIGMA_LOW, dtype=np.float64)
    sigma_high = np.full_like(option_price, SIGMA_HIGH, dtype=np.float64)

    for _ in range(MAX_IV_ITERATIONS):
        sigma_mid = 0.5 * (sigma_low + sigma_high)
        price_mid = _black_76_price(F, strikes, T, r, sigma_mid, calls)
        above_mask = price_mid > option_price
        sigma_high = np.where(above_mask, sigma_mid, sigma_high)
        sigma_low = np.where(~above_mask, sigma_mid, sigma_low)
        if np.all(np.abs(price_mid - option_price) < SIGMA_PRECISION):
            break

    final_sigma = 0.5 * (sigma_low + sigma_high)
    final_price = _black_76_price(F, strikes, T, r, final_sigma, calls)

    tolerance = np.maximum(option_price * 0.01, SIGMA_PRECISION)
    good_mask = np.abs(final_price - option_price) <= tolerance
    iv_values = np.where(good_mask, final_sigma, np.nan).astype(np.float32)
    result[eligible_idx] = iv_values
    return result


def _vectorized_black_76_delta(F, strike_value, T, r, sigma, is_call_mask):
    result = np.full(F.shape, np.nan, dtype=np.float32)
    valid_mask = (
        np.isfinite(F)
        & np.isfinite(T)
        & np.isfinite(sigma)
        & (F > 0)
        & (T > 0)
        & (sigma > 0)
    )
    if not np.any(valid_mask):
        return result

    valid_indices = np.nonzero(valid_mask)[0]
    F = F[valid_mask]
    T = T[valid_mask]
    sigma = sigma[valid_mask]
    calls = is_call_mask[valid_mask]
    strikes = np.full_like(F, strike_value, dtype=np.float64)

    safe_T = np.maximum(T, 1e-12)
    safe_F = np.maximum(F, 1e-12)
    safe_K = np.maximum(strikes, 1e-12)
    sigma = np.maximum(sigma, SIGMA_LOW)
    sqrt_T = np.sqrt(safe_T)

    d1 = (np.log(safe_F / safe_K) + 0.5 * sigma**2 * safe_T) / (sigma * sqrt_T)
    discount = np.exp(-r * safe_T)
    call_delta = discount * _norm_cdf(d1)
    put_delta = np.abs(discount * (_norm_cdf(d1) - 1.0))
    deltas = np.where(calls, call_delta, put_delta).astype(np.float32)
    result[valid_indices] = deltas
    return result


def _vectorized_black_76_vega_theta(F, strike_value, T, r, sigma, is_call_mask):
    vega_out = np.full(F.shape, np.nan, dtype=np.float32)
    theta_out = np.full(F.shape, np.nan, dtype=np.float32)
    valid_mask = (
        np.isfinite(F)
        & np.isfinite(T)
        & np.isfinite(sigma)
        & (F > 0)
        & (T > 0)
        & (sigma > 0)
    )
    if not np.any(valid_mask):
        return vega_out, theta_out

    idx = np.nonzero(valid_mask)[0]
    Fv = F[valid_mask]
    Tv = T[valid_mask]
    sig = sigma[valid_mask]
    calls = is_call_mask[valid_mask]
    K = np.full_like(Fv, strike_value, dtype=np.float64)

    safe_T = np.maximum(Tv, 1e-12)
    safe_F = np.maximum(Fv, 1e-12)
    safe_K = np.maximum(K, 1e-12)
    sig = np.maximum(sig, SIGMA_LOW)
    sqrt_T = np.sqrt(safe_T)

    d1 = (np.log(safe_F / safe_K) + 0.5 * sig**2 * safe_T) / (sig * sqrt_T)
    d2 = d1 - sig * sqrt_T
    discount = np.exp(-r * safe_T)
    pdf_d1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1 * d1)

    vega = discount * safe_F * sqrt_T * pdf_d1

    call_theta = -discount * (
        (safe_F * pdf_d1 * sig) / (2 * sqrt_T) + r * safe_K * _norm_cdf(d2)
    )
    put_theta = -discount * (
        (safe_F * pdf_d1 * sig) / (2 * sqrt_T) - r * safe_K * _norm_cdf(-d2)
    )
    theta = np.where(calls, call_theta, put_theta)

    vega_out[idx] = vega.astype(np.float32)
    theta_out[idx] = theta.astype(np.float32)
    return vega_out, theta_out


def _compute_option_iv(df, instrument_meta, risk_free_rate):
    option_type = str((instrument_meta or {}).get("option_type", "")).upper()
    if option_type not in {"CE", "PE"}:
        return _add_empty_option_metrics(df)

    strike = instrument_meta.get("strike")
    expiry_ms = _normalize_expiry_ms(instrument_meta.get("expiry"))
    if strike is None or expiry_ms is None:
        return _add_empty_option_metrics(df)

    try:
        strike_value = float(strike)
    except (TypeError, ValueError):
        return _add_empty_option_metrics(df)
    if strike_value <= 0:
        return _add_empty_option_metrics(df)

    spot_series = df.get_column(SPOT_COLUMN)
    option_series = df.get_column("Average")
    time_ms_series = df.get_column("Time").dt.timestamp("ms")

    spot_values = spot_series.to_numpy().astype(np.float64)
    option_values = option_series.to_numpy().astype(np.float64)
    time_ms = time_ms_series.to_numpy()

    T_years = np.maximum(expiry_ms - time_ms, 0) / MS_IN_YEAR
    call_mask = np.full(option_values.shape, option_type == "CE", dtype=bool)

    # Forward price = Spot × e^(r×T)
    forward_values = spot_values * np.exp(risk_free_rate * T_years)

    iv_values = _vectorized_black_76_iv(
        option_values,
        forward_values,
        strike_value,
        T_years.astype(np.float64),
        risk_free_rate,
        call_mask,
    )

    delta_values = _vectorized_black_76_delta(
        forward_values,
        strike_value,
        T_years.astype(np.float64),
        risk_free_rate,
        iv_values.astype(np.float64),
        call_mask,
    )

    vega_values, theta_values = _vectorized_black_76_vega_theta(
        forward_values,
        strike_value,
        T_years.astype(np.float64),
        risk_free_rate,
        iv_values.astype(np.float64),
        call_mask,
    )

    time_value = np.full(option_values.shape, np.nan, dtype=np.float64)
    valid_time_mask = np.isfinite(option_values) & np.isfinite(spot_values)
    if np.any(valid_time_mask):
        intrinsic_values = np.where(
            call_mask,
            spot_values - strike_value,
            strike_value - spot_values,
        )
        time_value[valid_time_mask] = (
            option_values[valid_time_mask] - intrinsic_values[valid_time_mask]
        )

    iv_series = pl.Series(IV_COLUMN, iv_values * 100.0).cast(pl.Float32)
    delta_series = pl.Series(DELTA_COLUMN, delta_values).cast(pl.Float32)
    time_value_series = pl.Series(
        TIMEVALUE_COLUMN, time_value.astype(np.float32)
    ).cast(pl.Float32)
    vega_series = pl.Series(VEGA_COLUMN, vega_values).cast(pl.Float32)
    theta_series = pl.Series(THETA_COLUMN, theta_values).cast(pl.Float32)
    return df.with_columns(iv_series, delta_series, vega_series, theta_series, time_value_series)


def _ensure_iv_column(df):
    if IV_COLUMN not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float32).alias(IV_COLUMN))
    return df


def _ensure_time_value_column(df):
    if TIMEVALUE_COLUMN not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float32).alias(TIMEVALUE_COLUMN))
    return df


def _add_empty_option_metrics(df):
    return df.with_columns(
        pl.lit(None).cast(pl.Float32).alias(IV_COLUMN),
        pl.lit(None).cast(pl.Float32).alias(DELTA_COLUMN),
        pl.lit(None).cast(pl.Float32).alias(TIMEVALUE_COLUMN),
    )


def _attach_enhancements(df, instrument_meta, risk_free_rate):
    # print("_attach_enhancements enter", file=sys.stderr, flush=True)
    
    is_option = _is_option_contract(instrument_meta)
    is_future = str((instrument_meta or {}).get("category", "")).lower() in {
        "index_future",
        "equity_future",
    }
    
    df = _attach_spot_average(df, instrument_meta)
    
    if is_option:
        df = _compute_option_iv(df, instrument_meta, risk_free_rate)
    else:
        df = _add_empty_option_metrics(df)
        
    if SPOT_COLUMN not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float32).alias(SPOT_COLUMN))

#     print(
#     df.select(
#         pl.col("Time"),
#         pl.col("Average"),
#         pl.col("SpotAverage"),
#     ).head(5)
# )
           
        
    df = _ensure_iv_column(df)
    
    if DELTA_COLUMN not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float32).alias(DELTA_COLUMN))
        
    if VEGA_COLUMN not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float32).alias(VEGA_COLUMN))
        
    if THETA_COLUMN not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float32).alias(THETA_COLUMN))
    
    df = _ensure_time_value_column(df)
    df = _attach_option_diff_features(df, is_option)
    df = _attach_future_spot_features(df, is_future)
    
    df = _attach_average_features(df)

    return df

def _add_diff_cumsum_batch(df, source_columns_map):
    """
    Optimized: Multiple columns ka diff aur cumsum ek hi batch me calculate karta hai.
    
    Args:
        df: Polars DataFrame
        source_columns_map: Dict of {source_col: (diff_col_name, cumsum_col_name)}
    
    Returns:
        DataFrame with all diff and cumsum columns added efficiently
    """
    # Step 1: Saare diff columns ek saath add karo
    diff_exprs = []
    for source_col, (diff_col_name, _) in source_columns_map.items():
        diff_exprs.append(
            pl.when(pl.col(source_col).is_finite())
            .then(pl.col(source_col))
            .otherwise(None)
            .diff()
            .cast(pl.Float32)
            .alias(diff_col_name)
        )
    
    if diff_exprs:
        df = df.with_columns(diff_exprs)
    
    # Step 2: Saare cumsum columns ek saath add karo
    cumsum_exprs = []
    for _, (diff_col_name, cumsum_col_name) in source_columns_map.items():
        cumsum_exprs.append(
            pl.col(diff_col_name)
            .fill_null(0.0)
            .cum_sum()
            .cast(pl.Float32)
            .alias(cumsum_col_name)
        )
    
    if cumsum_exprs:
        df = df.with_columns(cumsum_exprs)
    
    return df


def _add_product_cumsum_batch(df, products_config):
    """
    Optimized: Multiple product aur cumsum ek batch me calculate karta hai.
    
    Args:
        df: Polars DataFrame
        products_config: List of (diff_col, multiplier_col, prod_col, prod_cumsum_col)
    
    Returns:
        DataFrame with all product and cumsum columns added
    """
    # Step 1: Saare product columns ek saath
    prod_exprs = []
    for diff_col, mult_col, prod_col, _ in products_config:
        prod_exprs.append(
            (pl.col(diff_col).fill_null(0.0) * pl.col(mult_col).cast(pl.Float32))
            .cast(pl.Float32)
            .alias(prod_col)
        )
    
    if prod_exprs:
        df = df.with_columns(prod_exprs)
    
    # Step 2: Saare cumsum columns ek saath
    cumsum_exprs = []
    for _, _, prod_col, prod_cumsum_col in products_config:
        cumsum_exprs.append(
            pl.col(prod_col)
            .fill_null(0.0)
            .cum_sum()
            .cast(pl.Float32)
            .alias(prod_cumsum_col)
        )
    
    if cumsum_exprs:
        df = df.with_columns(cumsum_exprs)
    
    return df


def _attach_option_diff_features(df, is_option):
    """
    Option-specific diff aur cumsum features add karta hai - OPTIMIZED VERSION
    
    Args: 
        df: Polars DataFrame
        is_option: Boolean - agar True to features calculate karo, else None columns add karo
    
    Returns: 
        DataFrame with option diff features
    """
    # Agar option nahi hai to sab columns None se fill karo (ek hi call me)
    if not is_option:
        return df.with_columns([
            pl.lit(None).cast(pl.Float32).alias(IV_DIFF_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(IV_DIFF_CUMSUM_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(DELTA_DIFF_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(DELTA_DIFF_CUMSUM_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(VEGA_DIFF_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(VEGA_DIFF_CUMSUM_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(THETA_DIFF_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(THETA_DIFF_CUMSUM_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(TIMEVALUE_DIFF_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(TIMEVALUE_DIFF_CUMSUM_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(TIMEVALUE_VOL_PROD_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(TIMEVALUE_VOL_PROD_CUMSUM_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(OI_DIFF_COLUMN),
            pl.lit(None).cast(pl.Float32).alias(OI_DIFF_CUMSUM_COLUMN),
        ])

    # Option hai to batch me calculations karo
    source_columns_map = {
        IV_COLUMN: (IV_DIFF_COLUMN, IV_DIFF_CUMSUM_COLUMN),
        DELTA_COLUMN: (DELTA_DIFF_COLUMN, DELTA_DIFF_CUMSUM_COLUMN),
        VEGA_COLUMN: (VEGA_DIFF_COLUMN, VEGA_DIFF_CUMSUM_COLUMN),
        THETA_COLUMN: (THETA_DIFF_COLUMN, THETA_DIFF_CUMSUM_COLUMN),
        TIMEVALUE_COLUMN: (TIMEVALUE_DIFF_COLUMN, TIMEVALUE_DIFF_CUMSUM_COLUMN),
        "OpenInterest": (OI_DIFF_COLUMN, OI_DIFF_CUMSUM_COLUMN),
    }
    
    df = _add_diff_cumsum_batch(df, source_columns_map)
    
    # TimeValue * Volume product
    products_config = [
        (TIMEVALUE_DIFF_COLUMN, "Volume", TIMEVALUE_VOL_PROD_COLUMN, TIMEVALUE_VOL_PROD_CUMSUM_COLUMN),
    ]
    df = _add_product_cumsum_batch(df, products_config)
    
    return df

    # Option hai to batch me calculations karo
    source_columns_map = {
        IV_COLUMN: (IV_DIFF_COLUMN, IV_DIFF_CUMSUM_COLUMN),
        DELTA_COLUMN: (DELTA_DIFF_COLUMN, DELTA_DIFF_CUMSUM_COLUMN),
        VEGA_COLUMN: (VEGA_DIFF_COLUMN, VEGA_DIFF_CUMSUM_COLUMN),
        THETA_COLUMN: (THETA_DIFF_COLUMN, THETA_DIFF_CUMSUM_COLUMN),
        TIMEVALUE_COLUMN: (TIMEVALUE_DIFF_COLUMN, TIMEVALUE_DIFF_CUMSUM_COLUMN),
        "OpenInterest": (OI_DIFF_COLUMN, OI_DIFF_CUMSUM_COLUMN),
    }
    
    df = _add_diff_cumsum_batch(df, source_columns_map)
    
    # TimeValue * Volume product
    products_config = [
        (TIMEVALUE_DIFF_COLUMN, "Volume", TIMEVALUE_VOL_PROD_COLUMN, TIMEVALUE_VOL_PROD_CUMSUM_COLUMN),
    ]
    df = _add_product_cumsum_batch(df, products_config)
    
    return df

# def _attach_option_diff_features(df, is_option):
#     if not is_option:
#         return df.with_columns(
#             pl.lit(None).cast(pl.Float32).alias(IV_DIFF_COLUMN),
#             pl.lit(None).cast(pl.Float32).alias(IV_DIFF_CUMSUM_COLUMN),
#             pl.lit(None).cast(pl.Float32).alias(DELTA_DIFF_COLUMN),
#             pl.lit(None).cast(pl.Float32).alias(DELTA_DIFF_CUMSUM_COLUMN),
#             pl.lit(None).cast(pl.Float32).alias(VEGA_DIFF_COLUMN),
#             pl.lit(None).cast(pl.Float32).alias(VEGA_DIFF_CUMSUM_COLUMN),
#             pl.lit(None).cast(pl.Float32).alias(THETA_DIFF_COLUMN),
#             pl.lit(None).cast(pl.Float32).alias(THETA_DIFF_CUMSUM_COLUMN),
#             pl.lit(None).cast(pl.Float32).alias(TIMEVALUE_DIFF_COLUMN),
#             pl.lit(None).cast(pl.Float32).alias(TIMEVALUE_DIFF_CUMSUM_COLUMN),
#             pl.lit(None).cast(pl.Float32).alias(TIMEVALUE_VOL_PROD_COLUMN),
#             pl.lit(None).cast(pl.Float32).alias(TIMEVALUE_VOL_PROD_CUMSUM_COLUMN),
#             pl.lit(None).cast(pl.Float32).alias(OI_DIFF_COLUMN),
#             pl.lit(None).cast(pl.Float32).alias(OI_DIFF_CUMSUM_COLUMN),
#         )

#     iv_col = pl.col(IV_COLUMN)
#     iv_clean = pl.when(iv_col.is_finite()).then(iv_col).otherwise(None)
#     iv_diff = iv_clean.diff().cast(pl.Float32).alias(IV_DIFF_COLUMN)
#     iv_cumsum = (
#         pl.when(iv_diff.is_null()).then(0.0).otherwise(iv_diff)
#         .cumsum()
#         .cast(pl.Float32)
#         .alias(IV_DIFF_CUMSUM_COLUMN)
#     )

#     delta_col = pl.col(DELTA_COLUMN)
#     delta_clean = pl.when(delta_col.is_finite()).then(delta_col).otherwise(None)
#     delta_diff = delta_clean.diff().cast(pl.Float32).alias(DELTA_DIFF_COLUMN)
#     delta_diff_filled = pl.when(delta_diff.is_null()).then(0.0).otherwise(delta_diff)
#     delta_cumsum = (
#         delta_diff_filled.cumsum().cast(pl.Float32).alias(DELTA_DIFF_CUMSUM_COLUMN)
#     )

#     vega_col = pl.col(VEGA_COLUMN)
#     vega_clean = pl.when(vega_col.is_finite()).then(vega_col).otherwise(None)
#     vega_diff = vega_clean.diff().cast(pl.Float32).alias(VEGA_DIFF_COLUMN)
#     vega_diff_filled = pl.when(vega_diff.is_null()).then(0.0).otherwise(vega_diff)
#     vega_cumsum = (
#         vega_diff_filled.cumsum().cast(pl.Float32).alias(VEGA_DIFF_CUMSUM_COLUMN)
#     )

#     theta_col = pl.col(THETA_COLUMN)
#     theta_clean = pl.when(theta_col.is_finite()).then(theta_col).otherwise(None)
#     theta_diff = theta_clean.diff().cast(pl.Float32).alias(THETA_DIFF_COLUMN)
#     theta_diff_filled = pl.when(theta_diff.is_null()).then(0.0).otherwise(theta_diff)
#     theta_cumsum = (
#         theta_diff_filled.cumsum().cast(pl.Float32).alias(THETA_DIFF_CUMSUM_COLUMN)
#     )
#     tv_col = pl.col(TIMEVALUE_COLUMN)
#     tv_clean = pl.when(tv_col.is_finite()).then(tv_col).otherwise(None)
#     tv_diff = tv_clean.diff().cast(pl.Float32).alias(TIMEVALUE_DIFF_COLUMN)
#     tv_diff_filled = pl.when(tv_diff.is_null()).then(0.0).otherwise(tv_diff)
#     tv_cumsum = (
#         tv_diff_filled.cumsum().cast(pl.Float32).alias(TIMEVALUE_DIFF_CUMSUM_COLUMN)
#     )

#     vol_col = pl.col("Volume").cast(pl.Float32)
#     tv_vol_prod = (
#         (tv_diff_filled * vol_col)
#         .cast(pl.Float32)
#         .alias(TIMEVALUE_VOL_PROD_COLUMN)
#     )
#     tv_vol_prod_cumsum = (
#         tv_vol_prod.fill_null(0.0)
#         .cumsum()
#         .cast(pl.Float32)
#         .alias(TIMEVALUE_VOL_PROD_CUMSUM_COLUMN)
#     )

#     oi_col = pl.col("OpenInterest")
#     oi_clean = pl.when(oi_col.is_finite()).then(oi_col).otherwise(None)
#     oi_diff = oi_clean.diff().cast(pl.Float32).alias(OI_DIFF_COLUMN)
#     oi_diff_filled = pl.when(oi_diff.is_null()).then(0.0).otherwise(oi_diff)
#     oi_cumsum = (
#         oi_diff_filled.cumsum().cast(pl.Float32).alias(OI_DIFF_CUMSUM_COLUMN)
#     )

#     return df.with_columns(
#         iv_diff,
#         iv_cumsum,
#         delta_diff,
#         delta_cumsum,
#         vega_diff,
#         vega_cumsum,
#         theta_diff,
#         theta_cumsum,
#         tv_diff,
#         tv_cumsum,
#         tv_vol_prod,
#         tv_vol_prod_cumsum,
#         oi_diff,
#         oi_cumsum,
#     )




def transform_candles_for_today(
    candles,
    *,
    tz=IST,
    instrument_meta=None,
    risk_free_rate=RISK_FREE_RATE,
):
    """Normalize, reverse, and filter candles to the current trading date."""
    today = datetime.now(tz=tz).date()

    # print("transform_candles_for_today enter", file=sys.stderr, flush=True)
    
    df = _ensure_frame(candles)
    if df.is_empty():
        return [], today.isoformat()

    df = _format_time(df, tz).reverse()
    df, trading_date = _filter_to_trading_date(df, today)
    
    df = df.select(["Time", "Average", "Volume", "OpenInterest"])
    
    df = _attach_enhancements(df, instrument_meta, risk_free_rate)
    # print(df.head(5), file=sys.stderr, flush=True)  # DEBUG REMOVED
    columns = [
        "Time",
        "Average",
        "Volume",
        "OpenInterest",
        SPOT_COLUMN,
        IV_COLUMN,
        DELTA_COLUMN,
        VEGA_COLUMN,
        THETA_COLUMN,
        TIMEVALUE_COLUMN,
        IV_DIFF_COLUMN,
        IV_DIFF_CUMSUM_COLUMN,
        DELTA_DIFF_COLUMN,
        DELTA_DIFF_CUMSUM_COLUMN,
        VEGA_DIFF_COLUMN,
        VEGA_DIFF_CUMSUM_COLUMN,
        THETA_DIFF_COLUMN,
        THETA_DIFF_CUMSUM_COLUMN,
        TIMEVALUE_DIFF_COLUMN,
        TIMEVALUE_DIFF_CUMSUM_COLUMN,
        TIMEVALUE_VOL_PROD_COLUMN,
        TIMEVALUE_VOL_PROD_CUMSUM_COLUMN,
        OI_DIFF_COLUMN,
        OI_DIFF_CUMSUM_COLUMN,
        AVG_DIFF_COLUMN,
        AVG_DIFF_CUMSUM_COLUMN,
        AVG_VOL_PROD_COLUMN,
        AVG_VOL_PROD_CUMSUM_COLUMN,
        FUT_SPOT_DIFF_COLUMN,
        FUT_SPOT_DIFF_CUMSUM_COLUMN,
    ]
    df = df.select([column for column in columns if column in df.columns])
    
    return _records_from_frame(df), trading_date.isoformat()


def normalize_candles(
    candles,
    *,
    tz=IST,
):
    """Normalize raw candles without filtering by date."""
    df = _ensure_frame(candles)
    if df.is_empty():
        return []
    df = _format_time(df, tz).reverse()
    return _records_from_frame(df)


__all__ = ["transform_candles_for_today", "normalize_candles", "CANDLE_COLUMNS"]
