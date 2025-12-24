#!/usr/bin/env python3

from collections import deque
from datetime import datetime, timezone, time as time_cls
from threading import RLock
from typing import Dict, List, Optional

import numpy as np
from zoneinfo import ZoneInfo
from live_updates import publish as publish_live_update

IST = ZoneInfo("Asia/Kolkata")
ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
DEFAULT_NUMPY_CAPACITY = 120_000


def _parse_timestamp_ms(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000)
    if isinstance(value, str):
        try:
            dt = datetime.strptime(value, ISO_TIMESTAMP_FORMAT)
        except ValueError:
            return None
        return int(dt.timestamp() * 1000)
    return None


def _nan_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _float_or_default(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _seconds_of_day_from_timestamp(ts_ms):
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).astimezone(IST)
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def _time_string_from_seconds(seconds, trading_date):
    if trading_date:
        try:
            date_obj = datetime.strptime(trading_date, "%Y-%m-%d").date()
        except ValueError:
            date_obj = datetime.now(tz=IST).date()
    else:
        date_obj = datetime.now(tz=IST).date()
    hours = max(0, int(seconds)) // 3600
    minutes = (max(0, int(seconds)) % 3600) // 60
    secs = max(0, int(seconds)) % 60
    composed = datetime.combine(
        date_obj,
        time_cls(hour=hours % 24, minute=minutes % 60, second=secs % 60),
        tzinfo=IST,
    )
    return composed.strftime(ISO_TIMESTAMP_FORMAT)


class CandleBuffer:
    def __init__(self, capacity=DEFAULT_NUMPY_CAPACITY):
        self.capacity = max(1, int(capacity))
        self.time_seconds = np.zeros(self.capacity, dtype=np.int32)
        self.average = np.zeros(self.capacity, dtype=np.float32)
        self.volume = np.zeros(self.capacity, dtype=np.float32)
        self.open_interest = np.zeros(self.capacity, dtype=np.float32)
        self.spot = np.zeros(self.capacity, dtype=np.float32)
        self.iv = np.zeros(self.capacity, dtype=np.float32)
        self.delta = np.zeros(self.capacity, dtype=np.float32)
        self.vega = np.zeros(self.capacity, dtype=np.float32)
        self.theta = np.zeros(self.capacity, dtype=np.float32)
        self.avg_diff = np.zeros(self.capacity, dtype=np.float32)
        self.avg_diff_cumsum = np.zeros(self.capacity, dtype=np.float32)
        self.avg_vol_prod = np.zeros(self.capacity, dtype=np.float32)
        self.avg_vol_prod_cumsum = np.zeros(self.capacity, dtype=np.float32)
        self.future_spot_diff = np.zeros(self.capacity, dtype=np.float32)
        self.future_spot_diff_cumsum = np.zeros(self.capacity, dtype=np.float32)
        self.iv_diff = np.zeros(self.capacity, dtype=np.float32)
        self.iv_diff_cumsum = np.zeros(self.capacity, dtype=np.float32)
        self.delta_diff = np.zeros(self.capacity, dtype=np.float32)
        self.delta_diff_cumsum = np.zeros(self.capacity, dtype=np.float32)
        self.vega_diff = np.zeros(self.capacity, dtype=np.float32)
        self.vega_diff_cumsum = np.zeros(self.capacity, dtype=np.float32)
        self.theta_diff = np.zeros(self.capacity, dtype=np.float32)
        self.theta_diff_cumsum = np.zeros(self.capacity, dtype=np.float32)
        self.timevalue_diff = np.zeros(self.capacity, dtype=np.float32)
        self.timevalue_diff_cumsum = np.zeros(self.capacity, dtype=np.float32)
        self.timevalue_vol_prod = np.zeros(self.capacity, dtype=np.float32)
        self.timevalue_vol_prod_cumsum = np.zeros(self.capacity, dtype=np.float32)
        self.oi_diff = np.zeros(self.capacity, dtype=np.float32)
        self.oi_diff_cumsum = np.zeros(self.capacity, dtype=np.float32)
        self.head = 0
        self.size = 0
        self.latest_ts_ms: Optional[int] = None
        self.trading_date: Optional[str] = None
        self.metadata: Dict = {}

    def reset_for_date(self, candle_date: Optional[str]):
        if candle_date and candle_date != self.trading_date:
            self.trading_date = candle_date
            self.head = 0
            self.size = 0
            self.latest_ts_ms = None

    def update_metadata(self, record: dict):
        self.metadata = {
            "instrument_key": record.get("instrument_key"),
            "trading_symbol": record.get("trading_symbol"),
            "category": record.get("category"),
            "instrument_meta": record.get("instrument_meta", {}),
            "meta": record.get("meta", {}),
            "from_timestamp_ms": record.get("from_timestamp_ms"),
            "fetched_at": record.get("fetched_at"),
            "candle_date": record.get("candle_date"),
        }

    def append_many(self, candles: List[dict]):
        """Optimized vectorized append - processes all candles in batch."""
        if not candles:
            return
        
        # Step 1: Parse and filter candles - keep only valid ones
        valid_candles = []
        for candle in candles:
            ts_ms = _parse_timestamp_ms(candle.get("Time"))
            if ts_ms is None:
                continue
            # Skip if already processed
            if self.latest_ts_ms is not None and ts_ms <= self.latest_ts_ms:
                continue
            valid_candles.append((ts_ms, candle))
        
        if not valid_candles:
            return
        
        # Sort by timestamp
        valid_candles.sort(key=lambda x: x[0])
        
        n = len(valid_candles)
        
        # Step 2: Extract all values into numpy arrays (vectorized)
        ts_ms_arr = np.array([c[0] for c in valid_candles], dtype=np.int64)
        seconds_arr = np.array([_seconds_of_day_from_timestamp(c[0]) for c in valid_candles], dtype=np.int32)
        
        # Helper to extract column as numpy array
        def extract_float(key, default=np.nan):
            return np.array([_nan_float(c[1].get(key)) if default is np.nan else _float_or_default(c[1].get(key), default) for c in valid_candles], dtype=np.float32)
        
        avg_arr = extract_float("Average")
        volume_arr = extract_float("Volume", 0.0)
        oi_arr = extract_float("OpenInterest", 0.0)
        spot_arr = extract_float("SpotAverage")
        iv_arr = extract_float("IV")
        delta_arr = extract_float("Delta")
        vega_arr = extract_float("Vega")
        theta_arr = extract_float("Theta")
        avg_diff_arr = extract_float("AverageDiff")
        fut_spot_diff_arr = extract_float("FutureSpotDiff")
        iv_diff_arr = extract_float("IVDiff")
        delta_diff_arr = extract_float("DeltaDiff")
        vega_diff_arr = extract_float("VegaDiff")
        theta_diff_arr = extract_float("ThetaDiff")
        tv_diff_arr = extract_float("TimeValueDiff")
        oi_diff_arr = extract_float("OpenInterestDiff")
        
        # Step 3: Get previous cumsum values
        def _prev_cumsum(array):
            if self.size == 0:
                return 0.0
            idx = (self.head - 1) % self.capacity
            val = float(array[idx])
            return 0.0 if np.isnan(val) else val

        # Step 4: Compute cumsum arrays vectorized
        def compute_cumsum(diff_arr, prev_cumsum):
            safe_diff = np.where(np.isfinite(diff_arr), diff_arr, 0.0)
            cumsum = np.cumsum(safe_diff) + prev_cumsum
            return cumsum.astype(np.float32)
        
        def compute_prod_cumsum(diff_arr, mult_arr, prev_cumsum):
            safe_diff = np.where(np.isfinite(diff_arr), diff_arr, 0.0)
            safe_mult = np.where(np.isfinite(mult_arr), mult_arr, 0.0)
            prod = safe_diff * safe_mult
            cumsum = np.cumsum(prod) + prev_cumsum
            return prod.astype(np.float32), cumsum.astype(np.float32)
        
        avg_diff_cumsum_arr = compute_cumsum(avg_diff_arr, _prev_cumsum(self.avg_diff_cumsum))
        avg_vol_prod_arr, avg_vol_prod_cumsum_arr = compute_prod_cumsum(avg_diff_arr, volume_arr, _prev_cumsum(self.avg_vol_prod_cumsum))
        fut_spot_diff_cumsum_arr = compute_cumsum(fut_spot_diff_arr, _prev_cumsum(self.future_spot_diff_cumsum))
        iv_diff_cumsum_arr = compute_cumsum(iv_diff_arr, _prev_cumsum(self.iv_diff_cumsum))
        delta_diff_cumsum_arr = compute_cumsum(delta_diff_arr, _prev_cumsum(self.delta_diff_cumsum))
        vega_diff_cumsum_arr = compute_cumsum(vega_diff_arr, _prev_cumsum(self.vega_diff_cumsum))
        theta_diff_cumsum_arr = compute_cumsum(theta_diff_arr, _prev_cumsum(self.theta_diff_cumsum))
        tv_diff_cumsum_arr = compute_cumsum(tv_diff_arr, _prev_cumsum(self.timevalue_diff_cumsum))
        tv_vol_prod_arr, tv_vol_prod_cumsum_arr = compute_prod_cumsum(tv_diff_arr, volume_arr, _prev_cumsum(self.timevalue_vol_prod_cumsum))
        oi_diff_cumsum_arr = compute_cumsum(oi_diff_arr, _prev_cumsum(self.oi_diff_cumsum))
        
        # Step 5: Batch insert into buffer
        for i in range(n):
            idx = self.head
            self.time_seconds[idx] = seconds_arr[i]
            self.average[idx] = avg_arr[i]
            self.volume[idx] = volume_arr[i]
            self.open_interest[idx] = oi_arr[i]
            self.spot[idx] = spot_arr[i]
            self.iv[idx] = iv_arr[i]
            self.delta[idx] = delta_arr[i]
            self.vega[idx] = vega_arr[i]
            self.theta[idx] = theta_arr[i]
            self.avg_diff[idx] = avg_diff_arr[i]
            self.avg_diff_cumsum[idx] = avg_diff_cumsum_arr[i]
            self.avg_vol_prod[idx] = avg_vol_prod_arr[i]
            self.avg_vol_prod_cumsum[idx] = avg_vol_prod_cumsum_arr[i]
            self.future_spot_diff[idx] = fut_spot_diff_arr[i]
            self.future_spot_diff_cumsum[idx] = fut_spot_diff_cumsum_arr[i]
            self.iv_diff[idx] = iv_diff_arr[i]
            self.iv_diff_cumsum[idx] = iv_diff_cumsum_arr[i]
            self.delta_diff[idx] = delta_diff_arr[i]
            self.delta_diff_cumsum[idx] = delta_diff_cumsum_arr[i]
            self.vega_diff[idx] = vega_diff_arr[i]
            self.vega_diff_cumsum[idx] = vega_diff_cumsum_arr[i]
            self.theta_diff[idx] = theta_diff_arr[i]
            self.theta_diff_cumsum[idx] = theta_diff_cumsum_arr[i]
            self.timevalue_diff[idx] = tv_diff_arr[i]
            self.timevalue_diff_cumsum[idx] = tv_diff_cumsum_arr[i]
            self.timevalue_vol_prod[idx] = tv_vol_prod_arr[i]
            self.timevalue_vol_prod_cumsum[idx] = tv_vol_prod_cumsum_arr[i]
            self.oi_diff[idx] = oi_diff_arr[i]
            self.oi_diff_cumsum[idx] = oi_diff_cumsum_arr[i]
            
            self.head = (self.head + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1
            self.latest_ts_ms = ts_ms_arr[i]

    def _append_row(
        self,
        ts_ms,
        seconds,
        avg,
        volume,
        oi,
        spot,
        iv,
        delta,
        vega,
        theta,
        avg_diff,
        avg_diff_cumsum,
        avg_vol_prod,
        avg_vol_prod_cumsum,
        fut_spot_diff,
        fut_spot_diff_cumsum,
        iv_diff,
        iv_diff_cumsum,
        delta_diff,
        delta_diff_cumsum,
        vega_diff,
        vega_diff_cumsum,
        theta_diff,
        theta_diff_cumsum,
        tv_diff,
        tv_diff_cumsum,
        tv_vol_prod,
        tv_vol_prod_cumsum,
        oi_diff,
        oi_diff_cumsum,
    ):
        idx = self.head
        self.time_seconds[idx] = seconds
        self.average[idx] = avg
        self.volume[idx] = volume
        self.open_interest[idx] = oi
        self.spot[idx] = spot
        self.iv[idx] = iv
        self.delta[idx] = delta
        self.vega[idx] = vega
        self.theta[idx] = theta
        self.avg_diff[idx] = avg_diff
        self.avg_diff_cumsum[idx] = avg_diff_cumsum
        self.avg_vol_prod[idx] = avg_vol_prod
        self.avg_vol_prod_cumsum[idx] = avg_vol_prod_cumsum
        self.future_spot_diff[idx] = fut_spot_diff
        self.future_spot_diff_cumsum[idx] = fut_spot_diff_cumsum
        self.iv_diff[idx] = iv_diff
        self.iv_diff_cumsum[idx] = iv_diff_cumsum
        self.delta_diff[idx] = delta_diff
        self.delta_diff_cumsum[idx] = delta_diff_cumsum
        self.vega_diff[idx] = vega_diff
        self.vega_diff_cumsum[idx] = vega_diff_cumsum
        self.theta_diff[idx] = theta_diff
        self.theta_diff_cumsum[idx] = theta_diff_cumsum
        self.timevalue_diff[idx] = tv_diff
        self.timevalue_diff_cumsum[idx] = tv_diff_cumsum
        self.timevalue_vol_prod[idx] = tv_vol_prod
        self.timevalue_vol_prod_cumsum[idx] = tv_vol_prod_cumsum
        self.oi_diff[idx] = oi_diff
        self.oi_diff_cumsum[idx] = oi_diff_cumsum
        self.head = (self.head + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        self.latest_ts_ms = ts_ms

    def _ordered_indices(self):
        if self.size == 0:
            return np.array([], dtype=np.int64)
        start = (self.head - self.size) % self.capacity
        return (np.arange(self.size, dtype=np.int64) + start) % self.capacity

    def to_numpy_dict(self):
        idxs = self._ordered_indices()
        if idxs.size == 0:
            return {}
        return {
            "time_seconds": self.time_seconds[idxs].copy(),
            "average": self.average[idxs].copy(),
            "volume": self.volume[idxs].copy(),
            "open_interest": self.open_interest[idxs].copy(),
            "spot": self.spot[idxs].copy(),
            "iv": self.iv[idxs].copy(),
            "delta": self.delta[idxs].copy(),
            "vega": self.vega[idxs].copy(),
            "theta": self.theta[idxs].copy(),
            "average_diff": self.avg_diff[idxs].copy(),
            "average_diff_cumsum": self.avg_diff_cumsum[idxs].copy(),
            "average_vol_prod": self.avg_vol_prod[idxs].copy(),
            "average_vol_prod_cumsum": self.avg_vol_prod_cumsum[idxs].copy(),
            "future_spot_diff": self.future_spot_diff[idxs].copy(),
            "future_spot_diff_cumsum": self.future_spot_diff_cumsum[idxs].copy(),
            "iv_diff": self.iv_diff[idxs].copy(),
            "iv_diff_cumsum": self.iv_diff_cumsum[idxs].copy(),
            "delta_diff": self.delta_diff[idxs].copy(),
            "delta_diff_cumsum": self.delta_diff_cumsum[idxs].copy(),
            "vega_diff": self.vega_diff[idxs].copy(),
            "vega_diff_cumsum": self.vega_diff_cumsum[idxs].copy(),
            "theta_diff": self.theta_diff[idxs].copy(),
            "theta_diff_cumsum": self.theta_diff_cumsum[idxs].copy(),
            "timevalue_diff": self.timevalue_diff[idxs].copy(),
            "timevalue_diff_cumsum": self.timevalue_diff_cumsum[idxs].copy(),
            "timevalue_vol_prod": self.timevalue_vol_prod[idxs].copy(),
            "timevalue_vol_prod_cumsum": self.timevalue_vol_prod_cumsum[idxs].copy(),
            "oi_diff": self.oi_diff[idxs].copy(),
            "oi_diff_cumsum": self.oi_diff_cumsum[idxs].copy(),
            "trading_date": self.trading_date,
        }

    def to_records(self):
        idxs = self._ordered_indices()
        records = []
        for idx in idxs:
            record = {
                "Time": _time_string_from_seconds(
                    int(self.time_seconds[idx]), self.trading_date
                ),
                "Average": float(self.average[idx]),
                "Volume": float(self.volume[idx]),
                "OpenInterest": float(self.open_interest[idx]),
            }
            spot = float(self.spot[idx])
            iv = float(self.iv[idx])
            delta = float(self.delta[idx])
            vega = float(self.vega[idx])
            theta = float(self.theta[idx])
            avg_diff_val = float(self.avg_diff[idx])
            avg_diff_cum_val = float(self.avg_diff_cumsum[idx])
            avg_prod_val = float(self.avg_vol_prod[idx])
            avg_prod_cum_val = float(self.avg_vol_prod_cumsum[idx])
            record["SpotAverage"] = None if np.isnan(spot) else spot
            record["IV"] = None if np.isnan(iv) else iv
            record["Delta"] = None if np.isnan(delta) else delta
            record["Vega"] = None if np.isnan(vega) else vega
            record["Theta"] = None if np.isnan(theta) else theta
            record["AverageDiff"] = None if np.isnan(avg_diff_val) else avg_diff_val
            record["AverageDiffCumSum"] = (
                None if np.isnan(avg_diff_cum_val) else avg_diff_cum_val
            )
            record["AverageVolProd"] = None if np.isnan(avg_prod_val) else avg_prod_val
            record["AverageVolProdCumSum"] = (
                None if np.isnan(avg_prod_cum_val) else avg_prod_cum_val
            )
            fut_diff_val = float(self.future_spot_diff[idx])
            fut_diff_cum_val = float(self.future_spot_diff_cumsum[idx])
            record["FutureSpotDiff"] = None if np.isnan(fut_diff_val) else fut_diff_val
            record["FutureSpotDiffCumSum"] = (
                None if np.isnan(fut_diff_cum_val) else fut_diff_cum_val
            )
            iv_diff = float(self.iv_diff[idx])
            iv_diff_cumsum = float(self.iv_diff_cumsum[idx])
            record["IVDiff"] = None if np.isnan(iv_diff) else iv_diff
            record["IVDiffCumSum"] = None if np.isnan(iv_diff_cumsum) else iv_diff_cumsum
            delta_diff_val = float(self.delta_diff[idx])
            delta_diff_cum_val = float(self.delta_diff_cumsum[idx])
            record["DeltaDiff"] = None if np.isnan(delta_diff_val) else delta_diff_val
            record["DeltaDiffCumSum"] = (
                None if np.isnan(delta_diff_cum_val) else delta_diff_cum_val
            )
            vega_diff_val = float(self.vega_diff[idx])
            vega_diff_cum_val = float(self.vega_diff_cumsum[idx])
            record["VegaDiff"] = None if np.isnan(vega_diff_val) else vega_diff_val
            record["VegaDiffCumSum"] = (
                None if np.isnan(vega_diff_cum_val) else vega_diff_cum_val
            )
            theta_diff_val = float(self.theta_diff[idx])
            theta_diff_cum_val = float(self.theta_diff_cumsum[idx])
            record["ThetaDiff"] = None if np.isnan(theta_diff_val) else theta_diff_val
            record["ThetaDiffCumSum"] = (
                None if np.isnan(theta_diff_cum_val) else theta_diff_cum_val
            )
            tv_diff = float(self.timevalue_diff[idx])
            tv_diff_cumsum = float(self.timevalue_diff_cumsum[idx])
            tv_vol_prod = float(self.timevalue_vol_prod[idx])
            tv_vol_prod_cumsum = float(self.timevalue_vol_prod_cumsum[idx])
            record["TimeValueDiff"] = None if np.isnan(tv_diff) else tv_diff
            record["TimeValueDiffCumSum"] = None if np.isnan(tv_diff_cumsum) else tv_diff_cumsum
            record["TimeValueVolProd"] = None if np.isnan(tv_vol_prod) else tv_vol_prod
            record["TimeValueVolProdCumSum"] = (
                None if np.isnan(tv_vol_prod_cumsum) else tv_vol_prod_cumsum
            )
            oi_diff = float(self.oi_diff[idx])
            oi_diff_cumsum = float(self.oi_diff_cumsum[idx])
            record["OpenInterestDiff"] = None if np.isnan(oi_diff) else oi_diff
            record["OpenInterestDiffCumSum"] = (
                None if np.isnan(oi_diff_cumsum) else oi_diff_cumsum
            )
            records.append(record)
        return records

    def export_record(self):
        if self.size == 0:
            return {}
        base = dict(self.metadata)
        base.update(
            {
                "candle_date": self.trading_date,
                "candle_count": self.size,
                "candles": self.to_records(),
                "storage": {
                    "type": "numpy_buffer",
                    "capacity": self.capacity,
                },
            }
        )
        return base


class _InMemoryState:
    def __init__(self) -> None:
        self.payload = None
        self.trading_catalog = []
        self.candles = {}
        self.calculations = {}
        self.numpy_candles = {}


_STATE = _InMemoryState()
_LOCK = RLock()


def set_payload(payload, *, catalog=None) -> None:
    """Persist the latest filtered payload (and optional catalog) in RAM."""
    with _LOCK:
        _STATE.payload = payload
        if catalog is not None:
            _STATE.trading_catalog = catalog


def get_payload():
    """Return the current filtered payload snapshot."""
    with _LOCK:
        return _STATE.payload


def set_trading_catalog(catalog) -> None:
    with _LOCK:
        _STATE.trading_catalog = catalog


def get_trading_catalog():
    with _LOCK:
        return list(_STATE.trading_catalog)


def _ensure_numpy_buffer(trading_symbol, capacity=DEFAULT_NUMPY_CAPACITY):
    buffer = _STATE.numpy_candles.get(trading_symbol)
    if buffer is None:
        buffer = CandleBuffer(capacity=capacity)
        _STATE.numpy_candles[trading_symbol] = buffer
    return buffer


def _append_numpy_history_locked(trading_symbol, record):
    candles = record.get("candles")
    if not candles:
        return
    buffer = _ensure_numpy_buffer(trading_symbol)
    candle_date = record.get("candle_date")
    buffer.reset_for_date(candle_date)
    buffer.update_metadata(record)
    buffer.append_many(candles)


def set_candle_record(trading_symbol, record) -> None:
    with _LOCK:
        history = _STATE.candles.get(trading_symbol)
        if history is None:
            history = deque(maxlen=MAX_CANDLE_HISTORY)
            _STATE.candles[trading_symbol] = history
        history.append(record.copy())
        _append_numpy_history_locked(trading_symbol, record)
    publish_live_update(record)


def get_candle_record(trading_symbol):
    with _LOCK:
        history = _STATE.candles.get(trading_symbol)
        
        if not history:
            return None
        return history[-1]


def get_candle_history(trading_symbol):
    with _LOCK:
        buffer = _STATE.numpy_candles.get(trading_symbol)
        if buffer and buffer.size:
            record = buffer.export_record()
            if record:
                return [record]
        history = _STATE.candles.get(trading_symbol)
        if not history:
            return []
        return list(history)


def get_numpy_candle_snapshot(trading_symbol):
    with _LOCK:
        buffer = _STATE.numpy_candles.get(trading_symbol)
        if not buffer or buffer.size == 0:
            return None
        snapshot = buffer.to_numpy_dict()
        if not snapshot:
            return None
        snapshot["meta"] = dict(buffer.metadata)
        snapshot["meta"]["candle_date"] = buffer.trading_date
        snapshot["meta"]["storage"] = {
            "type": "numpy_buffer",
            "capacity": buffer.capacity,
        }
        return snapshot


def get_calculation_state(trading_symbol):
    with _LOCK:
        return dict(_STATE.calculations.get(trading_symbol, {}))


def update_calculation_state(trading_symbol, **fields):
    with _LOCK:
        state = _STATE.calculations.setdefault(trading_symbol, {})
        state.update(fields)
        return dict(state)
MAX_CANDLE_HISTORY = 5
