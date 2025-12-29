#!/usr/bin/env python3

import asyncio
import json
from datetime import datetime, time, timedelta, timezone
from urllib.parse import urlencode

import aiohttp
from zoneinfo import ZoneInfo
from candle_processing import transform_candles_for_today
from calculations import run_calculation
from combined_normalization import normalize_all_index_options
from logger import log_error, log_info
from extract import (
    build_filtered_payload,
    collect_trading_symbol_entries,
    load_instruments,
    store_payload_in_memory,
)
from state import (
    get_payload,
    get_trading_catalog,
    get_calculation_state,
    set_candle_record,
)

IST = ZoneInfo("Asia/Kolkata")
API_URL = "https://service.upstox.com/chart/open/v3/candles"
DEFAULT_INTERVAL = "S10"
MAX_LIMIT = 25_00
MIN_FETCH_LIMIT = 10
LIMIT_BUFFER_SECONDS = 5
THREE_MINUTES = 580

# Trading day cadence windows (IST) - Market hours 9:15 to 15:30
# TODO: Remove 23:59 window after testing - only for debugging
WINDOWS = [
    (time(9, 15), time(9, 45), 30),
    (time(9, 45), time(13, 45), 30),
    (time(13, 45), time(15, 30), 30),
    (time(15, 30), time(23, 59), 3000),  # DEBUG: Remove after testing
]


def ist_now():
    return datetime.now(tz=IST)


def closing_timestamp_ms(now=None):
    """Return epoch milliseconds for today's 15:30:00 IST."""
    now = now or ist_now()
    target = datetime.combine(now.date(), time(15, 30), tzinfo=IST)
    return int(target.astimezone(timezone.utc).timestamp() * 1000)


def cutoff_timestamp_ms(now=None):
    """
    Return the snapshot cutoff timestamp (current time capped at today's 15:30 IST).
    """
    now = now or ist_now()
    closing = datetime.combine(now.date(), time(15, 30), tzinfo=IST)
    cutoff = closing if now > closing else now
    return int(cutoff.astimezone(timezone.utc).timestamp() * 1000)


def trading_start_timestamp_ms(reference_ts_ms=None):
    """
    Return epoch milliseconds for 09:15 IST on the trading day of ``reference_ts_ms``.
    """
    if reference_ts_ms is None:
        reference = ist_now()
    else:
        reference = datetime.fromtimestamp(
            reference_ts_ms / 1000, tz=timezone.utc
        ).astimezone(IST)
    start = datetime.combine(reference.date(), time(9, 15), tzinfo=IST)
    return int(start.astimezone(timezone.utc).timestamp() * 1000)


def seconds_until(target_time, now):
    """Compute seconds until the next occurrence of target_time (IST)."""
    candidate = datetime.combine(now.date(), target_time, tzinfo=IST)
    if candidate <= now:
        candidate += timedelta(days=1)
    return (candidate - now).total_seconds()


def next_window_delay(now):
    """Seconds until the next active window begins."""
    today_time = now.time()
    for start, _, _ in WINDOWS:
        if today_time < start:
            return seconds_until(start, now)
    # Past last window → wait for tomorrow's first window
    return seconds_until(WINDOWS[0][0], now)


def current_interval_seconds(now):
    """Return the fetch spacing for the current time, if inside a window."""
    current_time = now.time()
    for start, end, spacing in WINDOWS:
        if start <= current_time < end:
            return spacing
    return None


class InstrumentSpec:
    def __init__(self, trading_symbol, instrument_key, category, metadata=None) -> None:
        self.trading_symbol = trading_symbol
        self.instrument_key = instrument_key
        self.category = category
        self.metadata = metadata or {}


class CandleFetcher:
    """Fetch minute-level candle snapshots for configured instruments."""

    def __init__(
        self,
        *,
        concurrency=8,
        request_timeout=10,
        interval=DEFAULT_INTERVAL,
        limit=MAX_LIMIT,
    ):
        self.concurrency = concurrency
        self.request_timeout = request_timeout
        self.interval = interval
        self.limit = limit
        self._session = None
        self._stop = asyncio.Event()
        self._task = None
        self._instruments = []
        self._next_fetch = {}
        self._spot_ready = {}

    async def start(self) -> None:
        if self._task:
            return
        self._stop = asyncio.Event()
        await self.reload_catalog()
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._task = asyncio.create_task(self._run(), name="candle-fetcher")

    async def stop(self) -> None:
        log_info("[CandleFetcher] Stop requested")
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        if self._session:
            await self._session.close()
        self._task = None
        self._session = None
        log_info("[CandleFetcher] Stopped")

    async def reload_catalog(self) -> None:
        specs = self._load_catalog_sync()
        if specs:
            self._instruments = specs

    def _load_catalog_sync(self):
        catalog = get_trading_catalog()
        if not catalog:
            payload = get_payload()
            if payload is None:
                payload = build_filtered_payload(load_instruments())
            catalog = store_payload_in_memory(payload)

        specs = []
        for entry in catalog:
            trading_symbol = entry.get("trading_symbol")
            instrument_key = entry.get("instrument_key")
            category = entry.get("category", "unknown")
            if not trading_symbol or not instrument_key:
                continue
            meta = {
                k: v
                for k, v in entry.items()
                if k not in {"trading_symbol", "instrument_key", "category"}
            }
            
            specs.append(
                InstrumentSpec(
                    trading_symbol=trading_symbol,
                    instrument_key=instrument_key,
                    category=category,
                    metadata=meta,
                )
            )
        return specs

    async def _run(self) -> None:
        log_info("[CandleFetcher] Background loop started")
        if self._instruments:
            log_info(
                "[CandleFetcher] Loaded instruments: "
                + ", ".join(
                    f"{spec.trading_symbol} ({spec.category})" for spec in self._instruments
                )
            )
        while not self._stop.is_set():
            now = ist_now()
            interval = current_interval_seconds(now)
            if interval is None:
                sleep_for = next_window_delay(now)
                log_info(
                    f"[CandleFetcher] Outside trading window; sleeping {sleep_for:.0f}s "
                    "until next window"
                )
                await self._sleep_with_stop(sleep_for)
                continue

            performed = await self._fetch_once(now, interval)
            if not performed:
                log_info(
                    f"[CandleFetcher] Cycle idle; sleeping {interval:.0f}s before next fetch"
                )
                await self._sleep_with_stop(interval)
        log_info("[CandleFetcher] Background loop stopped")

    async def _sleep_with_stop(self, seconds):
        if seconds <= 0:
            return
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _fetch_once(self, now, base_interval):
        if not self._instruments:
            await asyncio.sleep(5)
            return False

        session = self._session
        if not session:
            raise RuntimeError("CandleFetcher session not initialized")

        from_ts = cutoff_timestamp_ms(now)
        due_specs = [
            spec
            for spec in self._instruments
            if self._should_fetch_spec(spec, now, base_interval)
        ]
        if not due_specs:
            log_info("[CandleFetcher] No specs due this cycle")
        if not due_specs:
            return False

        trading_date = now.astimezone(IST).date().isoformat()
        ready_specs = []
        for spec in due_specs:
            if self._requires_spot_reference(spec) and not self._is_spot_ready(spec, trading_date):
                continue
            ready_specs.append(spec)

        if not ready_specs:
            return False

        ready_specs.sort(key=lambda spec: 0 if self._is_spot_spec(spec) else 1)

        for spec in ready_specs:
            self._mark_scheduled(spec, now, base_interval)

        semaphore = asyncio.Semaphore(self.concurrency)
        tasks = [
            asyncio.create_task(
                self._fetch_and_store(spec, from_ts, semaphore, session),
                name=f"candle-fetch-{spec.trading_symbol}",
            )
            for spec in ready_specs
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if stop was requested during fetch
        if self._stop.is_set():
            return True
        
        # Auto normalize all index options after fetch cycle completes (simple sequential)
        try:
            normalize_all_index_options()
            log_info("[CandleFetcher] Combined normalization completed")
            
            # Pre-compute EMA columns so first user request is fast
            from combined_normalization import get_normalized_index_data
            for index_name in ("NIFTY", "BANKNIFTY", "SENSEX"):
                get_normalized_index_data(index_name)
            log_info("[CandleFetcher] EMA columns pre-computed")
            
            # Broadcast to WebSocket clients (only if not stopping)
            if not self._stop.is_set():
                try:
                    from fast_api import broadcast_normalized_update
                    for index_name in ("NIFTY", "BANKNIFTY", "SENSEX"):
                        await broadcast_normalized_update(index_name)
                except Exception as ws_err:
                    log_error(f"[CandleFetcher] WebSocket broadcast failed: {ws_err}")
                
        except Exception as e:
            log_error(f"[CandleFetcher] Combined normalization failed: {e}")
        
        return True

    def _desired_spacing(self, spec, base_interval):
        if self._is_nifty_spec(spec):
            return base_interval
        return THREE_MINUTES

    def _is_nifty_spec(self, spec):
        label = str(spec.metadata.get("label", "")).upper()
        symbol = spec.trading_symbol.upper()
        if label == "NIFTY":
            return True
        return symbol.startswith("NIFTY")

    def _is_spot_spec(self, spec):
        category = str(spec.category).lower()
        return category in {"index_spot", "equity_spot"}

    def _is_future_spec(self, spec):
        category = str(spec.category).lower()
        return category in {"index_future", "equity_future"}

    def _requires_spot_reference(self, spec):
        category = str(spec.category).lower()
        # Futures and options depend on spot data being ready first.
        return category in {
            "index_future",
            "equity_future",
            "index_option",
            "equity_option",
        }

    def _spec_label(self, spec):
        label = spec.metadata.get("label")
        if label:
            return str(label).upper()
        return spec.trading_symbol.upper()


    def _is_spot_ready(self, spec, trading_date):
        label = self._spec_label(spec)
        ready_date = self._spot_ready.get(label)
        return ready_date == trading_date


    def _should_fetch_spec(
        self,
        spec,
        now,
        base_interval,
    ):
        spacing = max(1, self._desired_spacing(spec, base_interval))
        current_ts = now.timestamp()
        due_at = self._next_fetch.get(spec.trading_symbol)
        if due_at and current_ts < due_at:
            return False
        return True

    def _mark_scheduled(self, spec, now, base_interval):
        spacing = max(1, self._desired_spacing(spec, base_interval))
        current_ts = now.timestamp()
        self._next_fetch[spec.trading_symbol] = current_ts + spacing

    def _limit_for_spec(self, spec, target_ts_ms):
        """Determine how many candles to request for this instrument."""
        calc_state = get_calculation_state(spec.trading_symbol)
        last_ms = calc_state.get("last_timestamp_ms")
        if not last_ms:
            last_ms = trading_start_timestamp_ms(target_ts_ms)

        if last_ms >= target_ts_ms:
            return MIN_FETCH_LIMIT

        diff_seconds = max(1, (target_ts_ms - last_ms) // 1000)

        computed = diff_seconds + LIMIT_BUFFER_SECONDS
        allowed_max = min(self.limit, MAX_LIMIT)
        base_min = min(MIN_FETCH_LIMIT, allowed_max)
        return max(base_min, min(allowed_max, computed))

    async def _fetch_and_store(
        self,
        spec,
        from_ts,
        semaphore,
        session,
    ):
        async with semaphore:
            limit = self._limit_for_spec(spec, from_ts)
            params = {
                "instrumentKey": spec.instrument_key,
                "interval": self.interval,
                "from": str(from_ts),
                "limit": str(limit),
            }
            url_with_params = f"{API_URL}?{urlencode(params)}"
            # print(f"[CandleFetcher] Fetching {spec.trading_symbol} candles: {url_with_params}")
            try:
                async with session.get(API_URL, params=params) as response:
                    text = await response.text()
                    # print(text)
                    # import time
                    # time.sleep(10)
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {text[:200]}")
                    payload = json.loads(text)
            except Exception as exc:
                await self._record_error(spec, str(exc))
                return

            raw_candles = payload.get("data", {}).get("candles", [])
            instrument_meta = {"category": spec.category, **spec.metadata}
            processed_candles, candle_date = transform_candles_for_today(
                raw_candles,
                instrument_meta=instrument_meta,
            )
            # print(f"[CandleFetcher] Processed {len(processed_candles)} candles for {spec.trading_symbol} on {candle_date}")

            if self._is_spot_spec(spec) and processed_candles:
                label = self._spec_label(spec)
                self._spot_ready[label] = candle_date
            
            record = {
                "instrument_key": spec.instrument_key,
                "trading_symbol": spec.trading_symbol,
                "category": spec.category,
                "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
                "from_timestamp_ms": from_ts,
                "candle_date": candle_date,
                "candles": processed_candles,
                "candle_count": len(processed_candles),
                "meta": payload.get("data", {}).get("meta", {}),
                "instrument_meta": instrument_meta,
            }
            
            set_candle_record(spec.trading_symbol, record)
            run_calculation(spec.trading_symbol, record=record)

    async def _record_error(self, spec, message):
        log_error(f"[CandleFetcher] {spec.trading_symbol}: {message}")
        record = {
            "instrument_key": spec.instrument_key,
            "trading_symbol": spec.trading_symbol,
            "category": spec.category,
            "status": "error",
            "message": message,
            "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        set_candle_record(spec.trading_symbol, record)


__all__ = [
    "CandleFetcher",
    "closing_timestamp_ms",
    "current_interval_seconds",
    "next_window_delay",
]
