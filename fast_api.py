#!/usr/bin/env python3

import asyncio
import signal
import sys
import json
import re
from datetime import datetime, time as time_cls
from pathlib import Path
from typing import Set, Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect, Request, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from pydantic import BaseModel
import uvicorn
from starlette.concurrency import run_in_threadpool

from download_extract import clean_previous_downloads, download_and_extract
from extract import refresh_payload_in_memory, run_report_script
from market_fetcher import CandleFetcher
from state import get_payload, get_numpy_candle_snapshot
from combined_normalization import normalize_index_options, get_normalized_index_data, get_metadata_only, INDEX_NAMES
from futures_normalization import normalize_all_futures, get_futures_normalized_data, get_futures_metadata, ALL_FUTURES
import pandas as pd
import numpy as np
from logger import log_error, log_info
from live_updates import subscribe, unsubscribe

# WebSocket connections for normalized data
_normalized_ws_clients: Set[WebSocket] = set()
_normalized_ws_subscriptions: dict = {}  # ws -> index_name

# WebSocket connections for futures data
_futures_ws_clients: Set[WebSocket] = set()
_futures_ws_subscriptions: dict = {}  # ws -> {smooth: bool}

candle_fetcher = None
_fetcher_lock = asyncio.Lock()


def refresh_payload_and_report():
    """Refresh the in-memory payload and immediately emit the contract report."""
    refresh_payload_in_memory()
    run_report_script()


async def ensure_candle_fetcher(*, reload=False):
    """Start (or refresh) the candle fetcher background service."""
    global candle_fetcher
    async with _fetcher_lock:
        if candle_fetcher is None:
            candle_fetcher = CandleFetcher()
            await candle_fetcher.start()
        elif reload:
            await candle_fetcher.reload_catalog()


app = FastAPI(title="Upstox Instruments API (FastAPI + Uvicorn)")

# Mount static files
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# ---- Chrome DevTools Handler (suppress 404 noise) ---------------------------

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    """Handle Chrome DevTools request to suppress 404 logs."""
    return Response(content="{}", media_type="application/json")


# ---- Dashboard Routes -------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to dashboard."""
    return RedirectResponse(url="/dashboard", status_code=302)


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the options dashboard HTML page."""
    html_path = BASE_DIR / "templates" / "dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(html_path)


@app.get("/futures-dashboard", response_class=HTMLResponse)
async def serve_futures_dashboard():
    """Serve the futures dashboard HTML page."""
    html_path = BASE_DIR / "templates" / "futures.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Futures dashboard not found")
    return FileResponse(html_path)


# ---- Lifecycle --------------------------------------------------------------

IST = ZoneInfo("Asia/Kolkata")
MARKET_READY_TIME = time_cls(9, 16, 0)  # 09:16:00 IST


async def wait_until_market_ready():
    """
    Wait until 09:16 IST before fetching spot prices.
    
    - Weekend (Sat/Sun): Raise error with faketime instructions
    - Before 09:16: Wait with countdown
    - After 09:16 (until midnight): Proceed immediately
    """
    now = datetime.now(tz=IST)
    today = now.date()
    weekday = now.weekday()  # 0=Monday, 5=Saturday, 6=Sunday
    
    # Check for weekend
    if weekday in (5, 6):  # Saturday or Sunday
        day_name = "Saturday" if weekday == 5 else "Sunday"
        print(f"\n{'='*60}")
        print(f"⚠️  TODAY IS {day_name.upper()} - MARKET IS CLOSED")
        print(f"{'='*60}")
        print(f"\n💡 For debugging on weekends, use faketime:")
        print(f"")
        print(f"   # Install faketime (if not installed):")
        print(f"   sudo pacman -S libfaketime  # Arch")
        print(f"   sudo apt install faketime   # Ubuntu/Debian")
        print(f"")
        print(f"   # Run server with fake date (last trading day):")
        print(f"   faketime '2025-12-27 09:16:00' python fast_api.py")
        print(f"")
        print(f"{'='*60}\n")
        raise RuntimeError(f"Market is closed on {day_name}. Use faketime for debugging.")
    
    # Calculate target time: 09:16:00 IST today
    target_time = datetime.combine(today, MARKET_READY_TIME, tzinfo=IST)
    
    if now >= target_time:
        # Already past 09:16 - proceed immediately
        log_info(f"[startup] Market ready (current time: {now.strftime('%H:%M:%S')} IST)")
        return
    
    # Before 09:16 - wait with countdown
    wait_seconds = (target_time - now).total_seconds()
    wait_minutes = wait_seconds / 60
    
    print(f"\n{'='*60}")
    print(f"⏳ WAITING FOR MARKET TO OPEN")
    print(f"{'='*60}")
    print(f"   Current time: {now.strftime('%H:%M:%S')} IST")
    print(f"   Target time:  {MARKET_READY_TIME.strftime('%H:%M:%S')} IST")
    print(f"   Wait time:    {wait_minutes:.1f} minutes")
    print(f"{'='*60}\n")
    
    # Wait with countdown in terminal
    while True:
        now = datetime.now(tz=IST)
        if now >= target_time:
            break
        
        remaining = (target_time - now).total_seconds()
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        
        # Print countdown on same line (carriage return)
        print(f"\r⏳ Countdown: {mins:02d}:{secs:02d} remaining until 09:16 IST...  ", end="", flush=True)
        await asyncio.sleep(1)
    
    print(f"\r✅ 09:16 IST reached! Starting...                              ")
    print(f"{'='*60}\n")


@app.on_event("startup")
async def on_startup():
    # Step 1: Clean and download instruments (can happen anytime)
    await run_in_threadpool(clean_previous_downloads)
    await run_in_threadpool(download_and_extract)
    log_info("[startup] Instruments downloaded")
    
    # Step 2: Wait until 09:16 IST (market ready)
    await wait_until_market_ready()
    
    # Step 3: Fetch spot prices and build payload
    await run_in_threadpool(refresh_payload_and_report)
    
    # Step 4: Start candle fetcher
    await ensure_candle_fetcher()
    log_info("[startup] Payload ready and candle fetcher running")


@app.on_event("shutdown")
async def on_shutdown():
    global candle_fetcher
    log_info("[shutdown] Starting graceful shutdown...")
    
    # Stop candle fetcher first
    if candle_fetcher is not None:
        try:
            await asyncio.wait_for(candle_fetcher.stop(), timeout=5.0)
        except asyncio.TimeoutError:
            log_info("[shutdown] CandleFetcher stop timed out")
        except Exception as e:
            log_error(f"[shutdown] CandleFetcher stop error: {e}")
        candle_fetcher = None
    
    # Cancel all pending tasks except current
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        log_info(f"[shutdown] Cancelling {len(tasks)} pending tasks...")
        for task in tasks:
            task.cancel()
        # Wait for cancellation with timeout
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # Close all WebSocket connections
    for ws in list(_normalized_ws_clients):
        try:
            await ws.close()
        except Exception:
            pass
    _normalized_ws_clients.clear()
    _normalized_ws_subscriptions.clear()
    
    log_info("[shutdown] FastAPI app shutdown complete")


# ---- Routes -----------------------------------------------------------------


from state import get_candle_record, get_candle_history, IST

@app.get("/candles/{symbol}")
async def candles(symbol: str):
    snapshot = get_numpy_candle_snapshot(symbol.upper())
    if snapshot:
        date_str = snapshot["meta"].get("candle_date") or snapshot.get("trading_date")
        if not date_str:
            date_str = pd.Timestamp.now(tz=IST).strftime("%Y-%m-%d")
        base = pd.Timestamp(f"{date_str} 00:00:00", tz=IST)
        
        # Vectorized time formatting (faster than loop)
        time_seconds = snapshot["time_seconds"]
        n = len(time_seconds)
        
        # Pre-compute all times at once
        times = base + pd.to_timedelta(time_seconds, unit="s")
        formatted_times = times.strftime("%Y-%m-%dT%H:%M:%S%z").tolist()
        
        # Helper to safely convert numpy arrays to lists with None for NaN
        def safe_list(arr):
            if arr is None:
                return [None] * n
            arr = np.asarray(arr)
            result = arr.tolist()
            # Replace nan with None in one pass
            return [None if (isinstance(v, float) and np.isnan(v)) else v for v in result]
        
        # Build candles list directly (avoid pandas DataFrame overhead)
        candles_list = []
        
        # Pre-extract all arrays once
        average = safe_list(snapshot.get("average"))
        volume = safe_list(snapshot.get("volume"))
        open_interest = safe_list(snapshot.get("open_interest"))
        spot_average = safe_list(snapshot.get("spot"))
        iv = safe_list(snapshot.get("iv"))
        delta = safe_list(snapshot.get("delta"))
        vega = safe_list(snapshot.get("vega"))
        theta = safe_list(snapshot.get("theta"))
        average_diff = safe_list(snapshot.get("average_diff"))
        average_diff_cumsum = safe_list(snapshot.get("average_diff_cumsum"))
        average_vol_prod = safe_list(snapshot.get("average_vol_prod"))
        average_vol_prod_cumsum = safe_list(snapshot.get("average_vol_prod_cumsum"))
        future_spot_diff = safe_list(snapshot.get("future_spot_diff"))
        future_spot_diff_cumsum = safe_list(snapshot.get("future_spot_diff_cumsum"))
        iv_diff = safe_list(snapshot.get("iv_diff"))
        iv_diff_cumsum = safe_list(snapshot.get("iv_diff_cumsum"))
        delta_diff = safe_list(snapshot.get("delta_diff"))
        delta_diff_cumsum = safe_list(snapshot.get("delta_diff_cumsum"))
        vega_diff = safe_list(snapshot.get("vega_diff"))
        vega_diff_cumsum = safe_list(snapshot.get("vega_diff_cumsum"))
        theta_diff = safe_list(snapshot.get("theta_diff"))
        theta_diff_cumsum = safe_list(snapshot.get("theta_diff_cumsum"))
        timevalue_diff = safe_list(snapshot.get("timevalue_diff"))
        timevalue_diff_cumsum = safe_list(snapshot.get("timevalue_diff_cumsum"))
        timevalue_vol_prod = safe_list(snapshot.get("timevalue_vol_prod"))
        timevalue_vol_prod_cumsum = safe_list(snapshot.get("timevalue_vol_prod_cumsum"))
        oi_diff = safe_list(snapshot.get("oi_diff"))
        oi_diff_cumsum = safe_list(snapshot.get("oi_diff_cumsum"))
        
        for i in range(n):
            candles_list.append({
                "Time": formatted_times[i],
                "Average": average[i],
                "Volume": volume[i],
                "OpenInterest": open_interest[i],
                "SpotAverage": spot_average[i],
                "IV": iv[i],
                "Delta": delta[i],
                "Vega": vega[i],
                "Theta": theta[i],
                "AverageDiff": average_diff[i],
                "AverageDiffCumSum": average_diff_cumsum[i],
                "AverageVolProd": average_vol_prod[i],
                "AverageVolProdCumSum": average_vol_prod_cumsum[i],
                "FutureSpotDiff": future_spot_diff[i],
                "FutureSpotDiffCumSum": future_spot_diff_cumsum[i],
                "IVDiff": iv_diff[i],
                "IVDiffCumSum": iv_diff_cumsum[i],
                "DeltaDiff": delta_diff[i],
                "DeltaDiffCumSum": delta_diff_cumsum[i],
                "VegaDiff": vega_diff[i],
                "VegaDiffCumSum": vega_diff_cumsum[i],
                "ThetaDiff": theta_diff[i],
                "ThetaDiffCumSum": theta_diff_cumsum[i],
                "TimeValueDiff": timevalue_diff[i],
                "TimeValueDiffCumSum": timevalue_diff_cumsum[i],
                "TimeValueVolProd": timevalue_vol_prod[i],
                "TimeValueVolProdCumSum": timevalue_vol_prod_cumsum[i],
                "OpenInterestDiff": oi_diff[i],
                "OpenInterestDiffCumSum": oi_diff_cumsum[i],
            })
        
        return {
            "meta": snapshot["meta"],
            "candles": candles_list,
        }

    history = get_candle_history(symbol.upper())
    if not history:
        raise HTTPException(status_code=404, detail="No data in RAM")
    return history


@app.websocket("/ws/candles/{symbol}")
async def candles_ws(websocket: WebSocket, symbol: str):
    await websocket.accept()
    symbol = symbol.upper()
    queue = await subscribe(symbol)
    try:
        # Send the latest known record immediately if present.
        latest = get_candle_record(symbol)
        if latest:
            await websocket.send_json(latest)

        while True:
            record = await queue.get()
            if record:
                await websocket.send_json(record)
    except WebSocketDisconnect:
        pass
    finally:
        await unsubscribe(symbol)


@app.get("/health")
async def health():
    from state import get_candle_record
    for sym in ("NIFTY","BANKNIFTY"):
        rec = get_candle_record(sym)
        # print(sym, rec and rec.get("candle_date"))
    return {"status": "ok"}


@app.get("/download")
async def trigger_download():
    """
    Re-run the complete workflow and return paths plus a readiness flag.
    Everything stays inside the running process memory.
    """
    await run_in_threadpool(clean_previous_downloads)

    # Local vars only (not stored globally)
    result = await run_in_threadpool(download_and_extract)
    await run_in_threadpool(refresh_payload_and_report)
    await ensure_candle_fetcher(reload=True)

    download_path, extracted_path = result
    return {
        "downloaded_to": str(download_path.resolve()),
        "extracted_to": str(extracted_path.resolve()),
        "payload_ready": True,
    }


@app.get("/payload")
async def get_payload_route():
    """
    Return the current filtered payload directly from in-memory state.
    """
    payload = get_payload()
    if not payload:
        log_error("[/payload] No payload built yet")
        raise HTTPException(status_code=404, detail="No payload built yet")

    raw_bytes = json.dumps(payload).encode("utf-8")
    return Response(content=raw_bytes, media_type="application/json")


# ---- Normalized Data API ----------------------------------------------------

def _filter_by_expiry(normalized: dict, expiry: str = None) -> tuple:
    """
    Filter normalized data by expiry.
    Returns: (filtered_data, available_expiries)
    """
    if not normalized:
        return {}, []
    
    # Extract all available expiries from column names
    available_expiries = set()
    for col_name in normalized.keys():
        # Column format: DEC24_24000CE_iv_diff_cumsum
        if "_" in col_name:
            exp = col_name.split("_")[0]
            if exp and len(exp) >= 4 and exp[:3].isalpha():
                available_expiries.add(exp)
    
    available_expiries = sorted(available_expiries)
    
    # If no expiry specified or invalid, return all
    if not expiry or expiry not in available_expiries:
        return normalized, available_expiries
    
    # Filter only columns matching the expiry
    filtered = {}
    for col_name, values in normalized.items():
        if col_name.startswith(f"{expiry}_"):
            filtered[col_name] = values
    
    return filtered, available_expiries


@app.get("/api/normalized/{index_name}")
async def get_normalized_data(index_name: str, expiry: str = None, strikes: str = None, smooth: bool = True):
    """
    Get normalized options data for an index.
    
    Args:
        index_name: NIFTY, BANKNIFTY, or SENSEX
        expiry: Optional expiry filter (e.g., "DEC24", "JAN25"). If not provided, returns all expiries.
        strikes: Optional comma-separated strike prices (e.g., "24000,24100,24200"). 
                 If not provided, returns ONLY metadata (no normalized data) for fast initial load.
        smooth: If True, returns EMA smoothed data (_ema columns). If False, returns raw data. Default True.
    
    Returns: {normalized: {col_name: [values...]}, time_seconds: [...], spot_price: float, available_expiries: [...], current_expiry: str, available_strikes: [...]}
    """
    index_name = index_name.upper()
    if index_name not in INDEX_NAMES:
        raise HTTPException(status_code=400, detail=f"Invalid index. Use: {INDEX_NAMES}")
    
    # FAST PATH: If no strikes requested, return only metadata (no heavy computation)
    if not strikes:
        # Get metadata from cached calculation state (already computed in background)
        metadata = await run_in_threadpool(get_metadata_only, index_name)
        
        if not metadata:
            raise HTTPException(status_code=404, detail=f"No data for {index_name}")
        
        return metadata
    
    # SLOW PATH: Strikes requested - need full normalization
    normalized = await run_in_threadpool(get_normalized_index_data, index_name)
    
    if not normalized:
        raise HTTPException(status_code=404, detail=f"No data for {index_name}")
    
    # Filter by expiry if specified
    expiry_upper = expiry.upper() if expiry else None
    filtered_data, available_expiries = _filter_by_expiry(normalized, expiry_upper)
    
    # Filter by smooth mode - keep only EMA or only raw columns
    if smooth:
        # Keep only _ema columns (smooth mode)
        filtered_data = {k: v for k, v in filtered_data.items() if k.endswith('_ema')}
    else:
        # Keep only non-_ema columns (raw mode)
        filtered_data = {k: v for k, v in filtered_data.items() if not k.endswith('_ema')}
    
    # Parse all available strikes from filtered data
    all_strikes = set()
    for col_name in filtered_data.keys():
        # Match standard columns: EXPIRY_STRIKE_CE/PE_metric
        match = re.match(r'^([A-Z]{3}\d{2})_(\d+)(CE|PE)_', col_name)
        if match:
            all_strikes.add(int(match.group(2)))
        else:
            # Also match skew columns: EXPIRY_STRIKE_vega_skew
            match_skew = re.match(r'^([A-Z]{3}\d{2})_(\d+)_vega_skew', col_name)
            if match_skew:
                all_strikes.add(int(match_skew.group(2)))
    available_strikes = sorted(all_strikes)
    
    # Filter by strikes (lazy loading - only requested strikes)
    requested_strikes = set(int(s.strip()) for s in strikes.split(',') if s.strip().isdigit())
    filtered_by_strikes = {}
    for col_name, values in filtered_data.items():
        # Match standard CE/PE columns
        match = re.match(r'^([A-Z]{3}\d{2})_(\d+)(CE|PE)_', col_name)
        if match:
            strike = int(match.group(2))
            if strike in requested_strikes:
                filtered_by_strikes[col_name] = values
        else:
            # Match skew columns
            match_skew = re.match(r'^([A-Z]{3}\d{2})_(\d+)_vega_skew', col_name)
            if match_skew:
                strike = int(match_skew.group(2))
                if strike in requested_strikes:
                    filtered_by_strikes[col_name] = values
            else:
                # Keep non-strike columns (like SPOT columns)
                filtered_by_strikes[col_name] = values
    filtered_data = filtered_by_strikes
    
    # Get time_seconds from spot
    spot_snapshot = get_numpy_candle_snapshot(index_name)
    time_seconds = []
    spot_price = None
    
    if spot_snapshot:
        ts = spot_snapshot.get("time_seconds")
        size = spot_snapshot.get("size", len(ts) if ts is not None else 0)
        if ts is not None:
            time_seconds = ts[:size].tolist()
        
        # Get latest spot price
        avg = spot_snapshot.get("average")
        if avg is not None and size > 0:
            spot_price = float(avg[size - 1]) if not np.isnan(avg[size - 1]) else None
    
    # Convert numpy arrays to lists with rounding
    normalized_json = {}
    for col_name, values in filtered_data.items():
        if isinstance(values, np.ndarray):
            # Replace nan with None and round to 2 decimal places
            clean = [None if np.isnan(v) else round(float(v), 2) for v in values]
            normalized_json[col_name] = clean
        else:
            normalized_json[col_name] = values
    
    return {
        "index": index_name,
        "normalized": normalized_json,
        "time_seconds": time_seconds,
        "spot_price": spot_price,
        "available_expiries": available_expiries,
        "available_strikes": available_strikes,
        "smooth": smooth,
        "current_expiry": expiry_upper if expiry_upper in available_expiries else (available_expiries[0] if available_expiries else None),
    }


# ---- WebSocket for Normalized Data ------------------------------------------

@app.websocket("/ws/normalized")
async def normalized_ws(websocket: WebSocket):
    """
    WebSocket endpoint for real-time normalized data updates.
    
    Client sends: {"action": "subscribe", "index": "NIFTY", "expiry": "DEC24", "strikes": [24000, 24100], "smooth": true}
    Server sends: {"type": "update", "index": "NIFTY", "expiry": "DEC24", "normalized": {...}, ...}
    """
    await websocket.accept()
    _normalized_ws_clients.add(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("action") == "subscribe":
                index_name = data.get("index", "NIFTY").upper()
                expiry = data.get("expiry", "").upper() if data.get("expiry") else None
                strikes = data.get("strikes", [])  # List of strike prices to subscribe to
                smooth = data.get("smooth", True)  # True = EMA, False = Raw
                
                if index_name in INDEX_NAMES:
                    _normalized_ws_subscriptions[websocket] = {
                        "index": index_name, 
                        "expiry": expiry,
                        "strikes": set(strikes) if strikes else None,  # None means all strikes
                        "smooth": smooth  # Store smooth preference
                    }
                    log_info(f"[WS] Client subscribed to {index_name} expiry={expiry} strikes={strikes or 'all'} smooth={smooth}")
                    
                    # Send current data immediately
                    normalized = await run_in_threadpool(get_normalized_index_data, index_name)
                    if normalized:
                        # Filter by expiry
                        filtered_data, available_expiries = _filter_by_expiry(normalized, expiry)
                        
                        # Filter by smooth mode - keep only EMA or only raw columns
                        if smooth:
                            filtered_data = {k: v for k, v in filtered_data.items() if k.endswith('_ema')}
                        else:
                            filtered_data = {k: v for k, v in filtered_data.items() if not k.endswith('_ema')}
                        
                        # Parse available strikes
                        all_strikes = set()
                        for col_name in filtered_data.keys():
                            match = re.match(r'^([A-Z]{3}\d{2})_(\d+)(CE|PE)_', col_name)
                            if match:
                                all_strikes.add(int(match.group(2)))
                        available_strikes = sorted(all_strikes)
                        
                        # Filter by strikes if specified
                        if strikes:
                            requested_strikes = set(strikes)
                            filtered_by_strikes = {}
                            for col_name, values in filtered_data.items():
                                match = re.match(r'^([A-Z]{3}\d{2})_(\d+)(CE|PE)_', col_name)
                                if match:
                                    strike = int(match.group(2))
                                    if strike in requested_strikes:
                                        filtered_by_strikes[col_name] = values
                                else:
                                    filtered_by_strikes[col_name] = values
                            filtered_data = filtered_by_strikes
                        
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
                                spot_price = float(avg[size - 1]) if not np.isnan(avg[size - 1]) else None
                        
                        normalized_json = {}
                        for col_name, values in filtered_data.items():
                            if isinstance(values, np.ndarray):
                                clean = [None if np.isnan(v) else float(v) for v in values]
                                normalized_json[col_name] = clean
                            else:
                                normalized_json[col_name] = values
                        
                        await websocket.send_json({
                            "type": "update",
                            "index": index_name,
                            "expiry": expiry if expiry in available_expiries else (available_expiries[0] if available_expiries else None),
                            "available_expiries": available_expiries,
                            "available_strikes": available_strikes,
                            "normalized": normalized_json,
                            "time_seconds": time_seconds,
                            "spot_price": spot_price,
                            "smooth": smooth,
                        })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log_error(f"[WS] Error: {e}")
    finally:
        _normalized_ws_clients.discard(websocket)
        _normalized_ws_subscriptions.pop(websocket, None)


async def broadcast_normalized_update(index_name: str):
    """
    Broadcast normalized data update to all subscribed WebSocket clients.
    Call this after normalization completes.
    """
    if not _normalized_ws_clients:
        return
    
    # Get normalized data
    normalized = await run_in_threadpool(get_normalized_index_data, index_name)
    if not normalized:
        return
    
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
            spot_price = float(avg[size - 1]) if not np.isnan(avg[size - 1]) else None
    
    # Send to subscribed clients (with their specific expiry and strikes filter)
    disconnected = []
    for ws in _normalized_ws_clients:
        sub = _normalized_ws_subscriptions.get(ws)
        if not sub:
            continue
        
        # Check if subscription matches this index
        sub_index = sub.get("index") if isinstance(sub, dict) else sub
        if sub_index != index_name:
            continue
        
        # Get expiry filter for this client
        sub_expiry = sub.get("expiry") if isinstance(sub, dict) else None
        sub_strikes = sub.get("strikes") if isinstance(sub, dict) else None
        sub_smooth = sub.get("smooth", True) if isinstance(sub, dict) else True
        
        # Filter data by expiry
        filtered_data, available_expiries = _filter_by_expiry(normalized, sub_expiry)
        
        # Parse all available strikes (including skew columns)
        all_strikes = set()
        for col_name in filtered_data.keys():
            match = re.match(r'^([A-Z]{3}\d{2})_(\d+)(CE|PE)_', col_name)
            if match:
                all_strikes.add(int(match.group(2)))
            else:
                match_skew = re.match(r'^([A-Z]{3}\d{2})_(\d+)_vega_skew', col_name)
                if match_skew:
                    all_strikes.add(int(match_skew.group(2)))
        available_strikes = sorted(all_strikes)
        
        # Filter by strikes if client has subscribed to specific strikes
        if sub_strikes:
            filtered_by_strikes = {}
            for col_name, values in filtered_data.items():
                # Match standard CE/PE columns
                match = re.match(r'^([A-Z]{3}\d{2})_(\d+)(CE|PE)_', col_name)
                if match:
                    strike = int(match.group(2))
                    if strike in sub_strikes:
                        filtered_by_strikes[col_name] = values
                else:
                    # Match skew columns
                    match_skew = re.match(r'^([A-Z]{3}\d{2})_(\d+)_vega_skew', col_name)
                    if match_skew:
                        strike = int(match_skew.group(2))
                        if strike in sub_strikes:
                            filtered_by_strikes[col_name] = values
                    else:
                        # Keep non-strike columns
                        filtered_by_strikes[col_name] = values
            filtered_data = filtered_by_strikes
        
        # Filter by smooth preference - only send EMA columns or raw columns
        if sub_smooth:
            # Send only EMA columns
            filtered_data = {k: v for k, v in filtered_data.items() if k.endswith('_ema')}
        else:
            # Send only raw columns (non-EMA)
            filtered_data = {k: v for k, v in filtered_data.items() if not k.endswith('_ema')}
        
        normalized_json = {}
        for col_name, values in filtered_data.items():
            if isinstance(values, np.ndarray):
                clean = [None if np.isnan(v) else float(v) for v in values]
                normalized_json[col_name] = clean
            else:
                normalized_json[col_name] = values
        
        message = {
            "type": "update",
            "index": index_name,
            "expiry": sub_expiry if sub_expiry in available_expiries else (available_expiries[0] if available_expiries else None),
            "available_expiries": available_expiries,
            "available_strikes": available_strikes,
            "normalized": normalized_json,
            "time_seconds": time_seconds,
            "spot_price": spot_price,
        }
        
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    
    # Cleanup disconnected
    for ws in disconnected:
        _normalized_ws_clients.discard(ws)
        _normalized_ws_subscriptions.pop(ws, None)


# ---- Futures API Endpoints --------------------------------------------------

@app.get("/api/futures/metadata")
async def get_futures_metadata_api():
    """
    Get metadata for futures dashboard (fast initial load).
    Returns available futures list and time_seconds.
    """
    metadata = await run_in_threadpool(get_futures_metadata)
    return metadata


@app.get("/api/futures/normalized")
async def get_futures_normalized_api(smooth: bool = True):
    """
    Get normalized futures data.
    
    Args:
        smooth: If True, returns EMA smoothed data (_ema columns). If False, returns raw data.
    
    Returns: {
        normalized: {
            "NIFTY": {fut_spot_diff_cumsum: [...], oi_diff_cumsum: [...]},
            "RELIANCE": {...},
            ...
        },
        time_seconds: [...],
        index_futures: [...],
        equity_futures: [...]
    }
    """
    # Get normalized data
    normalized = await run_in_threadpool(get_futures_normalized_data)
    
    if not normalized:
        # Try to compute
        normalized = await run_in_threadpool(normalize_all_futures)
    
    if not normalized:
        raise HTTPException(status_code=404, detail="No futures data available")
    
    # Get metadata
    metadata = await run_in_threadpool(get_futures_metadata)
    
    # Filter by smooth mode
    filtered_normalized = {}
    for symbol, data in normalized.items():
        if smooth:
            # Keep only _ema columns
            filtered_normalized[symbol] = {k: v for k, v in data.items() if k.endswith('_ema')}
        else:
            # Keep only non-_ema columns
            filtered_normalized[symbol] = {k: v for k, v in data.items() if not k.endswith('_ema')}
    
    # Convert numpy arrays to lists
    normalized_json = {}
    for symbol, data in filtered_normalized.items():
        normalized_json[symbol] = {}
        for col_name, values in data.items():
            if isinstance(values, np.ndarray):
                clean = [None if np.isnan(v) else round(float(v), 2) for v in values]
                normalized_json[symbol][col_name] = clean
            else:
                normalized_json[symbol][col_name] = values
    
    return {
        "normalized": normalized_json,
        "time_seconds": metadata.get("time_seconds", []),
        "index_futures": metadata.get("index_futures", []),
        "equity_futures": metadata.get("equity_futures", []),
        "smooth": smooth,
    }


@app.websocket("/ws/futures")
async def futures_ws(websocket: WebSocket):
    """
    WebSocket endpoint for real-time futures data updates.
    
    Client sends: {"action": "subscribe", "smooth": true}
    Server sends: {"type": "update", "normalized": {...}, ...}
    """
    await websocket.accept()
    _futures_ws_clients.add(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("action") == "subscribe":
                smooth = data.get("smooth", True)
                _futures_ws_subscriptions[websocket] = {"smooth": smooth}
                log_info(f"[FuturesWS] Client subscribed smooth={smooth}")
                
                # Send current data immediately
                normalized = await run_in_threadpool(get_futures_normalized_data)
                metadata = await run_in_threadpool(get_futures_metadata)
                
                if normalized:
                    # Filter by smooth mode
                    filtered_normalized = {}
                    for symbol, symbol_data in normalized.items():
                        if smooth:
                            filtered_normalized[symbol] = {k: v for k, v in symbol_data.items() if k.endswith('_ema')}
                        else:
                            filtered_normalized[symbol] = {k: v for k, v in symbol_data.items() if not k.endswith('_ema')}
                    
                    # Convert to JSON
                    normalized_json = {}
                    for symbol, symbol_data in filtered_normalized.items():
                        normalized_json[symbol] = {}
                        for col_name, values in symbol_data.items():
                            if isinstance(values, np.ndarray):
                                clean = [None if np.isnan(v) else round(float(v), 2) for v in values]
                                normalized_json[symbol][col_name] = clean
                            else:
                                normalized_json[symbol][col_name] = values
                    
                    await websocket.send_json({
                        "type": "update",
                        "normalized": normalized_json,
                        "time_seconds": metadata.get("time_seconds", []),
                        "index_futures": metadata.get("index_futures", []),
                        "equity_futures": metadata.get("equity_futures", []),
                    })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log_error(f"[FuturesWS] Error: {e}")
    finally:
        _futures_ws_clients.discard(websocket)
        _futures_ws_subscriptions.pop(websocket, None)


async def broadcast_futures_update():
    """
    Broadcast futures normalized data update to all subscribed WebSocket clients.
    Call this after futures normalization completes.
    """
    if not _futures_ws_clients:
        return
    
    normalized = await run_in_threadpool(get_futures_normalized_data)
    if not normalized:
        return
    
    metadata = await run_in_threadpool(get_futures_metadata)
    
    disconnected = []
    for ws in _futures_ws_clients:
        sub = _futures_ws_subscriptions.get(ws)
        if not sub:
            continue
        
        smooth = sub.get("smooth", True)
        
        # Filter by smooth mode
        filtered_normalized = {}
        for symbol, symbol_data in normalized.items():
            if smooth:
                filtered_normalized[symbol] = {k: v for k, v in symbol_data.items() if k.endswith('_ema')}
            else:
                filtered_normalized[symbol] = {k: v for k, v in symbol_data.items() if not k.endswith('_ema')}
        
        # Convert to JSON
        normalized_json = {}
        for symbol, symbol_data in filtered_normalized.items():
            normalized_json[symbol] = {}
            for col_name, values in symbol_data.items():
                if isinstance(values, np.ndarray):
                    clean = [None if np.isnan(v) else round(float(v), 2) for v in values]
                    normalized_json[symbol][col_name] = clean
                else:
                    normalized_json[symbol][col_name] = values
        
        message = {
            "type": "update",
            "normalized": normalized_json,
            "time_seconds": metadata.get("time_seconds", []),
            "index_futures": metadata.get("index_futures", []),
            "equity_futures": metadata.get("equity_futures", []),
        }
        
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    
    # Cleanup disconnected
    for ws in disconnected:
        _futures_ws_clients.discard(ws)
        _futures_ws_subscriptions.pop(ws, None)


if __name__ == "__main__":
    # Disable reload for cleaner shutdown (no child processes)
    uvicorn.run("fast_api:app", host="127.0.0.1", port=8000, reload=False)
