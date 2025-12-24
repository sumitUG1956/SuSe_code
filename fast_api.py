#!/usr/bin/env python3

import asyncio
import json
from pathlib import Path
from typing import Set

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
from starlette.concurrency import run_in_threadpool

from download_extract import clean_previous_downloads, download_and_extract
from extract import refresh_payload_in_memory, run_report_script
from market_fetcher import CandleFetcher
from state import get_payload, get_numpy_candle_snapshot
from combined_normalization import normalize_index_options, get_normalized_index_data, INDEX_NAMES
import pandas as pd
import numpy as np
from logger import log_error, log_info
from live_updates import subscribe, unsubscribe

# WebSocket connections for normalized data
_normalized_ws_clients: Set[WebSocket] = set()
_normalized_ws_subscriptions: dict = {}  # ws -> index_name

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


# ---- Dashboard Route --------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the dashboard HTML page."""
    html_path = BASE_DIR / "templates" / "dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(html_path)


# ---- Lifecycle --------------------------------------------------------------


@app.on_event("startup")
async def on_startup():
    # Clean → download+extract → build+store payload in RAM
    await run_in_threadpool(clean_previous_downloads)
    await run_in_threadpool(download_and_extract)
    await run_in_threadpool(refresh_payload_and_report)
    await ensure_candle_fetcher()
    log_info("[startup] Payload ready and candle fetcher running")


@app.on_event("shutdown")
async def on_shutdown():
    global candle_fetcher
    if candle_fetcher is not None:
        await candle_fetcher.stop()
        candle_fetcher = None
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

@app.get("/api/normalized/{index_name}")
async def get_normalized_data(index_name: str):
    """
    Get normalized options data for an index.
    Returns: {normalized: {col_name: [values...]}, time_seconds: [...], spot_price: float}
    """
    index_name = index_name.upper()
    if index_name not in INDEX_NAMES:
        raise HTTPException(status_code=400, detail=f"Invalid index. Use: {INDEX_NAMES}")
    
    # Run normalization in thread pool
    normalized = await run_in_threadpool(get_normalized_index_data, index_name)
    
    if not normalized:
        raise HTTPException(status_code=404, detail=f"No data for {index_name}")
    
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
    for col_name, values in normalized.items():
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
    }


# ---- WebSocket for Normalized Data ------------------------------------------

@app.websocket("/ws/normalized")
async def normalized_ws(websocket: WebSocket):
    """
    WebSocket endpoint for real-time normalized data updates.
    
    Client sends: {"action": "subscribe", "index": "NIFTY"}
    Server sends: {"type": "update", "index": "NIFTY", "normalized": {...}, ...}
    """
    await websocket.accept()
    _normalized_ws_clients.add(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("action") == "subscribe":
                index_name = data.get("index", "NIFTY").upper()
                if index_name in INDEX_NAMES:
                    _normalized_ws_subscriptions[websocket] = index_name
                    log_info(f"[WS] Client subscribed to {index_name}")
                    
                    # Send current data immediately
                    normalized = await run_in_threadpool(get_normalized_index_data, index_name)
                    if normalized:
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
                        for col_name, values in normalized.items():
                            if isinstance(values, np.ndarray):
                                clean = [None if np.isnan(v) else float(v) for v in values]
                                normalized_json[col_name] = clean
                            else:
                                normalized_json[col_name] = values
                        
                        await websocket.send_json({
                            "type": "update",
                            "index": index_name,
                            "normalized": normalized_json,
                            "time_seconds": time_seconds,
                            "spot_price": spot_price,
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
    
    normalized_json = {}
    for col_name, values in normalized.items():
        if isinstance(values, np.ndarray):
            clean = [None if np.isnan(v) else float(v) for v in values]
            normalized_json[col_name] = clean
        else:
            normalized_json[col_name] = values
    
    message = {
        "type": "update",
        "index": index_name,
        "normalized": normalized_json,
        "time_seconds": time_seconds,
        "spot_price": spot_price,
    }
    
    # Send to subscribed clients
    disconnected = []
    for ws in _normalized_ws_clients:
        if _normalized_ws_subscriptions.get(ws) == index_name:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
    
    # Cleanup disconnected
    for ws in disconnected:
        _normalized_ws_clients.discard(ws)
        _normalized_ws_subscriptions.pop(ws, None)


if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="127.0.0.1", port=8000, reload=True)
