#!/usr/bin/env python3
"""
Pure PyQt6 + pyqtgraph viewer that runs CandleFetcher in-process (no FastAPI).

Startup flow:
1) Download + extract instrument dump.
2) Build payload in memory.
3) Start CandleFetcher in a background asyncio loop.
4) QTimer polls in-memory snapshot and updates the chart with time on X-axis.

Run: python pyqt_direct_viewer.py NIFTY
"""

import sys
import threading
import asyncio
from datetime import datetime, time
from typing import List, Tuple

import numpy as np
from PyQt6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

THEME_BG = "#0c1119"
THEME_PANEL = "#111927"
THEME_TEXT = "#e9ecf2"
THEME_ACCENT = "#5af0ff"
THEME_GRID = "#2a3344"
THEME_BANK = "#f59f00"

_PG_CONFIG = dict(antialias=True, background=THEME_BG, foreground=THEME_TEXT)
try:
    # Ask pyqtgraph to use OpenGL-backed ViewBox for smoother rendering on capable GPUs.
    pg.setConfigOptions(useOpenGL=True, **_PG_CONFIG)
except Exception:
    pg.setConfigOptions(**_PG_CONFIG)

from download_extract import clean_previous_downloads, download_and_extract
from extract import refresh_payload_in_memory
from market_fetcher import CandleFetcher
from state import (
    get_calculation_state,
    get_numpy_candle_snapshot,
    get_trading_catalog,
    IST,
)
from calculations import get_normalized_data
from logger import log_info, log_error


class TimeAxis(pg.AxisItem):
    """Format x-axis ticks as time strings in IST."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLabel(text="Time (IST)")

    def tickStrings(self, values, scale, spacing):
        labels = []
        for v in values:
            try:
                dt = datetime.fromtimestamp(v, tz=IST)
                labels.append(dt.strftime("%H:%M:%S"))
            except Exception:
                labels.append("")
        return labels


def _normalized_average(symbol: str) -> np.ndarray | None:
    """Return normalized 'average' series - computed on-demand."""
    # Use lazy normalization - only computes when needed
    norm = get_normalized_data(symbol, columns=["average_diff_cumsum"])
    avg = norm.get("average_diff_cumsum")
    if avg is None:
        return None
    arr = np.asarray(avg, dtype=float)
    return arr if arr.size else None


def _xy_from_snapshot(snapshot, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build x (epoch seconds) and y (Average) arrays from snapshot.
    Prefer numpy snapshot (time_seconds + averages); swap in normalized Y if available.
    Fall back to candle records if needed.
    """
    time_seconds = snapshot.get("time_seconds")
    averages = snapshot.get("average")
    date_str = None
    meta = snapshot.get("meta") or {}
    date_str = meta.get("candle_date") or snapshot.get("trading_date")
    if time_seconds is not None and averages is not None and date_str:
        try:
            base_date = datetime.strptime(date_str, "%Y-%m-%d")
            base = datetime.combine(base_date.date(), time.min, tzinfo=IST)
            base_ts = base.timestamp()
            xs_arr = np.asarray(time_seconds, dtype=float) + base_ts
            ys_arr = np.asarray(averages, dtype=float)
            norm_avg = _normalized_average(symbol)
            if norm_avg is not None and norm_avg.size:
                take = min(xs_arr.size, norm_avg.size)
                xs_arr = xs_arr[-take:]
                ys_arr = norm_avg[-take:]
            return xs_arr, ys_arr
        except Exception:
            pass

    # Fallback: parse candle Time strings only if numpy path not available
    candles = snapshot.get("candles") or []
    xs: List[float] = []
    ys: List[float] = []
    for candle in candles:
        t = candle.get("Time")
        avg = candle.get("Average")
        if t is None or avg is None:
            continue
        try:
            dt = datetime.fromisoformat(t)
            xs.append(dt.timestamp())
            ys.append(float(avg))
        except Exception:
            continue
    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)
    norm_avg = _normalized_average(symbol)
    if norm_avg is not None and norm_avg.size:
        take = min(xs_arr.size, norm_avg.size)
        xs_arr = xs_arr[-take:]
        ys_arr = norm_avg[-take:]
    return xs_arr, ys_arr


class FetcherThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self._loop = None

    async def _runner(self):
        fetcher = CandleFetcher()
        await fetcher.start()
        log_info("[PyQt] CandleFetcher started in-process")
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(0.5)
        finally:
            try:
                await fetcher.stop()
            except Exception as exc:
                log_error(f"[PyQt] fetcher.stop error: {exc}")
            log_info("[PyQt] CandleFetcher stopped")

    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._runner())
        finally:
            self._loop.close()

    def stop(self):
        self._stop_event.set()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: None)


class ChartWidget(QtWidgets.QWidget):
    crosshairMoved = QtCore.pyqtSignal(float)

    def __init__(self, symbol: str, title: str, line_color: str = THEME_ACCENT):
        super().__init__()
        self.symbol = symbol.upper()
        self.title = title
        self._line_color = line_color
        self._last_points = 0
        self._last_x = np.array([])
        self._last_y = np.array([])
        self._last_idx = -1
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 10)
        layout.setSpacing(4)
        self._status = QtWidgets.QLabel(f"{self.title}: waiting…")
        self._status.setStyleSheet(
            "color: {text}; font-size: 13px; font-weight: 500;".format(text=THEME_TEXT)
        )
        layout.addWidget(self._status)

        widget = pg.PlotWidget(axisItems={"bottom": TimeAxis(orientation="bottom")})
        widget.setBackground(THEME_PANEL)
        widget.showGrid(x=True, y=True, alpha=0.18)
        widget.setLabel("left", "Average", color=THEME_TEXT)
        widget.setMouseEnabled(x=False, y=False)
        widget.getPlotItem().hideButtons()
        line_pen = pg.mkPen(color=self._line_color, width=2.5)
        self._curve = widget.plot(pen=line_pen)
        cross_pen = pg.mkPen("#7f8c8d", width=1, style=QtCore.Qt.PenStyle.DashLine)
        self._crosshair_v = pg.InfiniteLine(angle=90, pen=cross_pen)
        self._crosshair_h = pg.InfiniteLine(angle=0, pen=cross_pen)
        widget.addItem(self._crosshair_v, ignoreBounds=True)
        widget.addItem(self._crosshair_h, ignoreBounds=True)
        self._marker = pg.ScatterPlotItem(
            size=11, brush=pg.mkBrush("#ffce73"), pen=pg.mkPen("#111", width=1)
        )
        widget.addItem(self._marker)
        self._label = pg.TextItem(anchor=(0, 1))
        self._label.setHtml(
            "<div style='background: rgba(12,17,25,0.85); padding:6px 8px; "
            "border-radius:6px; color: white; font-size:13px; border:1px solid #1f2735;'>—</div>"
        )
        widget.addItem(self._label)
        self._proxy = pg.SignalProxy(widget.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_move)
        axis_font = QtGui.QFont()
        axis_font.setPointSize(12)
        widget.getAxis("left").setStyle(tickFont=axis_font)
        widget.getAxis("bottom").setStyle(tickFont=axis_font)
        widget.getAxis("left").setPen(pg.mkPen(THEME_GRID))
        widget.getAxis("bottom").setPen(pg.mkPen(THEME_GRID))
        layout.addWidget(widget)
        self._plot_widget = widget
        self.setLayout(layout)
        self.setStyleSheet(
            "background-color: {bg}; border: 1px solid #1d2533; border-radius: 8px;".format(bg=THEME_PANEL)
        )

    def refresh(self):
        snapshot = get_numpy_candle_snapshot(self.symbol)
        if not snapshot:
            self._status.setText(f"{self.title}: waiting…")
            return
        x, y = _xy_from_snapshot(snapshot, self.symbol)
        if x.size == 0 or y.size == 0:
            self._status.setText(f"{self.title}: no points yet…")
            return
        self._last_x = x
        self._last_y = y
        self._curve.setData(x, y)
        self._last_points = len(x)
        self._status.setText(f"{self.title}: Live | points: {self._last_points}")
        self._curve.getViewBox().enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

    def _on_mouse_move(self, event):
        pos = event[0]  # QPointF
        plot_item = self._curve.getViewBox()
        if not plot_item.sceneBoundingRect().contains(pos):
            return
        mouse_point = plot_item.mapSceneToView(pos)
        x_val = mouse_point.x()
        self._set_crosshair_at_x(x_val, emit=True)

    def _set_crosshair_at_x(self, x_val: float, emit: bool = False):
        if not self._last_x.size:
            return

        idx = self._nearest_index(x_val)
        if idx < 0:
            return
        if idx == self._last_idx or (self._last_idx != -1 and abs(idx - self._last_idx) < 2):
            return
        self._last_idx = idx
        nearest_x = float(self._last_x[idx])
        nearest_y = float(self._last_y[idx]) if idx < self._last_y.size else 0.0
        try:
            dt = datetime.fromtimestamp(nearest_x, tz=IST)
            t_str = dt.strftime("%H:%M:%S")
        except Exception:
            t_str = f"{nearest_x:.2f}"

        self._crosshair_v.setPos(nearest_x)
        self._crosshair_h.setPos(nearest_y)
        self._marker.setData([nearest_x], [nearest_y])
        self._label.setHtml(
            f"<div style='background: rgba(0,0,0,0.65); "
            f"padding:4px; border-radius:4px; color:white; font-size:14px;'>"
            f"<b>{t_str}</b><br/>Avg: {nearest_y:.2f}"
            f"</div>"
        )
        self._label.setPos(nearest_x, nearest_y)

        if emit:
            self.crosshairMoved.emit(nearest_x)

    def sync_crosshair(self, x_val: float):
        """Move crosshair to x_val without re-emitting the signal."""
        self._set_crosshair_at_x(x_val, emit=False)

    def _nearest_index(self, x_val: float) -> int:
        """Return index of nearest x using binary search for speed."""
        arr = self._last_x
        if not arr.size:
            return -1
        idx = int(np.searchsorted(arr, x_val))
        if idx <= 0:
            return 0
        if idx >= arr.size:
            return arr.size - 1
        prev_idx = idx - 1
        return prev_idx if abs(arr[prev_idx] - x_val) <= abs(arr[idx] - x_val) else idx


def resolve_symbols():
    """Return tuple of (label, future_symbol) for NIFTY and BANKNIFTY."""
    catalog = get_trading_catalog()
    nifty_future = None
    bank_future = None
    for entry in catalog:
        if entry.get("category") == "index_future":
            if entry.get("label") == "NIFTY" and nifty_future is None:
                nifty_future = entry.get("trading_symbol")
            if entry.get("label") == "BANKNIFTY" and bank_future is None:
                bank_future = entry.get("trading_symbol")
        if nifty_future and bank_future:
            break
    return nifty_future, bank_future


class ChartWindow(QtWidgets.QMainWindow):
    def __init__(self, fetcher_thread: FetcherThread):
        super().__init__()
        self.fetcher_thread = fetcher_thread
        self.setWindowTitle("Live Candles (direct)")
        self._build_layout()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh_all)
        self._timer.start(1500)
        self._refresh_all()

    def closeEvent(self, event):
        self._timer.stop()
        self.fetcher_thread.stop()
        return super().closeEvent(event)

    def _build_layout(self):
        nifty_future, bank_future = resolve_symbols()
        charts = [
            ChartWidget("NIFTY", "NIFTY", line_color=THEME_ACCENT),
            ChartWidget(nifty_future or "NIFTY", "NIFTY FUT", line_color=THEME_ACCENT),
            ChartWidget("BANKNIFTY", "BANKNIFTY", line_color=THEME_BANK),
            ChartWidget(bank_future or "BANKNIFTY", "BANKNIFTY FUT", line_color=THEME_BANK),
        ]
        self._charts = charts
        self._wire_crosshair_sync()

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        for idx, chart in enumerate(charts):
            row = idx // 2
            col = idx % 2
            grid.addWidget(chart, row, col)

        container = QtWidgets.QWidget()
        container.setLayout(grid)
        container.setStyleSheet(f"background-color: {THEME_BG};")
        self.setCentralWidget(container)

    def _refresh_all(self):
        for chart in self._charts:
            chart.refresh()

    def _wire_crosshair_sync(self):
        for chart in self._charts:
            chart.crosshairMoved.connect(lambda x_val, sender=chart: self._broadcast_crosshair(x_val, sender))

    def _broadcast_crosshair(self, x_val: float, sender: ChartWidget):
        for chart in self._charts:
            if chart is sender:
                continue
            chart.sync_crosshair(x_val)


def bootstrap():
    log_info("[PyQt] Cleaning and downloading instruments…")
    clean_previous_downloads()
    download_and_extract()
    log_info("[PyQt] Building payload in memory…")
    refresh_payload_in_memory()


def main():
    bootstrap()
    fetcher_thread = FetcherThread()
    fetcher_thread.start()

    app = QtWidgets.QApplication(sys.argv)
    win = ChartWindow(fetcher_thread)
    win.resize(1200, 800)
    win.show()
    exit_code = app.exec()

    fetcher_thread.stop()
    fetcher_thread.join(timeout=2)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
