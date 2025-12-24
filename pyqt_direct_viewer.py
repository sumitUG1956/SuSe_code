#!/usr/bin/env python3
"""
Advanced PyQt6 + pyqtgraph Chart Viewer for Normalized Options Data

Features:
- 8 charts (4 CE + 4 PE) showing normalized cumsum columns
- Index toggle (NIFTY/BANKNIFTY/SENSEX)
- Expiry toggle (Current/Next)
- Progressive compression (old data sparse, new data dense)
- Synced crosshair across ALL charts
- GPU (OpenGL) enabled for smooth rendering
- Dark mode theme
- Zoom = auto detail increase
- Full detail toggle

Run: python pyqt_direct_viewer.py
"""

import sys
import threading
import asyncio
from datetime import datetime, time as time_cls
from typing import Dict, List, Optional, Tuple
from functools import partial

import numpy as np
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QButtonGroup, QRadioButton, QFrame,
    QSplitter, QStatusBar, QToolTip
)
import pyqtgraph as pg
from zoneinfo import ZoneInfo

# ============================================================================
# THEME CONFIGURATION
# ============================================================================
THEME = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "border": "#30363d",
    "text": "#e6edf3",
    "text_dim": "#7d8590",
    "accent": "#58a6ff",
    "green": "#3fb950",
    "red": "#f85149",
    "orange": "#d29922",
    "purple": "#a371f7",
    "cyan": "#39d353",
    "grid": "#21262d",
    "crosshair": "#f0f6fc",
}

# Chart colors for different strikes (gradient from ATM to OTM)
STRIKE_COLORS_CE = [
    "#58a6ff", "#79b8ff", "#9ecbff", "#bbdfff", "#d5edff",
    "#58a6ff", "#79b8ff", "#9ecbff", "#bbdfff", "#d5edff",
] * 10  # Repeat for more strikes

STRIKE_COLORS_PE = [
    "#f85149", "#ff7b72", "#ffa198", "#ffc1bb", "#ffdcd7",
    "#f85149", "#ff7b72", "#ffa198", "#ffc1bb", "#ffdcd7",
] * 10

# ============================================================================
# PYQTGRAPH CONFIGURATION - GPU ENABLED + PERFORMANCE
# ============================================================================
_PG_CONFIG = dict(
    antialias=False,  # Disable for performance
    background=THEME["bg"],
    foreground=THEME["text"]
)
try:
    pg.setConfigOptions(useOpenGL=True, enableExperimental=True, **_PG_CONFIG)
except Exception:
    pg.setConfigOptions(**_PG_CONFIG)

# ============================================================================
# IMPORTS FROM PROJECT
# ============================================================================
from download_extract import clean_previous_downloads, download_and_extract
from extract import refresh_payload_in_memory
from market_fetcher import CandleFetcher
from state import get_numpy_candle_snapshot, get_trading_catalog, get_calculation_state
from combined_normalization import normalize_all_index_options, INDEX_NAMES, NORMALIZE_COLUMNS
from logger import log_info, log_error

IST = ZoneInfo("Asia/Kolkata")


# ============================================================================
# TIME AXIS - Shows HH:MM instead of seconds
# ============================================================================

class TimeAxisItem(pg.AxisItem):
    """Custom axis that displays time in HH:MM format."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def tickStrings(self, values, scale, spacing):
        """Convert seconds to HH:MM format."""
        strings = []
        for v in values:
            try:
                seconds = int(v)
                h = seconds // 3600
                m = (seconds % 3600) // 60
                strings.append(f"{h:02d}:{m:02d}")
            except:
                strings.append("")
        return strings


# ============================================================================
# DATA COMPRESSION UTILITIES
# ============================================================================

def lttb_downsample(x: np.ndarray, y: np.ndarray, target_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Largest Triangle Three Buckets (LTTB) downsampling.
    Preserves visual shape while reducing points.
    """
    n = len(x)
    if n <= target_points or target_points < 3:
        return x, y
    
    # Always keep first and last
    sampled_x = [x[0]]
    sampled_y = [y[0]]
    
    bucket_size = (n - 2) / (target_points - 2)
    
    a = 0  # Previous selected point index
    
    for i in range(target_points - 2):
        # Calculate bucket range
        bucket_start = int((i + 1) * bucket_size) + 1
        bucket_end = int((i + 2) * bucket_size) + 1
        bucket_end = min(bucket_end, n - 1)
        
        # Calculate average of next bucket for reference
        next_start = int((i + 2) * bucket_size) + 1
        next_end = int((i + 3) * bucket_size) + 1
        next_end = min(next_end, n)
        
        if next_start < n:
            avg_x = np.mean(x[next_start:next_end])
            avg_y = np.nanmean(y[next_start:next_end])
        else:
            avg_x = x[-1]
            avg_y = y[-1]
        
        # Find point in current bucket with largest triangle area
        max_area = -1
        max_idx = bucket_start
        
        for j in range(bucket_start, bucket_end):
            if np.isnan(y[j]):
                continue
            # Triangle area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
            area = abs(
                (x[a] - avg_x) * (y[j] - sampled_y[-1]) -
                (x[a] - x[j]) * (avg_y - sampled_y[-1])
            )
            if area > max_area:
                max_area = area
                max_idx = j
        
        sampled_x.append(x[max_idx])
        sampled_y.append(y[max_idx])
        a = max_idx
    
    # Add last point
    sampled_x.append(x[-1])
    sampled_y.append(y[-1])
    
    return np.array(sampled_x), np.array(sampled_y)


def progressive_compress(
    time_arr: np.ndarray,
    value_arr: np.ndarray,
    current_time_seconds: int,
    full_detail: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Progressive compression: recent data = full, old data = sparse.
    More aggressive compression for better performance.
    
    Time windows:
    - 0-5 min ago: 100% (all points)
    - 5-15 min ago: 50% (every 2nd)
    - 15-30 min ago: 25% (every 4th)
    - 30-60 min ago: 10% (every 10th)
    - 1+ hours ago: 5% (every 20th)
    """
    if full_detail or len(time_arr) < 50:
        return time_arr, value_arr
    
    n = len(time_arr)
    mask = np.zeros(n, dtype=bool)
    
    for i, t in enumerate(time_arr):
        age_seconds = current_time_seconds - t
        age_minutes = age_seconds / 60
        
        if age_minutes <= 5:
            # Very recent: keep all
            mask[i] = True
        elif age_minutes <= 15:
            # 5-15 min: keep every 2nd
            mask[i] = (i % 2 == 0)
        elif age_minutes <= 30:
            # 15-30 min: keep every 4th
            mask[i] = (i % 4 == 0)
        elif age_minutes <= 60:
            # 30-60 min: keep every 10th
            mask[i] = (i % 10 == 0)
        else:
            # 1+ hours: keep every 20th
            mask[i] = (i % 20 == 0)
    
    # Always keep first and last
    if n > 0:
        mask[0] = True
        mask[-1] = True
    
    return time_arr[mask], value_arr[mask]


def get_opacity_for_age(age_minutes: float) -> float:
    """Get opacity based on data age."""
    if age_minutes <= 10:
        return 1.0
    elif age_minutes <= 30:
        return 0.8
    elif age_minutes <= 60:
        return 0.6
    elif age_minutes <= 120:
        return 0.4
    else:
        return 0.25


# ============================================================================
# SYNCED CROSSHAIR
# ============================================================================

class SyncedCrosshair:
    """Manages crosshair sync across multiple plots."""
    
    def __init__(self):
        self.plots: List[pg.PlotWidget] = []
        self.v_lines: List[pg.InfiniteLine] = []
        self.h_lines: List[pg.InfiniteLine] = []
        self.labels: List[pg.TextItem] = []
        self.enabled = True
    
    def add_plot(self, plot: pg.PlotWidget):
        """Add a plot to the sync group."""
        self.plots.append(plot)
        
        # Vertical line (synced across all)
        v_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(THEME["crosshair"], width=1, style=Qt.PenStyle.DashLine)
        )
        v_line.setVisible(False)
        plot.addItem(v_line)
        self.v_lines.append(v_line)
        
        # Horizontal line (per plot)
        h_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(THEME["crosshair"], width=1, style=Qt.PenStyle.DashLine)
        )
        h_line.setVisible(False)
        plot.addItem(h_line)
        self.h_lines.append(h_line)
        
        # Value label
        label = pg.TextItem(
            text="", color=THEME["text"],
            anchor=(0, 1), fill=pg.mkBrush(THEME["panel"])
        )
        label.setVisible(False)
        plot.addItem(label)
        self.labels.append(label)
        
        # Connect mouse move
        plot.scene().sigMouseMoved.connect(
            partial(self._on_mouse_move, plot_index=len(self.plots) - 1)
        )
        plot.scene().sigMouseClicked.connect(self._on_mouse_leave)
    
    def _on_mouse_move(self, pos, plot_index: int):
        """Handle mouse move on any plot."""
        if not self.enabled:
            return
        
        plot = self.plots[plot_index]
        vb = plot.getPlotItem().vb
        
        if plot.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()
            
            # Update all vertical lines (synced time)
            for i, v_line in enumerate(self.v_lines):
                v_line.setPos(x)
                v_line.setVisible(True)
            
            # Update horizontal line only for hovered plot
            for i, h_line in enumerate(self.h_lines):
                if i == plot_index:
                    h_line.setPos(y)
                    h_line.setVisible(True)
                else:
                    h_line.setVisible(False)
            
            # Update label for hovered plot
            time_str = self._seconds_to_time_str(int(x))
            for i, label in enumerate(self.labels):
                if i == plot_index:
                    label.setText(f"Time: {time_str}\nValue: {y:.2f}")
                    label.setPos(x, y)
                    label.setVisible(True)
                else:
                    label.setVisible(False)
    
    def _on_mouse_leave(self, event):
        """Hide crosshair on click/leave."""
        for v_line in self.v_lines:
            v_line.setVisible(False)
        for h_line in self.h_lines:
            h_line.setVisible(False)
        for label in self.labels:
            label.setVisible(False)
    
    def _seconds_to_time_str(self, seconds: int) -> str:
        """Convert seconds of day to HH:MM:SS string."""
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def set_enabled(self, enabled: bool):
        """Enable/disable crosshair."""
        self.enabled = enabled
        if not enabled:
            self._on_mouse_leave(None)


# ============================================================================
# CHART WIDGET
# ============================================================================

class NormalizedChart(pg.PlotWidget):
    """Single chart for one normalized column."""
    
    def __init__(self, title: str, column_name: str, option_type: str, parent=None):
        # Create with custom time axis
        super().__init__(
            parent=parent,
            axisItems={'bottom': TimeAxisItem(orientation='bottom')}
        )
        
        self.title_text = title
        self.column_name = column_name
        self.option_type = option_type  # "CE" or "PE"
        self.curves: Dict[str, pg.PlotDataItem] = {}
        self.full_detail = False
        
        self._setup_plot()
    
    def _setup_plot(self):
        """Configure plot appearance."""
        self.setBackground(THEME["panel"])
        self.getPlotItem().setTitle(self.title_text, color=THEME["text"], size="10pt")
        
        # Disable auto-range during updates for performance
        self.getPlotItem().setAutoVisible(y=True)
        self.getPlotItem().setClipToView(True)
        self.getPlotItem().setDownsampling(auto=True, mode='peak')
        
        # Axis styling
        for axis in ['left', 'bottom']:
            ax = self.getPlotItem().getAxis(axis)
            ax.setPen(pg.mkPen(THEME["border"]))
            ax.setTextPen(pg.mkPen(THEME["text_dim"]))
        
        # Grid
        self.showGrid(x=True, y=True, alpha=0.3)
        
        # Enable mouse interaction
        self.setMouseEnabled(x=True, y=True)
        self.enableAutoRange()
    
    def update_data(
        self,
        data: Dict[str, np.ndarray],
        time_seconds: np.ndarray,
        strikes: List[str],
        current_time: int
    ):
        """Update chart with new data."""
        colors = STRIKE_COLORS_CE if self.option_type == "CE" else STRIKE_COLORS_PE
        
        # Remove old curves
        for curve in self.curves.values():
            self.removeItem(curve)
        self.curves.clear()
        
        # Disable auto-range during batch update
        self.disableAutoRange()
        
        # Add new curves
        for i, strike_key in enumerate(strikes):
            col_key = f"{strike_key}_{self.column_name}"
            if col_key not in data:
                continue
            
            values = data[col_key]
            if values is None or len(values) == 0:
                continue
            
            # Progressive compression
            x, y = progressive_compress(
                time_seconds, values, current_time, self.full_detail
            )
            
            # Skip if no valid data
            if len(x) == 0:
                continue
            
            # Create curve with performance optimizations
            color = colors[i % len(colors)]
            pen = pg.mkPen(color, width=1)
            curve = self.plot(
                x, y, pen=pen, name=strike_key,
                connect='finite',  # Skip NaN gaps efficiently
                skipFiniteCheck=True
            )
            self.curves[strike_key] = curve
        
        # Re-enable auto-range
        self.enableAutoRange()
    
    def set_full_detail(self, full: bool):
        """Toggle full detail mode."""
        self.full_detail = full


# ============================================================================
# MAIN WINDOW
# ============================================================================

class ChartViewer(QMainWindow):
    """Main chart viewer window."""
    
    data_updated = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        self.current_index = "NIFTY"
        self.current_expiry = "current"  # "current" or "next"
        self.full_detail = False
        self.atm_only = False  # Show only ATM strike
        self.charts: List[NormalizedChart] = []
        self.crosshair = SyncedCrosshair()
        
        self._setup_ui()
        self._setup_data()
        self._setup_timers()
        
        self.data_updated.connect(self._on_data_updated)
    
    def _setup_ui(self):
        """Setup the UI layout."""
        self.setWindowTitle("📊 Options Normalized Data Viewer")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {THEME["bg"]};
            }}
            QWidget {{
                background-color: {THEME["bg"]};
                color: {THEME["text"]};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QPushButton {{
                background-color: {THEME["panel"]};
                border: 1px solid {THEME["border"]};
                border-radius: 6px;
                padding: 8px 16px;
                color: {THEME["text"]};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {THEME["border"]};
            }}
            QPushButton:checked {{
                background-color: {THEME["accent"]};
                color: {THEME["bg"]};
            }}
            QRadioButton {{
                color: {THEME["text"]};
                spacing: 8px;
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
            }}
            QLabel {{
                color: {THEME["text"]};
            }}
            QFrame#separator {{
                background-color: {THEME["border"]};
            }}
        """)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Top control bar
        control_bar = self._create_control_bar()
        layout.addWidget(control_bar)
        
        # Charts area
        charts_widget = self._create_charts_area()
        layout.addWidget(charts_widget, stretch=1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"color: {THEME['text_dim']};")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _create_control_bar(self) -> QWidget:
        """Create top control bar."""
        bar = QFrame()
        bar.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME["panel"]};
                border: 1px solid {THEME["border"]};
                border-radius: 8px;
                padding: 5px;
            }}
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Index selection
        index_label = QLabel("Index:")
        index_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(index_label)
        
        self.index_group = QButtonGroup(self)
        for idx_name in INDEX_NAMES:
            rb = QRadioButton(idx_name)
            rb.setChecked(idx_name == self.current_index)
            rb.toggled.connect(partial(self._on_index_changed, idx_name))
            self.index_group.addButton(rb)
            layout.addWidget(rb)
        
        # Separator
        sep1 = QFrame()
        sep1.setObjectName("separator")
        sep1.setFixedWidth(2)
        sep1.setFixedHeight(30)
        layout.addWidget(sep1)
        
        # Expiry toggle
        expiry_label = QLabel("Expiry:")
        expiry_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(expiry_label)
        
        self.btn_current_exp = QPushButton("Current")
        self.btn_current_exp.setCheckable(True)
        self.btn_current_exp.setChecked(True)
        self.btn_current_exp.clicked.connect(lambda: self._on_expiry_changed("current"))
        layout.addWidget(self.btn_current_exp)
        
        self.btn_next_exp = QPushButton("Next")
        self.btn_next_exp.setCheckable(True)
        self.btn_next_exp.clicked.connect(lambda: self._on_expiry_changed("next"))
        layout.addWidget(self.btn_next_exp)
        
        # Separator
        sep_atm = QFrame()
        sep_atm.setObjectName("separator")
        sep_atm.setFixedWidth(2)
        sep_atm.setFixedHeight(30)
        layout.addWidget(sep_atm)
        
        # ATM Only toggle
        self.btn_atm = QPushButton("🎯 ATM Only")
        self.btn_atm.setCheckable(True)
        self.btn_atm.setToolTip("Show only At-The-Money strike")
        self.btn_atm.clicked.connect(self._on_atm_toggle)
        layout.addWidget(self.btn_atm)
        
        layout.addStretch()
        
        # Separator
        sep2 = QFrame()
        sep2.setObjectName("separator")
        sep2.setFixedWidth(2)
        sep2.setFixedHeight(30)
        layout.addWidget(sep2)
        
        # Detail mode toggle
        self.btn_detail = QPushButton("⚡ Compressed")
        self.btn_detail.setCheckable(True)
        self.btn_detail.setToolTip("Toggle between compressed and full detail mode")
        self.btn_detail.clicked.connect(self._on_detail_toggle)
        layout.addWidget(self.btn_detail)
        
        # Refresh button
        self.btn_refresh = QPushButton("🔄 Refresh")
        self.btn_refresh.clicked.connect(self._refresh_data)
        layout.addWidget(self.btn_refresh)
        
        # Crosshair toggle
        self.btn_crosshair = QPushButton("✚ Crosshair")
        self.btn_crosshair.setCheckable(True)
        self.btn_crosshair.setChecked(True)
        self.btn_crosshair.clicked.connect(self._on_crosshair_toggle)
        layout.addWidget(self.btn_crosshair)
        
        return bar
    
    def _create_charts_area(self) -> QWidget:
        """Create the 8 chart grid (4 CE + 4 PE)."""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setSpacing(5)
        
        # Column titles
        columns = [
            ("IV Diff CumSum", "iv_diff_cumsum"),
            ("OI Diff CumSum", "oi_diff_cumsum"),
            ("TimeVal×Vol CumSum", "timevalue_vol_prod_cumsum"),
            ("TimeVal Diff CumSum", "timevalue_diff_cumsum"),
        ]
        
        # CE Charts (top row)
        ce_label = QLabel("📈 CALL OPTIONS (CE)")
        ce_label.setStyleSheet(f"color: {THEME['accent']}; font-weight: bold; font-size: 12pt;")
        main_layout.addWidget(ce_label)
        
        ce_grid = QGridLayout()
        ce_grid.setSpacing(5)
        for i, (title, col_name) in enumerate(columns):
            chart = NormalizedChart(title, col_name, "CE")
            self.charts.append(chart)
            self.crosshair.add_plot(chart)
            ce_grid.addWidget(chart, 0, i)
        
        ce_widget = QWidget()
        ce_widget.setLayout(ce_grid)
        main_layout.addWidget(ce_widget, stretch=1)
        
        # PE Charts (bottom row)
        pe_label = QLabel("📉 PUT OPTIONS (PE)")
        pe_label.setStyleSheet(f"color: {THEME['red']}; font-weight: bold; font-size: 12pt;")
        main_layout.addWidget(pe_label)
        
        pe_grid = QGridLayout()
        pe_grid.setSpacing(5)
        for i, (title, col_name) in enumerate(columns):
            chart = NormalizedChart(title, col_name, "PE")
            self.charts.append(chart)
            self.crosshair.add_plot(chart)
            pe_grid.addWidget(chart, 0, i)
        
        pe_widget = QWidget()
        pe_widget.setLayout(pe_grid)
        main_layout.addWidget(pe_widget, stretch=1)
        
        # Link X axes for all charts
        self._link_x_axes()
        
        return widget
    
    def _link_x_axes(self):
        """Link X axes of all charts for synchronized panning/zooming."""
        if len(self.charts) < 2:
            return
        
        main_plot = self.charts[0].getPlotItem()
        for chart in self.charts[1:]:
            chart.getPlotItem().setXLink(main_plot)
    
    def _setup_data(self):
        """Setup data fetching in background thread."""
        self.fetcher_thread = None
        self.loop = None
        self._start_fetcher()
    
    def _start_fetcher(self):
        """Start the candle fetcher in background thread."""
        def run_fetcher():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Download and setup
            clean_previous_downloads()
            download_and_extract()
            refresh_payload_in_memory()
            
            # Start fetcher
            fetcher = CandleFetcher()
            try:
                # Start the fetcher (creates background task)
                self.loop.run_until_complete(fetcher.start())
                # Keep the loop running
                self.loop.run_forever()
            except Exception as e:
                log_error(f"Fetcher error: {e}")
            finally:
                # Cleanup
                self.loop.run_until_complete(fetcher.stop())
                self.loop.close()
        
        self.fetcher_thread = threading.Thread(target=run_fetcher, daemon=True)
        self.fetcher_thread.start()
    
    def _setup_timers(self):
        """Setup update timers."""
        # Data refresh timer (every 5 seconds)
        self.data_timer = QTimer(self)
        self.data_timer.timeout.connect(self._refresh_data)
        self.data_timer.start(5000)
        
        # Initial load after 3 seconds
        QTimer.singleShot(3000, self._refresh_data)
    
    def _on_index_changed(self, index_name: str, checked: bool):
        """Handle index selection change."""
        if checked:
            self.current_index = index_name
            self._refresh_data()
    
    def _on_expiry_changed(self, expiry: str):
        """Handle expiry toggle."""
        self.current_expiry = expiry
        self.btn_current_exp.setChecked(expiry == "current")
        self.btn_next_exp.setChecked(expiry == "next")
        self._refresh_data()
    
    def _on_detail_toggle(self):
        """Toggle full detail mode."""
        self.full_detail = self.btn_detail.isChecked()
        self.btn_detail.setText("📊 Full Detail" if self.full_detail else "⚡ Compressed")
        
        for chart in self.charts:
            chart.set_full_detail(self.full_detail)
        
        self._refresh_data()
    
    def _on_crosshair_toggle(self):
        """Toggle crosshair."""
        self.crosshair.set_enabled(self.btn_crosshair.isChecked())
    
    def _on_atm_toggle(self):
        """Toggle ATM only mode."""
        self.atm_only = self.btn_atm.isChecked()
        self._refresh_data()
    
    def _get_current_spot_price(self) -> Optional[float]:
        """Get current spot price for the selected index."""
        spot_snapshot = get_numpy_candle_snapshot(self.current_index)
        if not spot_snapshot:
            return None
        
        avg = spot_snapshot.get("average")
        if avg is None:
            return None
        
        size = spot_snapshot.get("size", len(avg))
        if size == 0:
            return None
        
        # Return last average (current price)
        return float(avg[size - 1])
    
    def _find_atm_strike(self, strikes: List[str], spot_price: float) -> List[str]:
        """Find the ATM strike closest to spot price."""
        if not strikes or spot_price is None:
            return strikes
        
        def extract_strike_price(s: str) -> int:
            try:
                # "DEC24_24000CE" -> 24000
                for part in s.split("_"):
                    if part.endswith("CE") or part.endswith("PE"):
                        return int(part[:-2])
            except:
                pass
            return 0
        
        # Find closest strike to spot
        closest = min(strikes, key=lambda s: abs(extract_strike_price(s) - spot_price))
        return [closest]

    def _refresh_data(self):
        """Refresh chart data from normalized results."""
        try:
            # Get normalized data
            all_norm = normalize_all_index_options()
            norm_data = all_norm.get(self.current_index, {})
            
            if not norm_data:
                self.status_bar.showMessage(f"No data for {self.current_index}")
                return
            
            # Get time array from spot
            spot_snapshot = get_numpy_candle_snapshot(self.current_index)
            if not spot_snapshot:
                self.status_bar.showMessage(f"No spot data for {self.current_index}")
                return
            
            time_seconds = spot_snapshot.get("time_seconds")
            if time_seconds is None:
                return
            
            size = spot_snapshot.get("size", len(time_seconds))
            time_seconds = time_seconds[:size]
            
            # Current time
            now = datetime.now(IST)
            current_time_seconds = now.hour * 3600 + now.minute * 60 + now.second
            
            # Parse strikes from column names
            ce_strikes, pe_strikes = self._get_strikes_from_data(norm_data)
            
            # Filter by expiry
            ce_strikes = self._filter_by_expiry(ce_strikes)
            pe_strikes = self._filter_by_expiry(pe_strikes)
            
            # Filter by ATM if enabled
            spot_price = self._get_current_spot_price()
            if self.atm_only and spot_price:
                ce_strikes = self._find_atm_strike(ce_strikes, spot_price)
                pe_strikes = self._find_atm_strike(pe_strikes, spot_price)
            
            # Update CE charts (first 4)
            for i, chart in enumerate(self.charts[:4]):
                chart.update_data(norm_data, time_seconds, ce_strikes, current_time_seconds)
            
            # Update PE charts (last 4)
            for i, chart in enumerate(self.charts[4:]):
                chart.update_data(norm_data, time_seconds, pe_strikes, current_time_seconds)
            
            # Status
            atm_text = f" | ATM: {int(spot_price)}" if self.atm_only and spot_price else ""
            total_strikes = len(ce_strikes) + len(pe_strikes)
            points_per_strike = len(time_seconds)
            self.status_bar.showMessage(
                f"{self.current_index}{atm_text} | {total_strikes} strikes | "
                f"{points_per_strike} points | "
                f"{'Full Detail' if self.full_detail else 'Compressed'}"
            )
            
        except Exception as e:
            log_error(f"Refresh error: {e}")
            self.status_bar.showMessage(f"Error: {e}")
    
    def _get_strikes_from_data(self, data: Dict) -> Tuple[List[str], List[str]]:
        """Extract CE and PE strike keys from data."""
        ce_strikes = set()
        pe_strikes = set()
        
        for key in data.keys():
            if key.startswith("SPOT_"):
                continue
            
            # Format: "DEC24_24000CE_iv_diff_cumsum"
            parts = key.split("_")
            if len(parts) >= 2:
                strike_key = "_".join(parts[:2])  # "DEC24_24000CE"
                if "CE" in strike_key:
                    ce_strikes.add(strike_key)
                elif "PE" in strike_key:
                    pe_strikes.add(strike_key)
        
        # Sort by strike price
        def extract_strike(s):
            try:
                # "DEC24_24000CE" -> 24000
                for part in s.split("_"):
                    if part.endswith("CE") or part.endswith("PE"):
                        return int(part[:-2])
            except:
                pass
            return 0
        
        ce_list = sorted(list(ce_strikes), key=extract_strike)
        pe_list = sorted(list(pe_strikes), key=extract_strike, reverse=True)
        
        return ce_list, pe_list
    
    def _filter_by_expiry(self, strikes: List[str]) -> List[str]:
        """Filter strikes by current/next expiry."""
        if not strikes:
            return []
        
        # Group by expiry
        expiries = {}
        for s in strikes:
            exp = s.split("_")[0]  # "DEC24"
            if exp not in expiries:
                expiries[exp] = []
            expiries[exp].append(s)
        
        # Sort expiries
        sorted_expiries = sorted(expiries.keys())
        
        if self.current_expiry == "current" and len(sorted_expiries) >= 1:
            return expiries[sorted_expiries[0]]
        elif self.current_expiry == "next" and len(sorted_expiries) >= 2:
            return expiries[sorted_expiries[1]]
        elif sorted_expiries:
            return expiries[sorted_expiries[0]]
        
        return strikes
    
    def _on_data_updated(self):
        """Slot for data update signal."""
        self._refresh_data()
    
    def closeEvent(self, event):
        """Clean up on close."""
        self.data_timer.stop()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        event.accept()


# ============================================================================
# MAIN
# ============================================================================

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Dark palette
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(THEME["bg"]))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(THEME["text"]))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(THEME["panel"]))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(THEME["text"]))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(THEME["panel"]))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(THEME["text"]))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(THEME["accent"]))
    app.setPalette(palette)
    
    viewer = ChartViewer()
    viewer.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
