/**
 * Algo Normalized Futures Dashboard
 * 
 * Features:
 * - Compact charts (4 per row) for all futures
 * - 2 lines per chart: Basis (FutSpotDiff) + OI
 * - WebSocket for real-time updates
 * - Synchronized crosshair across all charts
 */

// ============== GLOBAL STATE ==============
const state = {
    // Futures list
    indexFutures: [],
    equityFutures: [],
    
    // Normalized data
    normalizedData: {},
    timeSeconds: [],
    
    // Charts map: {symbol: chart}
    charts: {},
    
    // WebSocket
    ws: null,
    wsConnected: false,
    
    // Display mode removed - always use raw normalized data
};

// Line colors
const COLORS = {
    basis: '#58a6ff',  // Blue for FutSpotDiff (Basis)
    oi: '#f0883e',     // Orange for OI
};

// ============== INITIALIZATION ==============

document.addEventListener('DOMContentLoaded', async () => {
    // Load metadata first to get futures list
    await loadMetadata();
    
    // Create charts for all futures
    createCharts();
    
    // Load normalized data
    await loadData();
    
    // Connect WebSocket
    connectWebSocket();
    
    // Handle resize
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            Object.values(state.charts).forEach(chart => {
                if (chart) {
                    const container = chart.container.parentNode;
                    if (container) {
                        chart.setSize(container.offsetWidth, null, false);
                    }
                }
            });
        }, 150);
    });
});

async function loadMetadata() {
    try {
        const response = await fetch('/api/futures/metadata');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        state.indexFutures = data.index_futures || [];
        state.equityFutures = data.equity_futures || [];
        state.timeSeconds = data.time_seconds || [];
        
    } catch (error) {
        console.error('Failed to load metadata:', error);
        // Use defaults
        state.indexFutures = ['NIFTY', 'BANKNIFTY'];
        state.equityFutures = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 
                               'INFY', 'BHARTIARTL', 'ITC', 'M&M', 'SBIN', 'LT'];
    }
}

function createCharts() {
    // Set Highcharts global options
    Highcharts.setOptions({
        accessibility: { enabled: false },
        time: { useUTC: true },
        chart: {
            backgroundColor: '#161b22',
            style: { fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' },
            animation: false,
            reflow: true,
            spacingTop: 5,
            spacingBottom: 5,
            spacingLeft: 5,
            spacingRight: 5
        },
        title: { text: null },
        credits: { enabled: false },
        legend: { enabled: false },
        xAxis: {
            type: 'datetime',
            gridLineColor: '#30363d',
            lineColor: '#30363d',
            tickColor: '#30363d',
            labels: {
                style: { color: '#c9d1d9', fontSize: '13px', fontWeight: '500' },
                formatter: function() {
                    return Highcharts.dateFormat('%H:%M', this.value);
                }
            },
            crosshair: {
                width: 1,
                color: '#8b949e',
                dashStyle: 'Dash'
            }
        },
        yAxis: {
            gridLineColor: '#30363d',
            labels: {
                style: { color: '#c9d1d9', fontSize: '13px', fontWeight: '500' },
                formatter: function() { return this.value.toFixed(1); }
            },
            crosshair: {
                width: 1,
                color: '#8b949e',
                dashStyle: 'Dash'
            }
        },
        tooltip: {
            animation: false,
            backgroundColor: '#21262d',
            borderColor: '#30363d',
            style: { color: '#f0f6fc', fontSize: '11px' },
            shared: true,
            xDateFormat: '%H:%M:%S',
            valueDecimals: 2,
        },
        plotOptions: {
            series: {
                animation: false,
                marker: { enabled: false },
                lineWidth: 1.5,
                states: {
                    hover: { lineWidth: 2 },
                    inactive: { opacity: 0.3 }
                },
                turboThreshold: 5000,
                enableMouseTracking: true,
            }
        }
    });
    
    // Create all futures charts in single grid
    const futuresGrid = document.getElementById('futuresGrid');
    
    // First add index futures
    state.indexFutures.forEach(symbol => {
        const container = createChartContainer(symbol, 'index');
        futuresGrid.appendChild(container);
        createChart(symbol);
    });
    
    // Then add stock futures
    state.equityFutures.forEach(symbol => {
        const container = createChartContainer(symbol, 'stock');
        futuresGrid.appendChild(container);
        createChart(symbol);
    });
    
    // Setup crosshair sync
    setupCrosshairSync();
}

function createChartContainer(symbol, type) {
    const container = document.createElement('div');
    container.className = 'chart-container';
    container.innerHTML = `
        <div class="chart-watermark">${symbol}</div>
        <div class="chart-header">
            <span class="chart-title ${type}">${symbol} FUT</span>
            <span class="chart-value" id="val-${symbol}">--</span>
        </div>
        <div class="chart-box" id="chart-${symbol}"></div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color basis"></div>
                <span>Basis</span>
            </div>
            <div class="legend-item">
                <div class="legend-color oi"></div>
                <span>OI</span>
            </div>
        </div>
    `;
    return container;
}

function createChart(symbol) {
    const containerId = `chart-${symbol}`;
    const container = document.getElementById(containerId);
    if (!container) return;
    
    state.charts[symbol] = Highcharts.chart(container, {
        chart: {
            type: 'line',
            height: 450,
            events: {
                load: function() {
                    this.pointer.reset = function() { return undefined; };
                }
            }
        },
        xAxis: {
            events: {
                afterSetExtremes: syncExtremes
            }
        },
        yAxis: {
            plotLines: [
                { value: 0, color: '#8b949e', width: 1.5, zIndex: 3 },
                { value: 3, color: '#8b949e', width: 1, dashStyle: 'Dash', zIndex: 3 },
                { value: -3, color: '#8b949e', width: 1, dashStyle: 'Dash', zIndex: 3 }
            ],
            plotBands: [
                { from: 3, to: Infinity, color: 'rgba(63, 185, 80, 0.1)', zIndex: 0 },
                { from: -Infinity, to: -3, color: 'rgba(248, 81, 73, 0.1)', zIndex: 0 }
            ]
        },
        series: [
            { name: 'Basis', data: [], color: COLORS.basis, lineWidth: 2 },
            { name: 'OI', data: [], color: COLORS.oi, lineWidth: 1.5 }
        ]
    });
}

function setupCrosshairSync() {
    const allContainers = document.querySelectorAll('.chart-container');
    
    allContainers.forEach(container => {
        ['mousemove', 'touchmove', 'touchstart'].forEach(eventType => {
            container.addEventListener(eventType, function(e) {
                syncCrosshairs(e, this);
            });
        });
        
        container.addEventListener('mouseleave', function() {
            Object.values(state.charts).forEach(chart => {
                if (chart && chart.xAxis && chart.xAxis[0]) {
                    chart.xAxis[0].hideCrosshair();
                    chart.tooltip?.hide();
                }
            });
        });
    });
}

function syncCrosshairs(e, container) {
    const sourceChart = Object.values(state.charts).find(c => 
        c && c.container && (c.container === container || c.container.parentElement === container || container.contains(c.container))
    );
    if (!sourceChart) return;
    
    const event = sourceChart.pointer.normalize(e);
    const plotX = event.chartX - sourceChart.plotLeft;
    
    if (plotX < 0 || plotX > sourceChart.plotWidth) return;
    
    let closestTimestamp = null;
    let minDist = Infinity;
    
    sourceChart.series.forEach(series => {
        if (!series.points) return;
        series.points.forEach(point => {
            if (point.plotX !== undefined) {
                const dist = Math.abs(point.plotX - plotX);
                if (dist < minDist) {
                    minDist = dist;
                    closestTimestamp = point.x;
                }
            }
        });
    });
    
    if (closestTimestamp === null) return;
    
    updateCrosshairTime(closestTimestamp);
    
    Object.values(state.charts).forEach(chart => {
        if (!chart || !chart.xAxis || !chart.xAxis[0]) return;
        const xAxis = chart.xAxis[0];
        
        let pointAtTime = null;
        for (const series of chart.series) {
            if (series.points) {
                pointAtTime = series.points.find(p => p.x === closestTimestamp);
                if (pointAtTime) break;
            }
        }
        
        if (pointAtTime) {
            xAxis.drawCrosshair({
                chartX: pointAtTime.plotX + chart.plotLeft,
                chartY: pointAtTime.plotY + chart.plotTop
            }, pointAtTime);
        } else {
            const xPixel = xAxis.toPixels(closestTimestamp);
            if (xPixel >= chart.plotLeft && xPixel <= chart.plotLeft + chart.plotWidth) {
                xAxis.drawCrosshair({ chartX: xPixel });
            }
        }
    });
    
    updateValueDisplays(closestTimestamp);
}

function syncExtremes(e) {
    if (e.trigger === 'syncExtremes') return;
    
    const thisChart = this.chart;
    Object.values(state.charts).forEach(chart => {
        if (chart && chart !== thisChart && chart.xAxis && chart.xAxis[0]) {
            chart.xAxis[0].setExtremes(e.min, e.max, true, false, { trigger: 'syncExtremes' });
        }
    });
}

function updateCrosshairTime(timestamp) {
    const timeEl = document.getElementById('crosshairTime');
    if (timeEl && timestamp) {
        const date = new Date(timestamp);
        const hours = String(date.getUTCHours()).padStart(2, '0');
        const mins = String(date.getUTCMinutes()).padStart(2, '0');
        timeEl.textContent = `${hours}:${mins}`;
    }
}

function updateValueDisplays(timestamp) {
    const allFutures = [...state.indexFutures, ...state.equityFutures];
    
    allFutures.forEach(symbol => {
        const chart = state.charts[symbol];
        if (!chart) return;
        
        const valueEl = document.getElementById(`val-${symbol}`);
        if (!valueEl) return;
        
        const values = [];
        chart.series.forEach(series => {
            const point = series.points?.find(p => p.x === timestamp);
            if (point && point.y != null) {
                values.push(`${series.name}: ${point.y.toFixed(2)}`);
            }
        });
        
        valueEl.textContent = values.length > 0 ? values.join(' | ') : '--';
    });
}

// ============== DATA LOADING ==============

async function loadData(showOverlay = true, retryCount = 0) {
    if (showOverlay) showLoading(true);
    showErrorMessage(null);  // Clear any previous error
    
    try {
        const response = await fetch('/api/futures/normalized?smooth=false');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        
        if (!data.normalized || Object.keys(data.normalized).length === 0) {
            if (retryCount < 5) {
                showErrorMessage(`⏳ Server warming up... (${retryCount + 1}/5)`);
                setTimeout(() => loadData(showOverlay, retryCount + 1), 2000);
                return;
            }
            showErrorMessage('⚠️ No futures data available');
        }
        
        state.normalizedData = data.normalized || {};
        state.timeSeconds = data.time_seconds || [];
        
        updateCharts();
        showErrorMessage(null);  // Clear error on success
        
    } catch (error) {
        console.error('Failed to load data:', error);
        if (retryCount < 3) {
            showErrorMessage(`🔄 Connection error, retrying... (${retryCount + 1}/3)`);
            setTimeout(() => loadData(showOverlay, retryCount + 1), 3000);
            return;
        }
        showErrorMessage('❌ Failed to load data. Check server connection.');
    } finally {
        if (showOverlay) showLoading(false);
    }
}

// ============== CHART UPDATES ==============

const BASE_DATE = Date.UTC(2025, 0, 1, 0, 0, 0);

function updateCharts() {
    if (state.timeSeconds.length === 0) return;
    
    const timestamps = state.timeSeconds.map(t => BASE_DATE + (t * 1000));
    
    const allFutures = [...state.indexFutures, ...state.equityFutures];
    
    allFutures.forEach(symbol => {
        const chart = state.charts[symbol];
        if (!chart) return;
        
        const symbolData = state.normalizedData[symbol];
        if (!symbolData) return;
        
        // Basis (fut_spot_diff_cumsum) - raw normalized
        const basisValues = symbolData['fut_spot_diff_cumsum'];
        
        // OI (oi_diff_cumsum) - raw normalized
        const oiValues = symbolData['oi_diff_cumsum'];
        
        // Build data arrays
        const basisData = [];
        const oiData = [];
        
        const len = timestamps.length;
        
        if (basisValues) {
            const vLen = Math.min(len, basisValues.length);
            for (let i = 0; i < vLen; i++) {
                const v = basisValues[i];
                if (v != null && v === v) {
                    basisData.push([timestamps[i], v]);
                }
            }
        }
        
        if (oiValues) {
            const vLen = Math.min(len, oiValues.length);
            for (let i = 0; i < vLen; i++) {
                const v = oiValues[i];
                if (v != null && v === v) {
                    oiData.push([timestamps[i], v]);
                }
            }
        }
        
        // Update series
        if (chart.series[0]) {
            chart.series[0].setData(basisData, false, false, false);
        }
        if (chart.series[1]) {
            chart.series[1].setData(oiData, false, false, false);
        }
        
        chart.redraw(false);
    });
}

// ============== WEBSOCKET ==============

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/futures`;
    
    try {
        state.ws = new WebSocket(wsUrl);
        
        state.ws.onopen = () => {
            state.wsConnected = true;
            updateConnectionStatus(true);
            console.log('WebSocket connected');
            sendWsSubscription();
        };
        
        state.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'update') {
                    processWsUpdate(data);
                }
            } catch (e) {
                console.error('WebSocket message error:', e);
            }
        };
        
        state.ws.onclose = () => {
            state.wsConnected = false;
            updateConnectionStatus(false);
            console.log('WebSocket disconnected, reconnecting in 3s...');
            setTimeout(connectWebSocket, 3000);
        };
        
        state.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
    } catch (e) {
        console.error('WebSocket connection failed:', e);
        setTimeout(connectWebSocket, 5000);
    }
}

function sendWsSubscription() {
    if (!state.ws || !state.wsConnected) return;
    
    state.ws.send(JSON.stringify({
        action: 'subscribe',
        smooth: false
    }));
}

function processWsUpdate(data) {
    if (data.normalized) {
        state.normalizedData = data.normalized;
    }
    if (data.time_seconds) {
        state.timeSeconds = data.time_seconds;
    }
    updateCharts();
}

function updateConnectionStatus(connected) {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    
    if (dot) dot.classList.toggle('connected', connected);
    if (text) text.textContent = connected ? 'Live' : 'Disconnected';
}

// ============== UTILITIES ==============

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.classList.toggle('hidden', !show);
}

function showErrorMessage(message) {
    let errorEl = document.getElementById('errorMessage');
    
    // Create error element if doesn't exist
    if (!errorEl) {
        errorEl = document.createElement('div');
        errorEl.id = 'errorMessage';
        errorEl.style.cssText = `
            position: fixed;
            top: 70px;
            left: 50%;
            transform: translateX(-50%);
            background: #21262d;
            border: 1px solid #d29922;
            color: #f0f6fc;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 14px;
            z-index: 9998;
            display: none;
        `;
        document.body.appendChild(errorEl);
    }
    
    if (message) {
        errorEl.textContent = message;
        errorEl.style.display = 'block';
    } else {
        errorEl.style.display = 'none';
    }
}
