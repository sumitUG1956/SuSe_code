/**
 * Algo Normalized Options Dashboard
 * 
 * Features:
 * - 8 synchronized charts (4 metrics × CE/PE)
 * - WebSocket for real-time updates
 * - Strike price selection
 * - Index switching (NIFTY/BANKNIFTY/SENSEX)
 * - Expiry selection
 */

// ============== GLOBAL STATE ==============
const state = {
    currentIndex: 'NIFTY',
    currentExpiry: null,
    availableExpiries: [],
    availableStrikes: [],
    selectedCEStrikes: new Set(),
    selectedPEStrikes: new Set(),
    atmStrike: null,
    
    // Normalized data from API
    normalizedData: {},
    timeSeconds: [],
    
    // Charts
    charts: {},
    
    // WebSocket
    ws: null,
    wsConnected: false,
};

// Chart definitions
const CHART_CONFIG = [
    { id: 'ce-iv-raw', optType: 'CE', metric: 'iv', label: 'IV Norm' },
    { id: 'pe-iv-raw', optType: 'PE', metric: 'iv', label: 'IV Norm' },
    { id: 'ce-oi', optType: 'CE', metric: 'oi_diff_cumsum', label: 'OI Diff' },
    { id: 'pe-oi', optType: 'PE', metric: 'oi_diff_cumsum', label: 'OI Diff' },
    { id: 'ce-tvv', optType: 'CE', metric: 'timevalue_vol_prod_cumsum', label: 'TV×Vol' },
    { id: 'pe-tvv', optType: 'PE', metric: 'timevalue_vol_prod_cumsum', label: 'TV×Vol' },
    { id: 'ce-tv', optType: 'CE', metric: 'timevalue_diff_cumsum', label: 'TV Diff' },
    { id: 'pe-tv', optType: 'PE', metric: 'timevalue_diff_cumsum', label: 'TV Diff' },
];

// Colors for different strikes
const STRIKE_COLORS = [
    '#58a6ff', '#3fb950', '#f85149', '#a371f7', '#f0883e',
    '#56d364', '#db6d28', '#bc8cff', '#79c0ff', '#7ee787',
];

// ============== INITIALIZATION ==============

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initEventListeners();
    loadData();
    connectWebSocket();
});

function initCharts() {
    // Highcharts global options - optimized for performance
    Highcharts.setOptions({
        time: {
            useUTC: true
        },
        chart: {
            backgroundColor: '#161b22',
            style: { fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' },
            animation: false
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
                style: { color: '#8b949e', fontSize: '10px' },
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
                style: { color: '#8b949e', fontSize: '10px' },
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
            style: { color: '#f0f6fc' },
            shared: false,
            xDateFormat: '%H:%M:%S',
            snap: 10,
            valueDecimals: 2,
            pointFormat: '<span style="color:{point.color}">●</span> {series.name}: <b>{point.y:.2f}</b><br/>'
        },
        plotOptions: {
            series: {
                animation: false,
                marker: { enabled: false },
                lineWidth: 1.5,
                findNearestPointBy: 'xy',
                states: {
                    hover: {
                        lineWidth: 2.5
                    },
                    inactive: {
                        opacity: 0.3
                    }
                },
                turboThreshold: 5000,
                enableMouseTracking: true,
                stickyTracking: false
            }
        }
    });
    
    // Create all charts
    CHART_CONFIG.forEach(config => {
        const container = document.getElementById(`chart-${config.id}`);
        if (!container) return;
        
        state.charts[config.id] = Highcharts.chart(container, {
            chart: {
                type: 'line',
                height: 400,
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
                    // Zero line - dark
                    {
                        value: 0,
                        color: '#8b949e',
                        width: 2,
                        zIndex: 3
                    },
                    // +3 threshold line - dark
                    {
                        value: 3,
                        color: '#8b949e',
                        width: 1.5,
                        dashStyle: 'Dash',
                        zIndex: 3
                    },
                    // -3 threshold line - dark
                    {
                        value: -3,
                        color: '#8b949e',
                        width: 1.5,
                        dashStyle: 'Dash',
                        zIndex: 3
                    }
                ],
                plotBands: [
                    // Above +3 - light green
                    {
                        from: 3,
                        to: Infinity,
                        color: 'rgba(63, 185, 80, 0.15)',
                        zIndex: 0
                    },
                    // Below -3 - light red
                    {
                        from: -Infinity,
                        to: -3,
                        color: 'rgba(248, 81, 73, 0.15)',
                        zIndex: 0
                    }
                ]
            },
            series: []
        });
    });
    
    // Setup crosshair sync
    setupCrosshairSync();
}

function setupCrosshairSync() {
    // Synchronized crosshair for all charts
    const allChartContainers = document.querySelectorAll('.chart-box');
    
    allChartContainers.forEach(container => {
        ['mousemove', 'touchmove', 'touchstart'].forEach(eventType => {
            container.addEventListener(eventType, function(e) {
                syncCrosshairs(e, this);
            });
        });
        
        container.addEventListener('mouseleave', function() {
            // Hide all crosshairs when mouse leaves any chart
            Object.values(state.charts).forEach(chart => {
                chart.xAxis[0]?.hideCrosshair();
                chart.tooltip?.hide();
            });
        });
    });
}

function syncCrosshairs(e, container) {
    // Find which chart is in this container
    const sourceChart = Object.values(state.charts).find(c => 
        c.container && (c.container === container || c.container.parentElement === container || container.contains(c.container))
    );
    if (!sourceChart) return;
    
    const event = sourceChart.pointer.normalize(e);
    const plotX = event.chartX - sourceChart.plotLeft;
    
    // Check if mouse is in plot area
    if (plotX < 0 || plotX > sourceChart.plotWidth) return;
    
    // Find the closest timestamp from any series in source chart
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
    
    // Update time display
    updateCrosshairTime(closestTimestamp);
    
    // Draw crosshair on ALL charts at this timestamp
    Object.values(state.charts).forEach(chart => {
        const xAxis = chart.xAxis[0];
        if (!xAxis) return;
        
        // Find a point at this timestamp
        let pointAtTime = null;
        for (const series of chart.series) {
            if (series.points) {
                pointAtTime = series.points.find(p => p.x === closestTimestamp);
                if (pointAtTime) break;
            }
        }
        
        if (pointAtTime) {
            // Draw crosshair using the point
            xAxis.drawCrosshair({
                chartX: pointAtTime.plotX + chart.plotLeft,
                chartY: pointAtTime.plotY + chart.plotTop
            }, pointAtTime);
        } else {
            // No point at this time, calculate X position manually
            const xPixel = xAxis.toPixels(closestTimestamp);
            if (xPixel >= chart.plotLeft && xPixel <= chart.plotLeft + chart.plotWidth) {
                xAxis.drawCrosshair({ chartX: xPixel });
            }
        }
    });
    
    // Update value displays
    updateValueDisplays(closestTimestamp);
}

function syncExtremes(e) {
    if (e.trigger === 'syncExtremes') return;
    
    const thisChart = this.chart;
    Object.values(state.charts).forEach(chart => {
        if (chart !== thisChart) {
            if (chart.xAxis[0].setExtremes) {
                chart.xAxis[0].setExtremes(e.min, e.max, true, false, { trigger: 'syncExtremes' });
            }
        }
    });
}

function updateCrosshairTime(timestamp) {
    const timeEl = document.getElementById('crosshairTime');
    if (timeEl && timestamp) {
        const date = new Date(timestamp);
        const hours = String(date.getHours()).padStart(2, '0');
        const mins = String(date.getMinutes()).padStart(2, '0');
        timeEl.textContent = `${hours}:${mins}`;
    }
}

function updateValueDisplays(timestamp) {
    CHART_CONFIG.forEach(config => {
        const chart = state.charts[config.id];
        if (!chart) return;
        
        const valueEl = document.getElementById(`val-${config.id}`);
        if (!valueEl) return;
        
        // Get all values at this timestamp
        const values = [];
        chart.series.forEach(series => {
            const point = series.points?.find(p => p.x === timestamp);
            if (point && point.y != null) {
                values.push(point.y.toFixed(2));
            }
        });
        
        valueEl.textContent = values.length > 0 ? values.join(' | ') : '--';
    });
}

// ============== DATA LOADING ==============

async function loadData() {
    showLoading(true);
    
    try {
        const response = await fetch(`/api/normalized/${state.currentIndex}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        processNormalizedData(data);
        
    } catch (error) {
        console.error('Failed to load data:', error);
    } finally {
        showLoading(false);
    }
}

function processNormalizedData(data) {
    state.normalizedData = data.normalized || {};
    state.timeSeconds = data.time_seconds || [];
    
    // Parse available expiries and strikes from column names
    const expiries = new Set();
    const strikes = new Set();
    
    Object.keys(state.normalizedData).forEach(colName => {
        // Column format: DEC24_24000CE_iv_diff_cumsum
        const match = colName.match(/^([A-Z]{3}\d{2})_(\d+)(CE|PE)_/);
        if (match) {
            expiries.add(match[1]);
            strikes.add(parseInt(match[2]));
        }
    });
    
    state.availableExpiries = Array.from(expiries).sort();
    state.availableStrikes = Array.from(strikes).sort((a, b) => a - b);
    
    // Set default expiry
    if (!state.currentExpiry && state.availableExpiries.length > 0) {
        state.currentExpiry = state.availableExpiries[0];
    }
    
    // Find ATM strike (closest to current spot)
    if (data.spot_price && state.availableStrikes.length > 0) {
        state.atmStrike = state.availableStrikes.reduce((prev, curr) => 
            Math.abs(curr - data.spot_price) < Math.abs(prev - data.spot_price) ? curr : prev
        );
    } else if (state.availableStrikes.length > 0) {
        // Default to middle strike
        state.atmStrike = state.availableStrikes[Math.floor(state.availableStrikes.length / 2)];
    }
    
    // Default: select ATM strike
    if (state.selectedCEStrikes.size === 0 && state.atmStrike) {
        state.selectedCEStrikes.add(state.atmStrike);
        state.selectedPEStrikes.add(state.atmStrike);
    }
    
    // Update UI
    renderExpiryTabs();
    renderStrikeButtons();
    updateCharts();
}

// ============== UI RENDERING ==============

function renderExpiryTabs() {
    const container = document.getElementById('expiryTabs');
    if (!container) return;
    
    container.innerHTML = state.availableExpiries.map(exp => `
        <button class="expiry-tab ${exp === state.currentExpiry ? 'active' : ''}" 
                data-expiry="${exp}">${exp}</button>
    `).join('');
}

function renderStrikeButtons() {
    const ceContainer = document.getElementById('ceStrikes');
    const peContainer = document.getElementById('peStrikes');
    if (!ceContainer || !peContainer) return;
    
    // CE buttons
    ceContainer.innerHTML = state.availableStrikes.map(strike => {
        const isActive = state.selectedCEStrikes.has(strike);
        const isATM = strike === state.atmStrike;
        return `<button class="strike-btn ${isActive ? 'active-ce' : ''} ${isATM ? 'atm-btn' : ''}" 
                        data-strike="${strike}" data-type="CE">${strike}</button>`;
    }).join('');
    
    // PE buttons  
    peContainer.innerHTML = state.availableStrikes.map(strike => {
        const isActive = state.selectedPEStrikes.has(strike);
        const isATM = strike === state.atmStrike;
        return `<button class="strike-btn ${isActive ? 'active-pe' : ''} ${isATM ? 'atm-btn' : ''}"
                        data-strike="${strike}" data-type="PE">${strike}</button>`;
    }).join('');
}

function initEventListeners() {
    // Index tabs
    document.querySelectorAll('.index-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.index-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            state.currentIndex = e.target.dataset.index;
            state.currentExpiry = null;
            state.selectedCEStrikes.clear();
            state.selectedPEStrikes.clear();
            loadData();
        });
    });
    
    // Expiry tabs (delegated)
    document.getElementById('expiryTabs')?.addEventListener('click', (e) => {
        if (e.target.classList.contains('expiry-tab')) {
            document.querySelectorAll('.expiry-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            state.currentExpiry = e.target.dataset.expiry;
            updateCharts();
        }
    });
    
    // Strike buttons (delegated)
    document.getElementById('ceStrikes')?.addEventListener('click', (e) => {
        if (e.target.classList.contains('strike-btn')) {
            toggleStrike('CE', parseInt(e.target.dataset.strike));
        }
    });
    
    document.getElementById('peStrikes')?.addEventListener('click', (e) => {
        if (e.target.classList.contains('strike-btn')) {
            toggleStrike('PE', parseInt(e.target.dataset.strike));
        }
    });
}

function toggleStrike(type, strike) {
    const set = type === 'CE' ? state.selectedCEStrikes : state.selectedPEStrikes;
    
    if (set.has(strike)) {
        set.delete(strike);
    } else {
        set.add(strike);
    }
    
    renderStrikeButtons();
    updateCharts();
}

// ============== CHART UPDATES ==============

// Debounce and RAF for smooth updates
let updatePending = false;
let updateQueued = false;
const BASE_DATE = Date.UTC(2025, 0, 1, 0, 0, 0);

function updateCharts(forceRedraw = false) {
    if (forceRedraw) {
        _doUpdateCharts();
        return;
    }
    
    // Queue update if one is pending
    if (updatePending) {
        updateQueued = true;
        return;
    }
    
    updatePending = true;
    requestAnimationFrame(() => {
        _doUpdateCharts();
        updatePending = false;
        
        // Process queued update
        if (updateQueued) {
            updateQueued = false;
            requestAnimationFrame(() => _doUpdateCharts());
        }
    });
}

function _doUpdateCharts() {
    if (!state.currentExpiry || state.timeSeconds.length === 0) return;
    
    // Pre-compute timestamps once
    const timestamps = state.timeSeconds.map(t => BASE_DATE + (t * 1000));
    const len = timestamps.length;
    
    // Batch all chart updates
    CHART_CONFIG.forEach(config => {
        const chart = state.charts[config.id];
        if (!chart) return;
        
        const selectedStrikes = config.optType === 'CE' ? state.selectedCEStrikes : state.selectedPEStrikes;
        const strikesArray = Array.from(selectedStrikes);
        
        // Build all series data first
        const newSeriesData = [];
        strikesArray.forEach((strike, colorIndex) => {
            const colName = `${state.currentExpiry}_${strike}${config.optType}_${config.metric}`;
            const values = state.normalizedData[colName];
            if (!values) return;
            
            // Build data array efficiently
            const data = [];
            const vLen = Math.min(len, values.length);
            for (let i = 0; i < vLen; i++) {
                const v = values[i];
                if (v != null && v === v) { // v === v is faster NaN check
                    data.push([timestamps[i], v]);
                }
            }
            
            if (data.length > 0) {
                newSeriesData.push({
                    name: `${strike}`,
                    data: data,
                    color: STRIKE_COLORS[colorIndex % STRIKE_COLORS.length]
                });
            }
        });
        
        // Get existing series
        const existingNames = new Set(chart.series.map(s => s.name));
        const newNames = new Set(newSeriesData.map(s => s.name));
        
        // Remove old series (backwards to avoid index issues)
        for (let i = chart.series.length - 1; i >= 0; i--) {
            if (!newNames.has(chart.series[i].name)) {
                chart.series[i].remove(false);
            }
        }
        
        // Update or add series
        newSeriesData.forEach(sd => {
            const existing = chart.series.find(s => s.name === sd.name);
            if (existing) {
                existing.setData(sd.data, false, false, false);
            } else {
                chart.addSeries(sd, false);
            }
        });
        
        // Single redraw
        chart.redraw(false);
    });
}

// ============== WEBSOCKET ==============

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/normalized`;
    
    try {
        state.ws = new WebSocket(wsUrl);
        
        state.ws.onopen = () => {
            state.wsConnected = true;
            updateConnectionStatus(true);
            console.log('WebSocket connected');
            
            // Send current index subscription
            state.ws.send(JSON.stringify({ action: 'subscribe', index: state.currentIndex }));
        };
        
        state.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'update' && data.index === state.currentIndex) {
                    processNormalizedData(data);
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

function updateConnectionStatus(connected) {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    
    if (dot) {
        dot.classList.toggle('connected', connected);
    }
    if (text) {
        text.textContent = connected ? 'Live' : 'Disconnected';
    }
}

// ============== UTILITIES ==============

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.toggle('hidden', !show);
    }
}
