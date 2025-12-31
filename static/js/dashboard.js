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
    
    // Normalized data from API (lazy loaded per strike)
    normalizedData: {},
    loadedStrikes: new Set(),  // Track which strikes have been loaded
    timeSeconds: [],
    
    // Charts
    charts: {},
    
    // WebSocket
    ws: null,
    wsConnected: false,
    
    // Loading state
    loadingStrikes: new Set(),  // Track strikes currently being loaded
    
    // Display mode: 'ema' (smooth) or 'raw' (zigzag)
    displayMode: 'ema',
    
    // Zoom mode: true = drag-to-zoom enabled, false = disabled
    zoomEnabled: false,
};

// Toggle zoom mode
function toggleZoomMode() {
    state.zoomEnabled = !state.zoomEnabled;
    
    const btn = document.getElementById('zoomModeBtn');
    if (state.zoomEnabled) {
        btn.textContent = '🔍 Zoom On';
        btn.style.background = 'var(--accent-blue)';
        btn.style.color = 'white';
        btn.title = 'Zoom enabled - drag to zoom, click to disable';
    } else {
        btn.textContent = '🔍 Zoom Off';
        btn.style.background = '';
        btn.style.color = '';
        btn.title = 'Enable drag-to-zoom';
    }
    
    // Update all charts zoom type
    Object.values(state.charts).forEach(chart => {
        if (chart) {
            chart.update({
                chart: {
                    zoomType: state.zoomEnabled ? 'x' : undefined
                }
            }, false);
            
            // Reset zoom if disabling
            if (!state.zoomEnabled) {
                chart.zoomOut();
            }
        }
    });
}

// Chart definitions - OI first, then Volume → Premium Decay → IV → Greeks
// metric is base name, _ema suffix added for smooth display
const CHART_CONFIG = [
    { id: 'ce-oi', optType: 'CE', metric: 'oi_diff_cumsum', label: 'Open Interest' },
    { id: 'pe-oi', optType: 'PE', metric: 'oi_diff_cumsum', label: 'Open Interest' },
    { id: 'ce-tvv', optType: 'CE', metric: 'timevalue_vol_prod_cumsum', label: 'Volume' },
    { id: 'pe-tvv', optType: 'PE', metric: 'timevalue_vol_prod_cumsum', label: 'Volume' },
    { id: 'ce-tv', optType: 'CE', metric: 'timevalue_diff_cumsum', label: 'Premium Decay' },
    { id: 'pe-tv', optType: 'PE', metric: 'timevalue_diff_cumsum', label: 'Premium Decay' },
    { id: 'ce-iv', optType: 'CE', metric: 'iv_diff_cumsum', label: 'Implied Volatility' },
    { id: 'pe-iv', optType: 'PE', metric: 'iv_diff_cumsum', label: 'Implied Volatility' },
    { id: 'ce-theta', optType: 'CE', metric: 'theta_diff_cumsum', label: 'Theta' },
    { id: 'pe-theta', optType: 'PE', metric: 'theta_diff_cumsum', label: 'Theta' },
    { id: 'ce-vega', optType: 'CE', metric: 'vega_diff_cumsum', label: 'Vega' },
    { id: 'pe-vega', optType: 'PE', metric: 'vega_diff_cumsum', label: 'Vega' },
    { id: 'ce-delta', optType: 'CE', metric: 'delta_diff_cumsum', label: 'Delta' },
    { id: 'pe-delta', optType: 'PE', metric: 'delta_diff_cumsum', label: 'Delta' },
];

// Colors for different strikes - high contrast for dark mode
const STRIKE_COLORS = [
    '#00BFFF',  // Deep Sky Blue
    '#00FF7F',  // Spring Green
    '#FF6347',  // Tomato Red
    '#DDA0DD',  // Plum Purple
    '#FFD700',  // Gold
    '#00FFFF',  // Cyan
    '#FF69B4',  // Hot Pink
    '#7FFF00',  // Chartreuse
    '#FF8C00',  // Dark Orange
    '#87CEEB',  // Sky Blue
];

// Map strike to its color index for consistent coloring
const strikeColorMap = new Map();
let nextColorIndex = 0;

function getStrikeColor(strike) {
    if (!strikeColorMap.has(strike)) {
        strikeColorMap.set(strike, nextColorIndex);
        nextColorIndex = (nextColorIndex + 1) % STRIKE_COLORS.length;
    }
    return STRIKE_COLORS[strikeColorMap.get(strike)];
}

function resetStrikeColors() {
    strikeColorMap.clear();
    nextColorIndex = 0;
}

// ============== INITIALIZATION ==============

document.addEventListener('DOMContentLoaded', async () => {
    initCharts();
    initEventListeners();
    loadData();
    connectWebSocket();
    
    // Handle browser zoom/resize - reflow all charts automatically
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            // Reflow all charts to fit new dimensions
            Object.values(state.charts).forEach(chart => {
                if (chart) {
                    // Force container width recalculation
                    const container = chart.container.parentNode;
                    if (container) {
                        chart.setSize(container.offsetWidth, null, false);
                    }
                }
            });
        }, 150);  // Debounce 150ms
    });
});

function initCharts() {
    // Highcharts global options - optimized for performance
    Highcharts.setOptions({
        accessibility: {
            enabled: false  // Disable accessibility warning
        },
        time: {
            useUTC: true
        },
        chart: {
            backgroundColor: '#161b22',
            style: { fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' },
            animation: false,
            reflow: true,  // Enable auto-reflow
            spacingTop: 10,
            spacingBottom: 10,
            spacingLeft: 10,
            spacingRight: 10
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
                style: { color: '#c9d1d9', fontSize: '12px', fontWeight: '500' },
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
                style: { color: '#c9d1d9', fontSize: '12px', fontWeight: '500' },
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
                height: 550,
                zoomType: undefined,  // Disabled by default, enabled via button
                resetZoomButton: {
                    theme: {
                        fill: '#21262d',
                        stroke: '#58a6ff',
                        style: {
                            color: '#58a6ff',
                            fontWeight: 'bold'
                        },
                        states: {
                            hover: {
                                fill: '#58a6ff',
                                style: { color: '#fff' }
                            }
                        }
                    },
                    position: {
                        align: 'right',
                        verticalAlign: 'top',
                        x: -10,
                        y: 10
                    }
                },
                events: {
                    load: function() {
                        this.pointer.reset = function() { return undefined; };
                    }
                }
            },
            xAxis: {
                min: BASE_DATE + (MARKET_OPEN * 1000),   // 9:15 AM
                max: BASE_DATE + (MARKET_CLOSE * 1000),  // 3:30 PM
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
        const hours = String(date.getUTCHours()).padStart(2, '0');
        const mins = String(date.getUTCMinutes()).padStart(2, '0');
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

function resetState(preserveExpiry = false) {
    // Clear all cached/stale data on fresh load
    const savedExpiry = state.currentExpiry;
    const savedIndex = state.currentIndex;
    
    state.currentExpiry = null;
    state.availableExpiries = [];
    state.availableStrikes = [];
    state.selectedCEStrikes = new Set();
    state.selectedPEStrikes = new Set();
    state.atmStrike = null;
    state.normalizedData = {};
    state.loadedStrikes = new Set();
    state.timeSeconds = [];
    state.loadingStrikes = new Set();
    
    // Reset strike colors for fresh assignment
    resetStrikeColors();
    
    // Restore expiry if requested (for expiry tab change)
    if (preserveExpiry) {
        state.currentExpiry = savedExpiry;
    }
    state.currentIndex = savedIndex;
}

async function loadData(showOverlay = true, retryCount = 0, skipReset = false) {
    if (showOverlay) showLoading(true);
    
    // Reset state on first load to clear any stale browser cache
    // Skip reset when explicitly requested (expiry change clears manually)
    if (retryCount === 0 && !skipReset) {
        resetState();
    }
    
    try {
        // First, load metadata only (no strikes) to get available expiries and strikes
        let url = `/api/normalized/${state.currentIndex}`;
        const params = new URLSearchParams();
        if (state.currentExpiry) {
            params.append('expiry', state.currentExpiry);
        }
        // Add smooth parameter based on display mode
        params.append('smooth', state.displayMode === 'ema' ? 'true' : 'false');
        // Don't request any strikes initially - just get metadata
        
        if (params.toString()) {
            url += `?${params.toString()}`;
        }
        
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        
        // Check if server has data ready (strikes available)
        if (!data.available_strikes || data.available_strikes.length === 0) {
            if (retryCount < 5) {
                console.log(`⏳ Server warming up, retry ${retryCount + 1}/5 in 2s...`);
                setTimeout(() => loadData(showOverlay, retryCount + 1), 2000);
                return;
            } else {
                console.warn('Server has no strike data after 5 retries');
            }
        }
        
        processMetadata(data);
        
        // Now load data for selected strikes (ATM by default)
        await loadSelectedStrikesData();
        
    } catch (error) {
        console.error('Failed to load data:', error);
        // Retry on network error (server might be starting)
        if (retryCount < 3) {
            console.log(`🔄 Connection error, retry ${retryCount + 1}/3 in 3s...`);
            setTimeout(() => loadData(showOverlay, retryCount + 1), 3000);
            return;
        }
    } finally {
        if (showOverlay) showLoading(false);
    }
}

function processMetadata(data) {
    // Get available expiries
    if (data.available_expiries && data.available_expiries.length > 0) {
        state.availableExpiries = data.available_expiries;
    }
    
    // Get available strikes from API response
    if (data.available_strikes && data.available_strikes.length > 0) {
        state.availableStrikes = data.available_strikes;
    }
    
    // Set current expiry - DON'T overwrite if user already selected one
    if (!state.currentExpiry) {
        // Only set from API if not already set by user
        if (data.current_expiry) {
            state.currentExpiry = data.current_expiry;
        } else if (state.availableExpiries.length > 0) {
            state.currentExpiry = state.availableExpiries[0];
        }
    }
    
    // Store time_seconds
    state.timeSeconds = data.time_seconds || [];
    
    // Find ATM strike
    if (data.spot_price && state.availableStrikes.length > 0) {
        state.atmStrike = state.availableStrikes.reduce((prev, curr) => 
            Math.abs(curr - data.spot_price) < Math.abs(prev - data.spot_price) ? curr : prev
        );
    } else if (state.availableStrikes.length > 0) {
        state.atmStrike = state.availableStrikes[Math.floor(state.availableStrikes.length / 2)];
    }
    
    // Default: select ATM strike if nothing selected
    if (state.selectedCEStrikes.size === 0 && state.atmStrike) {
        state.selectedCEStrikes.add(state.atmStrike);
        state.selectedPEStrikes.add(state.atmStrike);
    }
    
    // Merge any incoming normalized data
    if (data.normalized) {
        Object.assign(state.normalizedData, data.normalized);
        // Mark loaded strikes
        Object.keys(data.normalized).forEach(colName => {
            const match = colName.match(/^([A-Z]{3}\d{2})_(\d+)(CE|PE)_/);
            if (match) {
                state.loadedStrikes.add(parseInt(match[2]));
            }
        });
    }
    
    // Update UI
    renderExpiryTabs();
    renderStrikeButtons();
}

async function loadSelectedStrikesData() {
    // Get all selected strikes that haven't been loaded yet
    const allSelected = new Set([...state.selectedCEStrikes, ...state.selectedPEStrikes]);
    const strikesToLoad = [...allSelected].filter(s => !state.loadedStrikes.has(s));
    
    if (strikesToLoad.length === 0) {
        updateCharts();
        return;
    }
    
    // Mark as loading
    strikesToLoad.forEach(s => state.loadingStrikes.add(s));
    renderStrikeButtons();  // Show loading state
    
    try {
        let url = `/api/normalized/${state.currentIndex}`;
        const params = new URLSearchParams();
        if (state.currentExpiry) {
            params.append('expiry', state.currentExpiry);
        }
        params.append('strikes', strikesToLoad.join(','));
        // Add smooth parameter - only fetch EMA or raw based on mode
        params.append('smooth', state.displayMode === 'ema' ? 'true' : 'false');
        url += `?${params.toString()}`;
        
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        
        // Merge normalized data
        if (data.normalized) {
            Object.assign(state.normalizedData, data.normalized);
        }
        
        // Update time_seconds if newer
        if (data.time_seconds && data.time_seconds.length > state.timeSeconds.length) {
            state.timeSeconds = data.time_seconds;
        }
        
        // Mark strikes as loaded
        strikesToLoad.forEach(s => {
            state.loadedStrikes.add(s);
            state.loadingStrikes.delete(s);
        });
        
    } catch (error) {
        console.error('Failed to load strike data:', error);
        strikesToLoad.forEach(s => state.loadingStrikes.delete(s));
    }
    
    renderStrikeButtons();
    updateCharts();
}

function processNormalizedData(data) {
    // Process WebSocket updates
    state.timeSeconds = data.time_seconds || state.timeSeconds;
    
    // Merge normalized data (only for loaded strikes)
    if (data.normalized) {
        Object.assign(state.normalizedData, data.normalized);
    }
    
    // Update available expiries if provided
    if (data.available_expiries && data.available_expiries.length > 0) {
        state.availableExpiries = data.available_expiries;
    }
    
    // Update available strikes if provided
    if (data.available_strikes && data.available_strikes.length > 0) {
        state.availableStrikes = data.available_strikes;
    }
    
    // Set expiry
    if (data.current_expiry) {
        state.currentExpiry = data.current_expiry;
    }
    
    // Find ATM
    if (data.spot_price && state.availableStrikes.length > 0) {
        state.atmStrike = state.availableStrikes.reduce((prev, curr) => 
            Math.abs(curr - data.spot_price) < Math.abs(prev - data.spot_price) ? curr : prev
        );
    }
    
    // Default selection
    if (state.selectedCEStrikes.size === 0 && state.atmStrike) {
        state.selectedCEStrikes.add(state.atmStrike);
        state.selectedPEStrikes.add(state.atmStrike);
    }
    
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
    
    // CE buttons - with color indicator matching chart line
    ceContainer.innerHTML = state.availableStrikes.map(strike => {
        const isActive = state.selectedCEStrikes.has(strike);
        const isATM = strike === state.atmStrike;
        const isLoading = state.loadingStrikes.has(strike);
        const isLoaded = state.loadedStrikes.has(strike);
        const color = isActive ? getStrikeColor(strike) : '';
        const style = isActive ? `background: ${color}; border-color: ${color}; color: #000; font-weight: 600;` : '';
        return `<button class="strike-btn ${isATM && !isActive ? 'atm-btn' : ''} ${isLoading ? 'loading' : ''}" 
                        style="${style}"
                        data-strike="${strike}" data-type="CE">${isLoading ? '⏳' : ''}${strike}</button>`;
    }).join('');
    
    // PE buttons - with color indicator matching chart line
    peContainer.innerHTML = state.availableStrikes.map(strike => {
        const isActive = state.selectedPEStrikes.has(strike);
        const isATM = strike === state.atmStrike;
        const isLoading = state.loadingStrikes.has(strike);
        const isLoaded = state.loadedStrikes.has(strike);
        const color = isActive ? getStrikeColor(strike) : '';
        const style = isActive ? `background: ${color}; border-color: ${color}; color: #000; font-weight: 600;` : '';
        return `<button class="strike-btn ${isATM && !isActive ? 'atm-btn' : ''} ${isLoading ? 'loading' : ''}"
                        style="${style}"
                        data-strike="${strike}" data-type="PE">${isLoading ? '⏳' : ''}${strike}</button>`;
    }).join('');
}

function initEventListeners() {
    // Index tabs
    document.querySelectorAll('.index-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.index-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            state.currentIndex = e.target.dataset.index;
            state.currentExpiry = null;  // Reset expiry when changing index
            state.selectedCEStrikes.clear();
            state.selectedPEStrikes.clear();
            state.loadedStrikes.clear();  // Clear loaded strikes cache
            state.normalizedData = {};    // Clear cached data
            loadData(false);  // Don't show full loading overlay
            
            // Re-subscribe WebSocket with new index
            sendWsSubscription();
        });
    });
    
    // Expiry tabs (delegated) - reload data with new expiry filter
    document.getElementById('expiryTabs')?.addEventListener('click', (e) => {
        if (e.target.classList.contains('expiry-tab')) {
            document.querySelectorAll('.expiry-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            const newExpiry = e.target.dataset.expiry;
            
            // Clear strike-related cache when changing expiry (keep index same)
            state.selectedCEStrikes.clear();
            state.selectedPEStrikes.clear();
            state.loadedStrikes.clear();
            state.normalizedData = {};
            state.atmStrike = null;
            state.availableStrikes = [];
            state.timeSeconds = [];
            
            // Set new expiry AFTER clearing (so loadData doesn't reset it)
            state.currentExpiry = newExpiry;
            
            // Reload data with new expiry filter (skipReset=true to preserve expiry)
            loadData(false, 0, true);
            
            // Re-subscribe WebSocket with new expiry
            sendWsSubscription();
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
        renderStrikeButtons();
        updateCharts();
        // Update WebSocket subscription (now with fewer strikes)
        sendWsSubscription();
    } else {
        set.add(strike);
        renderStrikeButtons();
        
        // Lazy load: fetch data for this strike if not already loaded
        if (!state.loadedStrikes.has(strike)) {
            loadSelectedStrikesData().then(() => {
                // Update WebSocket subscription after data loaded
                sendWsSubscription();
            });
        } else {
            updateCharts();
            // Update WebSocket subscription (now with more strikes)
            sendWsSubscription();
        }
    }
}

// ============== CHART UPDATES ==============

// Debounce and RAF for smooth updates
let updatePending = false;
let updateQueued = false;
const BASE_DATE = Date.UTC(2025, 0, 1, 0, 0, 0);

// Market timing constants (in seconds from midnight)
const MARKET_OPEN = 33300;   // 9:15 AM IST
const MARKET_CLOSE = 55800;  // 3:30 PM IST

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
    
    // Column suffix based on display mode (API sends _ema columns when smooth=true)
    const suffix = state.displayMode === 'ema' ? '_ema' : '';
    
    // Batch all chart updates
    CHART_CONFIG.forEach(config => {
        const chart = state.charts[config.id];
        if (!chart) return;
        
        const selectedStrikes = config.optType === 'CE' ? state.selectedCEStrikes : state.selectedPEStrikes;
        const strikesArray = Array.from(selectedStrikes);
        
        // Build all series data first
        const newSeriesData = [];
        strikesArray.forEach((strike, colorIndex) => {
            // Column name matches API format
            const colName = `${state.currentExpiry}_${strike}${config.optType}_${config.metric}${suffix}`;
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
                    color: getStrikeColor(strike),
                    lineWidth: 2,
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
                existing.update({ lineWidth: sd.lineWidth }, false);
            } else {
                chart.addSeries(sd, false);
            }
        });
    });
    
    // Single batch redraw for ALL charts (much faster than individual redraws)
    Object.values(state.charts).forEach(chart => {
        if (chart) chart.redraw(false);
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
            
            // Subscribe with current index, expiry and selected strikes
            sendWsSubscription();
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

// Send WebSocket subscription with current selected strikes
function sendWsSubscription() {
    if (!state.ws || !state.wsConnected) return;
    
    const allSelectedStrikes = [...new Set([...state.selectedCEStrikes, ...state.selectedPEStrikes])];
    
    state.ws.send(JSON.stringify({ 
        action: 'subscribe', 
        index: state.currentIndex,
        expiry: state.currentExpiry,
        strikes: allSelectedStrikes.length > 0 ? allSelectedStrikes : null,
        smooth: state.displayMode === 'ema'
    }));
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

// ============== DISPLAY MODE TOGGLE ==============

function toggleDisplayMode() {
    state.displayMode = state.displayMode === 'ema' ? 'raw' : 'ema';
    
    // Update button text
    const btn = document.getElementById('displayModeBtn');
    if (btn) {
        btn.textContent = state.displayMode === 'ema' ? '📈 Smooth' : '📊 Raw';
        btn.title = state.displayMode === 'ema' ? 'Showing EMA smoothed data (click for raw)' : 'Showing raw data (click for smooth)';
    }
    
    // Clear cached data and reload - mode change means different data from server
    state.normalizedData = {};
    state.loadedStrikes.clear();
    
    // Reload data with new mode
    loadSelectedStrikesData().then(() => {
        sendWsSubscription();
    });
}

// ============== UTILITIES ==============

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.toggle('hidden', !show);
    }
}
