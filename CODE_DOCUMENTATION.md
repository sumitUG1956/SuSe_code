# CODE DOCUMENTATION (कोड डॉक्यूमेंटेशन)

यह document SuSe_code project के सभी major files की detailed explanation provide करता है।

## Project Overview (प्रोजेक्ट ओवरव्यू)

**Purpose:** Real-time options और futures data analysis system जो Upstox API से market data fetch करके advanced calculations (Greeks, Implied Volatility, Normalization) perform करता है।

**Technology Stack:**
- Python 3.x
- FastAPI (REST API + WebSocket server)
- NumPy/Pandas/Polars (Data processing)
- Async/Await (Concurrent operations)

---

## Files Overview (फाइलें ओवरव्यू)

### ✅ Fully Commented Files (पूरी तरह से commented)

1. **logger.py** - Simple logging utility
   - Timestamp के साथ INFO और ERROR messages log करता है
   - UTC timezone में consistent timestamps

2. **live_updates.py** - WebSocket pub/sub system
   - Real-time candle updates को subscribers तक push करता है
   - asyncio.Queue based architecture
   - Symbol-wise subscriptions manage करता है

3. **download_extract.py** - Upstox instruments downloader
   - Compressed (.gz) instrument list download करता है
   - Extract करके JSON format में convert करता है
   - data/ directory में store करता है

4. **report_filtered_instruments.py** - Report generator
   - Filtered instruments की human-readable summary बनाता है
   - Indices, futures, options सभी को formatted output में show करता है

5. **requirements.txt** - Python dependencies
   - सभी required packages की documented list
   - Installation instructions included

6. **calculations.py** - Calculation orchestrator
   - Candle fetch के बाद state management करता है
   - Fresh vs old candles filter करता है
   - Normalization trigger करता है (disabled यहाँ, background में runs)

7. **.gitignore** - Git ignore patterns
   - Python bytecode, virtual env, IDE files ignore करता है

---

## Detailed File Explanations (बाकी files की detailed व्याख्या)

### 🔧 candle_processing.py (849 lines)

**Purpose:** Candle data transformation और options pricing calculations

**Key Components:**

1. **Data Structures:**
   - `CANDLE_COLUMNS` - Time, Open, High, Low, Close, Volume, OpenInterest
   - Constants: `RISK_FREE_RATE`, `SIGMA_LOW`, `SIGMA_HIGH` (for Black-76 model)

2. **Main Functions:**

   **`_ensure_frame(candles)`** - Raw candles को Polars DataFrame में convert करता है
   - Missing columns add करता है (Volume, OpenInterest defaults के साथ)
   - Data types cast करता है (Float32 for efficiency)
   - Average column calculate करता है: mean(Open, High, Low, Close)

   **`transform_candles_for_today(candles, instrument_meta)`** - Complete transformation pipeline
   - Steps:
     1. DataFrame बनाओ और timezone convert करो (IST)
     2. Latest trading date filter करो
     3. Reverse करो (chronological order)
     4. Spot average attach करो (SPOT_CACHE से)
     5. Option Greeks calculate करो (IV, Delta, Vega, Theta)
     6. Diff और cumsum features add करो
     7. Future-spot difference features add करो

3. **Black-76 Option Pricing:**

   **`_black_76_price(F, K, T, r, sigma, is_call_mask)`** - Option price calculate करता है
   - F = Forward price (spot × e^(r×T))
   - K = Strike price
   - T = Time to expiry (years)
   - r = Risk-free rate
   - sigma = Implied volatility
   - Formula: Uses cumulative normal distribution (_norm_cdf)

   **`_vectorized_black_76_iv(option_price, ...)`** - Implied Volatility solver
   - Bisection method use करता है (binary search)
   - Range: [SIGMA_LOW, SIGMA_HIGH] = [0.001, 5.0]
   - Max iterations: 100
   - Returns: IV as percentage (e.g., 15.5% volatility)

4. **Greeks Calculation:**

   **`_vectorized_black_76_delta()`** - Delta (price sensitivity to spot)
   - Call Delta = discount × N(d1)
   - Put Delta = discount × (N(d1) - 1)

   **`_vectorized_black_76_vega_theta()`** - Vega और Theta
   - Vega = spot sensitivity to IV
   - Theta = time decay (per day price change)

5. **Feature Engineering:**

   **`_attach_spot_average()`** - Spot reference attach करता है
   - Cache में spot data store करता है (SPOT_CACHE)
   - Options/Futures को spot से join करता है

   **`_attach_option_diff_features()`** - Option-specific features
   - IV diff/cumsum
   - Delta diff/cumsum
   - Vega diff/cumsum
   - Theta diff/cumsum
   - TimeValue diff/cumsum
   - OpenInterest diff/cumsum
   - TimeValue × Volume products

   **`_attach_future_spot_features()`** - Future-specific features
   - Future-Spot difference
   - Cumulative difference
   - NaN until first valid value (normalization के लिए important)

   **`_attach_average_features()`** - Universal features
   - Average diff (price change)
   - Average diff cumsum
   - Average × Volume product
   - Product cumsum

**Algorithm Details:**

**Cumsum Logic (Important!):**
```python
# Leading NaN preservation - normalization के लिए critical
valid_seen = col.is_not_null().cum_sum() > 0
raw_cumsum = col.fill_null(0.0).cum_sum()
final = pl.when(valid_seen).then(raw_cumsum).otherwise(None)
```
- यह ensure करता है कि cumsum केवल first valid value के बाद start हो
- Leading zeros से normalization बर्बाद नहीं होगा

**Performance Optimizations:**
- Vectorized NumPy operations (loops avoid)
- Polars expressions (lazy evaluation)
- Batch processing (multiple diffs/cumsums एक साथ)

---

### 📊 combined_normalization.py (863 lines)

**Purpose:** सभी index options (NIFTY, BANKNIFTY, SENSEX) का combined normalization

**Strategy:** Hybrid approach for efficiency
- **FIRST RUN:** Pandas expanding() - Fast bulk calculation
- **INCREMENTAL:** NumPy loop - Only new rows (cached results reuse)

**Key Components:**

1. **Normalization Columns:**
```python
NORMALIZE_COLUMNS = (
    "iv_diff_cumsum",      # Implied Volatility changes
    "oi_diff_cumsum",      # Open Interest changes  
    "timevalue_diff_cumsum", # Time value changes
    "delta_diff_cumsum",   # Delta changes
    "theta_diff_cumsum",   # Theta changes
    "vega_diff_cumsum",    # Vega changes
)
```

2. **Main Functions:**

   **`normalize_index_options(index_name)`** - Single index normalize करता है
   - Steps:
     1. Spot data get करो (base time_seconds)
     2. सभी options symbols get करो
     3. Time-aligned matrix build करो
     4. Linear interpolation for gaps (middle gaps only)
     5. Normalize करो (IQR या Z-score method)
     6. Cache results (incremental updates के लिए)

   **`_build_combined_matrix(index_name, options_symbols, base_time)`**
   - सभी options को single matrix में align करता है
   - Column format: `EXPIRY_STRIKE_TYPE_metric`
   - Example: `DEC24_24000CE_iv_diff_cumsum`
   - Linear interpolation से gaps fill करता है

3. **Normalization Methods:**

   **IQR Method (Interquartile Range):**
   ```python
   normalized = (value - median) / IQR
   IQR = Q3 - Q1
   ```
   - Used for: Most metrics (iv_diff_cumsum, timevalue, etc.)
   - Dynamic floor: max(0.01, abs(median) × 0.1)
   - Robust to outliers

   **Z-Score Method:**
   ```python
   z-score = (value - mean) / std
   ```
   - Used for: OI diff cumsum only
   - Dynamic floor: max(0.01, abs(mean) × 0.1)

4. **Pandas Expanding Window:**
```python
exp_median = series.expanding(min_periods=1).median()
exp_q1 = series.expanding(min_periods=2).quantile(0.25)
exp_q3 = series.expanding(min_periods=2).quantile(0.75)
scaled = (series - exp_median) / IQR.clip(lower=dynamic_floor)
```
- Expanding window = growing window (1st value, 1st-2nd, 1st-3rd, ...)
- Fast because Pandas optimizes internally
- Used for bulk calculation (first run)

5. **NumPy Incremental:**
```python
for i in range(start_index, n):
    prefix = data[first_valid:i+1]
    med = np.median(prefix)
    q1, q3 = np.percentile(prefix, [25, 75])
    normalized[i] = (data[i] - med) / max(q3-q1, floor)
```
- Loop-based but only for new rows
- Reuses cached normalized values for old rows
- Efficient for small updates

6. **EMA Smoothing:**
```python
EMA_today = (value_today × multiplier) + (EMA_yesterday × (1 - multiplier))
multiplier = 2 / (period + 1)
```
- EMA_PERIOD = 12 (≈2 minutes for 10s data)
- Smooths normalized values for cleaner charts
- Applied after normalization

7. **Vega Skew:**
```python
skew = CE_Vega - PE_Vega  # For same strike
```
- Indicates market sentiment
- Positive skew = calls more expensive
- Negative skew = puts more expensive

**Cache Management:**
- State key: `{index_name}_COMBINED`
- Cached fields: `norm`, `normalized_size`, `cache_version`
- Cache invalidation on version change

**Performance Numbers:**
- First run: ~100-200ms for 2000 rows × 500 columns
- Incremental: ~10-20ms for +50 rows
- 10x faster than pure NumPy approach!

---

### 📦 extract.py (603 lines)

**Purpose:** Upstox instrument list को filter करके relevant contracts extract करना

**Key Components:**

1. **Configuration:**
```python
INDEX_TARGETS = [
    {"label": "NIFTY", "spot": {...}, "futures": {...}, "options": {...}},
    {"label": "BANKNIFTY", ...},
    {"label": "SENSEX", ...},
]

EQUITY_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", ...]
```

2. **Main Functions:**

   **`load_instruments()`** - Complete.json file load करता है
   - ~50MB JSON with ~100k instruments

   **`find_spot()`** - Spot instrument search करता है
   - Segment और trading_symbol match करता है

   **`find_futures()`** - Future contracts find करता है
   - Earliest expiry sort करके select करता है
   - Count parameter: कितने futures चाहिए (default 2)

   **`collect_options()`** - सभी options gather करता है
   - Asset symbol और segment match करता है
   - CE और PE दोनों types

   **`group_options_by_expiry()`** - Options को expiry से group करता है
   - Weekly vs monthly flag set करता है
   - Chronologically sort करता है

3. **Expiry Selection Logic:**

   **`select_expiries(current, secondary)`**
   - **Current expiry:** अभी active weekly/monthly
   - **Secondary expiry:** अगला monthly या next month weekly
   - Business rules:
     * अगर monthly same as current → next month weekly
     * अगर no weeklies → next monthly

4. **Strike Selection:**

   **`summarize_option_slice(contracts, spot_price)`**
   - ATM strike find करता है: closest to spot
   - ITM strikes select करता है: < spot
   - OTM strikes select करता है: > spot
   - Default: 10 strikes each side
   - Position labeling: ITM/ATM/OTM

5. **Live Spot Price Fetching:**

   **`fetch_live_spot_prices(instruments)`**
   - Upstox API call करता है: `/chart/open/v3/candles`
   - Time: Today 09:16 IST (market open candle)
   - Interval: S10 (10 seconds)
   - Close price extract करता है
   - Required for option slice calculation

6. **Payload Building:**

   **`build_filtered_payload(instruments, spot_overrides)`**
   - Spot prices get/override
   - Indices process करता है (spot, futures, options)
   - Equities process करता है (spot, futures only)
   - Returns: Complete filtered payload

   **`collect_trading_symbol_entries(payload)`**
   - Flattens nested structure
   - Creates catalog: instrument_key → metadata
   - Categories: index_spot, index_future, index_option, equity_spot, equity_future

7. **Manual Fallback:**
```python
MANUAL_SPOT_PRICES = {
    "NIFTY": 24000.0,
    "BANKNIFTY": 52000.0,
    "SENSEX": 79000.0,
}
```
- Used when API call fails
- Weekend/holiday debugging के लिए useful

**Error Handling:**
- API failures gracefully handle
- Fallback mechanisms available
- Clear error messages with debugging hints

---

### 🚀 fast_api.py (980 lines)

**Purpose:** FastAPI server with REST APIs और WebSocket endpoints

**Key Components:**

1. **Server Setup:**
```python
app = FastAPI(title="Upstox Instruments API")
app.mount("/static", StaticFiles(directory="static"))
```

2. **Startup Sequence:**

   **`on_startup()`**
   - Steps:
     1. Clean और download instruments
     2. Wait until 09:16 IST (market ready time)
     3. Fetch live spot prices
     4. Build payload
     5. Start candle fetcher background service

   **Weekend Detection:**
   - Saturday/Sunday को detect करता है
   - Faketime instructions provide करता है
   - Development debugging के लिए useful

3. **REST Endpoints:**

   **`GET /`** - Redirect to dashboard

   **`GET /dashboard`** - Options dashboard HTML

   **`GET /futures-dashboard`** - Futures dashboard HTML

   **`GET /health`** - Health check

   **`GET /payload`** - Complete filtered payload

   **`GET /download`** - Trigger fresh instrument download

   **`GET /candles/{symbol}`** - Symbol के candles get करो
   - NumPy snapshot से fast conversion
   - Vectorized time formatting
   - NaN handling (None में convert)

4. **Normalized Data API:**

   **`GET /api/normalized/{index_name}`**
   - Query params:
     * `expiry`: Filter by expiry (e.g., DEC24)
     * `strikes`: Comma-separated strikes (e.g., "24000,24100")
     * `smooth`: EMA smoothed (true/false)
   
   - Fast path (no strikes): Metadata only
   - Slow path (with strikes): Full normalization
   - Lazy loading architecture

   **`GET /api/futures/metadata`** - Futures metadata

   **`GET /api/futures/normalized`** - Futures normalized data

5. **WebSocket Endpoints:**

   **`WS /ws/candles/{symbol}`** - Live candle updates
   - Subscribe pattern
   - Queue-based push
   - Automatic cleanup on disconnect

   **`WS /ws/normalized`** - Live normalized updates
   - Subscribe with: `{"action": "subscribe", "index": "NIFTY", ...}`
   - Filtered by expiry और strikes
   - Smooth mode support (EMA/raw)

   **`WS /ws/futures`** - Live futures updates

6. **Broadcasting:**

   **`broadcast_normalized_update(index_name)`**
   - सभी subscribed clients को update push करता है
   - Filters apply करता है (expiry, strikes, smooth)
   - Disconnected clients cleanup करता है

7. **Graceful Shutdown:**

   **`on_shutdown()`**
   - Candle fetcher stop करता है
   - Pending tasks cancel करता है
   - WebSocket connections close करता है
   - Clean shutdown ensure करता है

**Performance Considerations:**
- Async/await for non-blocking I/O
- Concurrent request handling
- Memory-efficient data structures
- Lazy loading for large datasets

---

### 📡 market_fetcher.py (479 lines)

**Purpose:** Background service जो continuously Upstox से candle data fetch करती है

**Key Components:**

1. **CandleFetcher Class:**

   **Configuration:**
   ```python
   concurrency = 8        # Parallel fetches
   request_timeout = 10   # Seconds
   interval = "S10"       # 10-second candles
   limit = 2500          # Max candles per request
   ```

2. **Trading Windows:**
```python
WINDOWS = [
    (time(9, 16), time(9, 45), 30),   # Opening: 30s interval
    (time(9, 45), time(13, 45), 30),  # Mid-day: 30s interval
    (time(13, 45), time(15, 30), 30), # Closing: 30s interval
]
```
- Different intervals for different times
- Market hours: 09:15 to 15:30 IST
- Outside windows: Sleep करता है

3. **Fetch Logic:**

   **`_fetch_once(now, base_interval)`**
   - Steps:
     1. Due specs filter करो (time-based scheduling)
     2. Spot readiness check करो (futures/options के लिए)
     3. Specs sort करो (spot first priority)
     4. Concurrent fetch with semaphore (max 8 parallel)
     5. Process responses
     6. Auto-normalize after cycle completes

   **Scheduling:**
   - NIFTY: Every base_interval (30s)
   - Others: Every 3 minutes (180s)
   - Tracks `_next_fetch` per symbol

4. **Spot Dependency:**
```python
_requires_spot_reference(spec):
    return spec.category in {
        "index_future", "equity_future",
        "index_option", "equity_option"
    }
```
- Options/Futures को spot data चाहिए (SpotAverage column)
- Spot instruments pehle fetch होती हैं
- Spot ready check: `_spot_ready[label] == trading_date`

5. **Fetch Request:**

   **`_fetch_and_store(spec, from_ts, semaphore, session)`**
   - API URL: `https://service.upstox.com/chart/open/v3/candles`
   - Parameters:
     * `instrumentKey`: Upstox instrument key
     * `interval`: S10 (10 seconds)
     * `from`: Cutoff timestamp (milliseconds)
     * `limit`: Dynamic (based on last processed)
   
   - Processing:
     1. HTTP GET request
     2. Parse JSON response
     3. Transform candles (candle_processing)
     4. Store in state
     5. Run calculations

6. **Dynamic Limit:**
```python
def _limit_for_spec(spec, target_ts_ms):
    last_ms = get_last_processed_timestamp()
    if last_ms >= target_ts_ms:
        return MIN_FETCH_LIMIT  # Already up-to-date
    diff_seconds = (target_ts_ms - last_ms) / 1000
    return min(MAX_LIMIT, diff_seconds + BUFFER)
```
- Efficient: Only fetch needed candles
- Min: 10 candles
- Max: 2500 candles

7. **Auto-Normalization:**
```python
# After fetch cycle:
normalize_all_index_options()  # Options
normalize_all_futures()        # Futures

# Broadcast via WebSocket:
broadcast_normalized_update("NIFTY")
broadcast_futures_update()
```

8. **Error Handling:**
   - HTTP errors: Log और continue
   - Timeout: Configurable (10s default)
   - Retry: Implicit (next cycle में automatically retry)

**Performance Optimizations:**
- Semaphore limiting (avoid overwhelming API)
- Async concurrency (non-blocking I/O)
- Dynamic fetch limits (avoid unnecessary data)
- Conditional normalization (only after all fetches)

---

### 💾 state.py (599 lines)

**Purpose:** In-memory state management with NumPy-based circular buffers

**Key Components:**

1. **CandleBuffer Class:**

   **Structure:**
   ```python
   capacity = 3000  # ~1 day of 10s candles + buffer
   
   # NumPy arrays:
   time_seconds: np.int32[capacity]
   average: np.float32[capacity]
   volume: np.float32[capacity]
   open_interest: np.float32[capacity]
   spot: np.float32[capacity]
   iv: np.float32[capacity]
   delta: np.float32[capacity]
   # ... + 20 more columns
   
   # Metadata:
   head: int          # Current write position
   size: int          # Number of elements stored
   latest_ts_ms: int  # Latest timestamp
   trading_date: str  # Current trading date
   ```

   **Circular Buffer Logic:**
   ```python
   idx = head
   array[idx] = value
   head = (head + 1) % capacity  # Wrap around
   if size < capacity:
       size += 1
   ```
   - Efficient: No memory reallocation
   - Fixed capacity: Prevents memory leaks
   - Oldest data automatically overwritten

2. **Vectorized Append:**

   **`append_many(candles)`**
   - Batch processing approach (vs loop)
   - Steps:
     1. Parse and validate all candles
     2. Sort by timestamp
     3. Extract all values into NumPy arrays
     4. Compute cumsums vectorized
     5. Batch insert into buffer
   
   - Performance: ~10x faster than loop

3. **Cumsum Calculation:**
```python
def compute_cumsum(diff_arr, prev_cumsum):
    safe_diff = np.where(np.isfinite(diff_arr), diff_arr, 0.0)
    cumsum = np.cumsum(safe_diff) + prev_cumsum
    
    # NaN until first valid value:
    valid_count = np.cumsum(np.isfinite(diff_arr))
    cumsum = np.where(valid_count > 0, cumsum, np.nan)
    
    return cumsum
```
- Preserves leading NaN (normalization के लिए critical)
- Handles missing data gracefully

4. **Product Cumsum:**
```python
prod = diff × multiplier
prod_cumsum = cumsum(prod)  # With NaN handling
```
- Examples:
  * Average × Volume
  * TimeValue × Volume

5. **Global State:**
```python
class _InMemoryState:
    payload: dict              # Filtered instruments
    trading_catalog: list      # Symbol catalog
    candles: dict             # Recent candle records (deque)
    calculations: dict        # Calculation states
    numpy_candles: dict       # NumPy buffers per symbol

_STATE = _InMemoryState()
_LOCK = RLock()  # Thread-safe access
```

6. **Key Functions:**

   **`set_candle_record(trading_symbol, record)`**
   - Store candle record in deque (max 5)
   - Update NumPy buffer (vectorized append)
   - Publish live update (WebSocket)
   - Thread-safe with lock

   **`get_numpy_candle_snapshot(trading_symbol)`**
   - Extract ordered data from circular buffer
   - Return dict with all arrays
   - Include metadata
   - Used by: API endpoints, normalization

   **`get_calculation_state(trading_symbol)`**
   - Return: last_timestamp_ms, processed_count
   - Used by: calculations.py, market_fetcher.py

   **`update_calculation_state(trading_symbol, **fields)`**
   - Update arbitrary fields
   - Thread-safe
   - Used for: timestamps, counts, cache, normalization results

7. **Date Reset:**
```python
def reset_for_date(candle_date):
    if candle_date != self.trading_date:
        self.trading_date = candle_date
        self.head = 0
        self.size = 0
        self.latest_ts_ms = None
```
- Fresh day = fresh buffer
- Prevents mixing data from different days

**Memory Management:**
- Fixed capacity per buffer (3000 × 30 columns × 4 bytes ≈ 360 KB)
- Total for 500 instruments ≈ 180 MB (manageable)
- No dynamic allocation during runtime
- Circular buffer prevents memory leaks

---

### 📈 futures_normalization.py (521 lines)

**Purpose:** Futures contracts का normalization (same approach as options)

**Key Differences from Options:**

1. **Simpler Structure:**
   - Only 2 metrics: `fut_spot_diff_cumsum`, `oi_diff_cumsum`
   - No Greeks (no IV, Delta, Vega, Theta)

2. **Futures List:**
```python
INDEX_FUTURES = ["NIFTY", "BANKNIFTY"]
EQUITY_FUTURES = ["RELIANCE", "TCS", "HDFCBANK", ...]
ALL_FUTURES = INDEX_FUTURES + EQUITY_FUTURES
```

3. **Normalization:**
   - **FutSpotDiff:** EMA smooth → IQR normalize
   - **OI:** Z-score normalize
   
   ```python
   fut_spot_smoothed = _calculate_ema(fut_spot_diff, period=12)
   normalized = (smoothed - median) / IQR
   ```

4. **Cache Structure:**
   - State key: `"FUTURES_COMBINED"`
   - Results: `{"NIFTY": {...}, "RELIANCE": {...}, ...}`

5. **Metadata API:**
```python
get_futures_metadata() -> {
    "available_futures": [...],
    "time_seconds": [...],
    "index_futures": [...],
    "equity_futures": [...]
}
```

**Usage:**
```python
# Normalize all:
normalized = normalize_all_futures()

# Get cached:
normalized = get_futures_normalized_data()

# Get metadata only:
metadata = get_futures_metadata()
```

---

## Data Flow Diagram (डेटा फ्लो)

```
┌─────────────────────────────────────────────────────────────┐
│ STARTUP                                                      │
│ 1. Download instruments (download_extract.py)               │
│ 2. Wait for 09:16 IST (fast_api.py)                        │
│ 3. Fetch spot prices (extract.py)                          │
│ 4. Build payload (extract.py)                              │
│ 5. Start candle fetcher (market_fetcher.py)                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ CONTINUOUS LOOP (market_fetcher.py)                         │
│ Every 30s during market hours:                              │
│                                                              │
│ 1. Fetch candles (Upstox API)                              │
│ 2. Transform (candle_processing.py)                        │
│    - Calculate Greeks (Black-76)                            │
│    - Compute diff/cumsum features                           │
│ 3. Store (state.py - NumPy buffers)                        │
│ 4. Calculate (calculations.py)                             │
│    - Track processed timestamps                             │
│ 5. Normalize (combined_normalization.py)                   │
│    - IQR/Z-score across all options                        │
│    - Cache results                                          │
│ 6. Broadcast (fast_api.py WebSockets)                      │
│    - Push to subscribed clients                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ CLIENT ACCESS                                                │
│                                                              │
│ REST APIs:                                                   │
│ - GET /api/normalized/{index}?strikes=...&smooth=true      │
│ - GET /api/futures/normalized?smooth=true                  │
│ - GET /candles/{symbol}                                     │
│                                                              │
│ WebSocket:                                                   │
│ - WS /ws/normalized (live options updates)                 │
│ - WS /ws/futures (live futures updates)                    │
│ - WS /ws/candles/{symbol} (live candle updates)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Algorithms (मुख्य अल्गोरिदम)

### 1. Black-76 Implied Volatility (Bisection Method)

```python
# Objective: Find σ such that Black76(σ) = market_price
sigma_low = 0.001
sigma_high = 5.0

for i in range(100):  # Max iterations
    sigma_mid = (sigma_low + sigma_high) / 2
    price_mid = Black76(F, K, T, r, sigma_mid)
    
    if price_mid > market_price:
        sigma_high = sigma_mid  # σ too high
    else:
        sigma_low = sigma_mid   # σ too low
    
    if abs(price_mid - market_price) < tolerance:
        break

return sigma_mid
```

### 2. IQR Normalization (Robust to Outliers)

```python
# For each time point t:
history = data[0:t+1]  # Expanding window
median = np.median(history)
q1 = np.percentile(history, 25)
q3 = np.percentile(history, 75)
iqr = q3 - q1

# Dynamic floor prevents division by zero:
floor = max(0.01, abs(median) * 0.1)

normalized[t] = (data[t] - median) / max(iqr, floor)
```

### 3. Cumsum with NaN Handling

```python
# Prevent leading zeros from polluting normalization:
safe_diff = np.where(np.isfinite(diff), diff, 0.0)
raw_cumsum = np.cumsum(safe_diff)

# Track where we've seen valid data:
valid_seen = np.cumsum(np.isfinite(diff)) > 0

# NaN until first valid value:
cumsum = np.where(valid_seen, raw_cumsum, np.nan)
```

### 4. EMA Smoothing

```python
multiplier = 2 / (period + 1)
ema[0] = data[0]

for i in range(1, n):
    ema[i] = data[i] * multiplier + ema[i-1] * (1 - multiplier)
```

---

## Performance Tips (परफॉर्मेंस टिप्स)

1. **Normalization:**
   - First run: Pandas expanding (~200ms for 2000 rows)
   - Incremental: NumPy loop (~20ms for 50 new rows)
   - Cache results in calculation_state

2. **NumPy Buffers:**
   - Pre-allocated arrays (no reallocation)
   - Vectorized operations (10x faster than loops)
   - Circular buffer (memory efficient)

3. **API Responses:**
   - Lazy loading (metadata first, data on demand)
   - Streaming (chunk-based processing)
   - Compression (gzip for large responses)

4. **WebSocket:**
   - Filter before send (don't send unwanted data)
   - Batch updates (combine multiple changes)
   - Disconnect cleanup (prevent memory leaks)

---

## Common Issues & Solutions (सामान्य समस्याएं और समाधान)

### Issue: Weekend/Holiday Server Start
```bash
# Solution: Use faketime for debugging
faketime '2025-12-27 09:16:00' python fast_api.py
```

### Issue: No Spot Data for Options
```python
# Check: _spot_ready[label] == trading_date
# Solution: Wait for spot fetch first, or check _is_spot_ready()
```

### Issue: Normalization Taking Too Long
```python
# Check cache version:
calc_state = get_calculation_state(f"{index}_COMBINED")
if calc_state.get("cache_version") < CURRENT_VERSION:
    # Cache invalidated, will recalculate
    pass
```

### Issue: Memory Usage Growing
```python
# Check buffer sizes:
for symbol, buffer in _STATE.numpy_candles.items():
    print(f"{symbol}: size={buffer.size}, capacity={buffer.capacity}")

# If size > capacity, circular buffer wrapping correctly
```

---

## Testing Checklist (टेस्टिंग चेकलिस्ट)

- [ ] Server starts successfully on weekday 09:16+
- [ ] Instruments downloaded and extracted
- [ ] Spot prices fetched from API
- [ ] Candles fetching continuously (30s interval)
- [ ] Options Greeks calculated correctly
- [ ] Normalization running after each cycle
- [ ] WebSocket connections working
- [ ] Real-time updates broadcasting
- [ ] Memory stable (no leaks)
- [ ] CPU usage reasonable (<50% average)
- [ ] API response times <100ms for cached data
- [ ] Dashboard loading and displaying charts

---

## Deployment Notes (डिप्लॉयमेंट नोट्स)

1. **Server Requirements:**
   - Python 3.10+
   - 2GB RAM minimum (4GB recommended)
   - CPU: 2 cores minimum (4 recommended)
   - Network: Stable internet for Upstox API

2. **Environment Variables:**
   ```bash
   # Optional - if using .env
   UPSTOX_API_KEY=your_key_here
   LOG_LEVEL=INFO
   ```

3. **Startup Command:**
   ```bash
   # Production
   uvicorn fast_api:app --host 0.0.0.0 --port 8000 --workers 1

   # Development
   python fast_api.py  # Runs on 127.0.0.1:8000
   ```

4. **Monitoring:**
   - Check `/health` endpoint regularly
   - Monitor log files for errors
   - Track memory usage (should stabilize)
   - Watch API latency (should be <100ms)

---

## Contributing (योगदान)

यदि आप इस project में contribute करना चाहते हैं:

1. Code comments add करें (bilingual preferred)
2. Documentation update करें if logic changes
3. Performance optimizations suggest करें
4. Edge cases handle करें
5. Tests add करें (if possible)

---

## Contact & Support

For questions or issues, please:
- Check this documentation first
- Review code comments in respective files
- Check GitHub issues

---

**Document Version:** 1.0
**Last Updated:** 2025-01-01
**Maintainer:** SuSe_code Team
