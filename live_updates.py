#!/usr/bin/env python3
# Shebang line - Python 3 se script run karne ke liye

"""
LIVE UPDATES MODULE (लाइव अपडेट मॉड्यूल)
==================================

Purpose: Lightweight publish-subscribe (pub/sub) system for real-time candle updates
यह module asyncio queues का use करके WebSocket clients को real-time candle data push करता है

Architecture:
- Publisher: market_fetcher जब नया candle data आता है तो publish() करता है
- Subscribers: WebSocket clients subscribe() करके live updates receive करते हैं
- Each trading symbol (जैसे NIFTY, BANKNIFTY) की अपनी queue होती है
"""

import asyncio  # Async operations के लिए
from collections import defaultdict  # Default values वाले dict के लिए
from typing import Dict  # Type hints के लिए

# GLOBAL STATE VARIABLES
# ये variables सभी trading symbols के लिए queues और subscriber counts track करते हैं

# Queues keyed by trading symbol - हर symbol की अपनी queue
# Example: {"NIFTY": Queue(), "BANKNIFTY": Queue()}
_QUEUES: Dict[str, asyncio.Queue] = {}

# Subscriber counts per symbol - हर symbol के लिए कितने clients subscribed हैं
# Example: {"NIFTY": 5, "BANKNIFTY": 3} means 5 clients subscribed to NIFTY
_COUNTS: Dict[str, int] = defaultdict(int)

# Lock for thread-safe access - concurrent access को safe बनाने के लिए
_LOCK = asyncio.Lock()


async def subscribe(symbol: str) -> asyncio.Queue:
    """
    Trading symbol के लिए subscribe करो और asyncio.Queue return करो
    
    Purpose: WebSocket client को specific symbol (जैसे NIFTY) की live updates receive करने के लिए queue provide करता है
    
    Args:
        symbol: Trading symbol (e.g., "NIFTY", "BANKNIFTY", "RELIANCE")
    
    Returns:
        asyncio.Queue: Queue जिसमें real-time candle updates push होंगे
    
    Process:
        1. Symbol को uppercase में convert करो (consistency के लिए)
        2. Lock acquire करो (thread-safe access के लिए)
        3. अगर queue exist नहीं करती तो नई queue create करो
        4. Subscriber count increment करो
        5. Queue return करो
    
    Example:
        queue = await subscribe("NIFTY")
        # अब इस queue से data read करो: record = await queue.get()
    """
    symbol = symbol.upper()  # Symbol को uppercase में convert करो (NIFTY, nifty -> NIFTY)
    async with _LOCK:  # Thread-safe access के लिए lock acquire करो
        queue = _QUEUES.get(symbol)  # Check करो कि symbol की queue already exist करती है
        if queue is None:  # अगर queue नहीं है
            queue = asyncio.Queue()  # नई queue create करो
            _QUEUES[symbol] = queue  # Global dict में store करो
        _COUNTS[symbol] += 1  # Subscriber count increment करो (एक और client subscribed हो गया)
        return queue  # Queue return करो जो client use करेगा data receive करने के लिए


async def unsubscribe(symbol: str) -> None:
    """
    Trading symbol के लिए unsubscribe करो (जब client disconnect हो जाए)
    
    Purpose: जब WebSocket client disconnect करता है तो subscriber count decrease करना और unused queues को cleanup करना
    
    Args:
        symbol: Trading symbol जिससे unsubscribe करना है
    
    Process:
        1. Symbol को uppercase में convert करो
        2. Subscriber count decrement करो
        3. अगर कोई subscribers नहीं बचे (count = 0) तो queue delete कर दो (memory save करने के लिए)
    
    Example:
        await unsubscribe("NIFTY")
        # अब NIFTY के लिए कम subscribers हैं, या queue delete हो गई अगर कोई subscribers नहीं
    """
    symbol = symbol.upper()  # Symbol को uppercase में convert करो
    async with _LOCK:  # Thread-safe access के लिए lock acquire करो
        if symbol not in _COUNTS:  # अगर इस symbol के लिए कोई subscribers नहीं हैं
            return  # कुछ नहीं करना, simply return करो
        _COUNTS[symbol] = max(0, _COUNTS[symbol] - 1)  # Count decrease करो, minimum 0 (negative नहीं होना चाहिए)
        if _COUNTS[symbol] == 0:  # अगर अब कोई subscribers नहीं बचे
            _COUNTS.pop(symbol, None)  # Count dictionary से remove करो
            _QUEUES.pop(symbol, None)  # Queue dictionary से remove करो (memory cleanup)


def publish(record: dict) -> None:
    """
    Candle record को सभी subscribers को publish करो (broadcast करो)
    
    Purpose: जब market_fetcher नया candle data fetch करता है, तो सभी subscribed WebSocket clients को immediately push करो
    
    Args:
        record: Candle data record (dict) जिसमें trading_symbol होना चाहिए
                Example: {"trading_symbol": "NIFTY", "candles": [...], ...}
    
    Process:
        1. Record से trading_symbol extract करो
        2. Running event loop get करो (asyncio के लिए)
        3. Symbol की queue में record को put करो (non-blocking way में)
        4. अगर कोई queue नहीं है तो silently ignore करो
    
    Note: यह function non-blocking है - इसे synchronous code से भी safely call कर सकते हैं
    
    Example:
        record = {"trading_symbol": "NIFTY", "candles": [...]}
        publish(record)
        # अब सभी NIFTY subscribers को यह record automatically receive हो जाएगा
    """
    # Record से trading_symbol extract करो और uppercase में convert करो
    symbol = str(record.get("trading_symbol") or "").upper()
    if not symbol:  # अगर symbol नहीं मिला
        return  # कुछ नहीं करना, simply return करो

    # Running event loop get करने की कोशिश करो
    try:
        loop = asyncio.get_running_loop()  # Currently running async loop get करो
    except RuntimeError:  # अगर कोई running loop नहीं है
        try:
            loop = asyncio.get_event_loop()  # Default event loop get करो
        except RuntimeError:  # अगर event loop create नहीं हो सका
            return  # Publish नहीं कर सकते, return करो

    def _put():
        """
        Internal helper function जो actually queue में data put करता है
        
        Purpose: Queue में record को add करना (non-blocking way)
        """
        queue = _QUEUES.get(symbol)  # Symbol की queue get करो
        if queue:  # अगर queue exist करती है (कोई subscribers हैं)
            queue.put_nowait(record)  # Queue में record add करो (without blocking)

    # Event loop में _put function को schedule करो
    if loop.is_running():  # अगर loop already running है
        loop.call_soon_threadsafe(_put)  # Thread-safe way में _put को schedule करो
    else:  # अगर loop running नहीं है
        _put()  # Directly _put को call करो


# Export किए जाने वाले public functions
__all__ = ["subscribe", "unsubscribe", "publish"]
