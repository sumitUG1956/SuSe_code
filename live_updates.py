#!/usr/bin/env python3

"""
Live Updates Module - Publish-Subscribe System for Real-Time Candle Data
Uses asyncio queues to push candle updates to WebSocket clients in real-time
"""

import asyncio  # Async/await support for concurrent operations
from collections import defaultdict  # Dict with default values
from typing import Dict  # Type hints

# Global state variables for managing subscriptions
# Maps trading symbol to its asyncio.Queue for pub/sub
_QUEUES: Dict[str, asyncio.Queue] = {}
# Maps trading symbol to count of active subscribers
_COUNTS: Dict[str, int] = defaultdict(int)
# Lock for thread-safe access to global state
_LOCK = asyncio.Lock()


async def subscribe(symbol: str) -> asyncio.Queue:
    """
    Subscribe to real-time updates for a trading symbol
    
    Args:
        symbol: Trading symbol to subscribe to (e.g., "NIFTY", "BANKNIFTY")
    
    Returns:
        asyncio.Queue: Queue that will receive real-time candle updates
    
    Process:
        1. Convert symbol to uppercase for consistency
        2. Acquire lock for thread-safe access
        3. Create queue if it doesn't exist
        4. Increment subscriber count
        5. Return the queue for receiving updates
    
    Usage:
        queue = await subscribe("NIFTY")
        record = await queue.get()  # Wait for and receive next update
    """
    symbol = symbol.upper()  # Normalize to uppercase
    async with _LOCK:  # Thread-safe access to global state
        queue = _QUEUES.get(symbol)  # Check if queue exists for this symbol
        if queue is None:  # If queue doesn't exist
            queue = asyncio.Queue()  # Create new queue
            _QUEUES[symbol] = queue  # Store in global dict
        _COUNTS[symbol] += 1  # Increment subscriber count
        return queue  # Return queue for receiving updates


async def unsubscribe(symbol: str) -> None:
    """
    Unsubscribe from updates for a trading symbol
    
    Args:
        symbol: Trading symbol to unsubscribe from
    
    Process:
        1. Convert symbol to uppercase
        2. Acquire lock for thread-safe access
        3. Decrement subscriber count
        4. If no subscribers remain, clean up queue to free memory
    
    Usage:
        await unsubscribe("NIFTY")  # Clean up subscription
    """
    symbol = symbol.upper()  # Normalize to uppercase
    async with _LOCK:  # Thread-safe access
        if symbol not in _COUNTS:  # If no subscribers for this symbol
            return  # Nothing to do
        _COUNTS[symbol] = max(0, _COUNTS[symbol] - 1)  # Decrement count (min 0)
        if _COUNTS[symbol] == 0:  # If no subscribers remain
            _COUNTS.pop(symbol, None)  # Remove from counts dict
            _QUEUES.pop(symbol, None)  # Remove queue to free memory


def publish(record: dict) -> None:
    """
    Publish a candle record to all subscribers of its trading symbol
    
    Args:
        record: Candle data record (dict) containing at minimum "trading_symbol"
    
    Process:
        1. Extract trading symbol from record
        2. Get running event loop
        3. Schedule non-blocking put to all subscriber queues
        4. Handle both running and non-running loop scenarios
    
    Usage:
        record = {"trading_symbol": "NIFTY", "candles": [...], ...}
        publish(record)  # Broadcasts to all NIFTY subscribers
    
    Note: Non-blocking operation - safe to call from synchronous code
    """
    # Extract and normalize trading symbol
    symbol = str(record.get("trading_symbol") or "").upper()
    if not symbol:  # If no symbol in record
        return  # Skip publishing

    # Get the running event loop
    try:
        loop = asyncio.get_running_loop()  # Try to get currently running loop
    except RuntimeError:  # No running loop
        try:
            loop = asyncio.get_event_loop()  # Try to get default loop
        except RuntimeError:  # Can't get any loop
            return  # Can't publish without event loop

    def _put():
        """Internal helper to put record in queue"""
        queue = _QUEUES.get(symbol)  # Get queue for this symbol
        if queue:  # If queue exists (has subscribers)
            queue.put_nowait(record)  # Add record to queue (non-blocking)

    # Schedule the put operation
    if loop.is_running():  # If loop is already running
        loop.call_soon_threadsafe(_put)  # Schedule in thread-safe manner
    else:  # If loop is not running
        _put()  # Call directly


# Export public functions
__all__ = ["subscribe", "unsubscribe", "publish"]
