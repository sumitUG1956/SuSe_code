#!/usr/bin/env python3

"""
Lightweight pub/sub for candle updates using asyncio queues.
Used by WebSocket endpoint to push new records to clients.
"""

import asyncio
from collections import defaultdict
from typing import Dict

# Queues keyed by trading symbol
_QUEUES: Dict[str, asyncio.Queue] = {}
_COUNTS: Dict[str, int] = defaultdict(int)
_LOCK = asyncio.Lock()


async def subscribe(symbol: str) -> asyncio.Queue:
    """Return an asyncio.Queue for the symbol and bump subscriber count."""
    symbol = symbol.upper()
    async with _LOCK:
        queue = _QUEUES.get(symbol)
        if queue is None:
            queue = asyncio.Queue()
            _QUEUES[symbol] = queue
        _COUNTS[symbol] += 1
        return queue


async def unsubscribe(symbol: str) -> None:
    """Decrement subscriber count and clean up queue if unused."""
    symbol = symbol.upper()
    async with _LOCK:
        if symbol not in _COUNTS:
            return
        _COUNTS[symbol] = max(0, _COUNTS[symbol] - 1)
        if _COUNTS[symbol] == 0:
            _COUNTS.pop(symbol, None)
            _QUEUES.pop(symbol, None)


def publish(record: dict) -> None:
    """
    Push a record to all subscribers of its trading symbol.

    Non-blocking: schedules a put_nowait on the running event loop.
    """
    symbol = str(record.get("trading_symbol") or "").upper()
    if not symbol:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return

    def _put():
        queue = _QUEUES.get(symbol)
        if queue:
            queue.put_nowait(record)

    if loop.is_running():
        loop.call_soon_threadsafe(_put)
    else:
        _put()


__all__ = ["subscribe", "unsubscribe", "publish"]
