#!/usr/bin/env python3
# Shebang line to run this script with Python 3

"""
Logger Module - Simple logging utility for application-wide logging
Provides timestamp-prefixed INFO and ERROR level logging to stdout
"""

from datetime import datetime, timezone  # For timestamp generation


def _timestamp():
    """
    Generate current UTC timestamp in ISO 8601 format
    
    Returns:
        str: ISO formatted timestamp (e.g., "2024-01-01T12:00:00+00:00")
    
    Usage: Internal helper for log functions to add consistent timestamps
    """
    return datetime.now(tz=timezone.utc).isoformat()  # Get current time in UTC and convert to ISO string


def log_info(message: str):
    """
    Log an informational message with timestamp
    
    Args:
        message: The informational message to log
    
    Output format: [INFO <timestamp>] <message>
    Example: [INFO 2024-01-01T12:00:00+00:00] Server started successfully
    """
    print(f"[INFO { _timestamp()}] {message}")  # Print with INFO prefix and timestamp


def log_error(message: str):
    """
    Log an error message with timestamp
    
    Args:
        message: The error message to log
    
    Output format: [ERROR <timestamp>] <message>
    Example: [ERROR 2024-01-01T12:00:00+00:00] Failed to fetch data
    """
    print(f"[ERROR { _timestamp()}] {message}")  # Print with ERROR prefix and timestamp
