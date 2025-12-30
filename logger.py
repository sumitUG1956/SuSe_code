#!/usr/bin/env python3
# Shebang line - यह script को Python 3 के साथ run करने के लिए है

# LOGGER MODULE (लॉगर मॉड्यूल)
# Purpose: Simple logging utility for info और error messages के साथ timestamps
# यह module application में log messages को print करने के लिए use होता है

from datetime import datetime, timezone  # Date/time handling के लिए import


def _timestamp():
    """
    Private function जो current UTC timestamp को ISO format में return करता है
    
    Returns:
        str: ISO formatted timestamp (e.g., "2024-01-01T12:00:00+00:00")
    
    Purpose: सभी log messages में consistent timestamp add करने के लिए
    """
    return datetime.now(tz=timezone.utc).isoformat()  # Current time को UTC में ISO format में convert करो


def log_info(message: str):
    """
    Info level message को log करो (informational messages के लिए)
    
    Args:
        message: Log करने वाला informational message
    
    Usage: log_info("Server started successfully")
    Output: [INFO 2024-01-01T12:00:00+00:00] Server started successfully
    """
    print(f"[INFO { _timestamp()}] {message}")  # INFO prefix के साथ message को timestamp के साथ print करो


def log_error(message: str):
    """
    Error level message को log करो (error reporting के लिए)
    
    Args:
        message: Log करने वाला error message
    
    Usage: log_error("Failed to fetch data")
    Output: [ERROR 2024-01-01T12:00:00+00:00] Failed to fetch data
    """
    print(f"[ERROR { _timestamp()}] {message}")  # ERROR prefix के साथ message को timestamp के साथ print करो
