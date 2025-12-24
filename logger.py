#!/usr/bin/env python3

from datetime import datetime, timezone


def _timestamp():
    return datetime.now(tz=timezone.utc).isoformat()


def log_info(message: str):
    print(f"[INFO { _timestamp()}] {message}")


def log_error(message: str):
    print(f"[ERROR { _timestamp()}] {message}")
