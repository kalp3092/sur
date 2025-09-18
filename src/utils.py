"""Utility functions: logging, alert queue, and simple IO helpers."""

from loguru import logger as _logger
from typing import List
import threading


# Configure loguru basic
_logger.remove()
_logger.add(lambda msg: print(msg, end=''), level="INFO")

logger = _logger


class AlertQueue:
    def __init__(self):
        self.lock = threading.Lock()
        self.queue: List[dict] = []

    def push(self, alert: dict):
        with self.lock:
            self.queue.append(alert)

    def pop_all(self):
        with self.lock:
            items = list(self.queue)
            self.queue.clear()
        return items
