"""Threaded video capture and clip extraction utilities."""

import cv2
import threading
import time
from collections import deque
from typing import Deque, Tuple, Optional, List
import numpy as np


class VideoCaptureThread:
    def __init__(self, source=0, max_buffer=256):
        self.source = int(source) if str(source).isdigit() else source
        self.cap = cv2.VideoCapture(self.source)
        self.stopped = False
        self.lock = threading.Lock()
        self.buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=max_buffer)
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if not self.thread:
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            ts = time.time()
            with self.lock:
                self.buffer.append((ts, frame.copy()))

    def read_buffer(self) -> List[Tuple[float, np.ndarray]]:
        with self.lock:
            return list(self.buffer)

    def get_latest(self) -> Optional[Tuple[float, np.ndarray]]:
        with self.lock:
            if len(self.buffer) == 0:
                return None
            return self.buffer[-1]

    def stop(self):
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass


def extract_clip(buffer: Deque[Tuple[float, np.ndarray]], center_idx: int, clip_len: int) -> np.ndarray:
    """Extract a temporal clip of length clip_len centered at center_idx from the buffer list.

    buffer: list-like of (ts, frame)
    center_idx: index in buffer to center on
    Returns numpy array (T, H, W, C)
    """
    buf = list(buffer)
    n = len(buf)
    if n == 0:
        return np.zeros((clip_len, 224, 224, 3), dtype=np.uint8)
    half = clip_len // 2
    start = max(0, center_idx - half)
    end = min(n, start + clip_len)
    clip_frames = [f for (_, f) in buf[start:end]]
    # pad if needed
    while len(clip_frames) < clip_len:
        clip_frames.insert(0, clip_frames[0].copy())
    # Resize to 224x224
    clip_resized = [cv2.resize(f, (224, 224)) for f in clip_frames]
    return np.stack(clip_resized, axis=0)
