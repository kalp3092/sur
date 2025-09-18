"""Integration of YOLO detection and 3D CNN behavior analysis.

Exposes DetectorSystem which runs detection loop in a thread and calls alert callbacks.
"""
import threading
import time
from typing import Callable, List, Optional
import numpy as np

from .yolo_detector import YOLODetector, Detection
from .behavior_model import BehaviorModel
from .video_pipeline import VideoCaptureThread, extract_clip
from .config import get_settings
from .utils import logger


class DetectorSystem:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.vcap = VideoCaptureThread(self.settings.video_source)
        self.detector = YOLODetector(model_name=self.settings.yolo_model, conf_thresh=self.settings.detection_confidence)
        self.behavior = BehaviorModel(model_path=self.settings.behavior_model_path)
        self.alert_callbacks: List[Callable[[dict], None]] = []
        self.thread: Optional[threading.Thread] = None
        self.running = False

    def start(self):
        self.vcap.start()
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.vcap.stop()

    def register_alert_callback(self, cb: Callable[[dict], None]):
        self.alert_callbacks.append(cb)

    def _notify(self, alert: dict):
        for cb in self.alert_callbacks:
            try:
                cb(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def _run_loop(self):
        logger.info("Detector system started")
        while self.running:
            latest = self.vcap.read_buffer()
            if not latest:
                time.sleep(0.05)
                continue
            # use most recent frame for detection
            ts, frame = latest[-1]
            try:
                dets = self.detector.detect(frame)
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")
                time.sleep(0.1)
                continue

            # Filter person class (COCO class 0 typical)
            persons = [d for d in dets if d.cls == 0 and d.conf >= self.settings.detection_confidence]
            if not persons:
                time.sleep(0.02)
                continue

            # For simplicity, analyze the clip centered on latest frame
            clip = extract_clip(self.vcap.buffer, len(self.vcap.buffer)-1, self.settings.clip_len)
            res = self.behavior.predict(clip)
            if res["label"] == "shoplifting" and res["confidence"] >= self.settings.alert_confidence:
                alert = {
                    "time": time.time(),
                    "type": "shoplifting",
                    "confidence": res["confidence"],
                    "frame_ts": ts,
                }
                logger.warning(f"Alert generated: {alert}")
                self._notify(alert)

            time.sleep(0.02)
