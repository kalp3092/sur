"""YOLODetector: wraps Ultralytics YOLOv8 for person detection."""

from typing import List, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class Detection:
    def __init__(self, xyxy: Tuple[int, int, int, int], conf: float, cls: int):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = int(cls)


class YOLODetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.4):
        if YOLO is None:
            raise RuntimeError("ultralytics package is required. Install via pip install ultralytics")
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single frame and return person detections.

        Returns detections with bbox xyxy, confidence, and class id.
        """
        results = self.model.predict(source=frame, imgsz=640, conf=self.conf_thresh, verbose=False)
        detections: List[Detection] = []
        for r in results:
            if hasattr(r, 'boxes'):
                for b in r.boxes:
                    cls = int(b.cls.cpu().numpy()[0]) if hasattr(b, 'cls') else 0
                    conf = float(b.conf.cpu().numpy()[0]) if hasattr(b, 'conf') else 0.0
                    xyxy = tuple(map(int, b.xyxy.cpu().numpy()[0])) if hasattr(b, 'xyxy') else (0,0,0,0)
                    # Typically class 0 is person in COCO
                    detections.append(Detection(xyxy=xyxy, conf=conf, cls=cls))
        return detections
