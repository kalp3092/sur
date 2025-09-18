# Shoplifting Detection MVP

This repository contains a minimal, modular shoplifting detection MVP using YOLOv8 for person detection and a small 3D CNN for short-clip behavior classification. It includes a FastAPI web dashboard that receives real-time alerts over WebSocket.

Requirements
- Python 3.9+
- GPU recommended for real-time performance

Setup
1. Create and activate a virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

Running
- Start the app:

  python main.py

Open http://localhost:8000 to view the simple dashboard and receive WebSocket alerts.

Project layout
- `src/` - main package
  - `config.py` - Pydantic settings
  - `yolo_detector.py` - Ultralytics YOLOv8 wrapper
  - `behavior_model.py` - PyTorch 3D CNN
  - `video_pipeline.py` - threaded capture and clip extraction
  - `pipeline.py` - integration and alerting
  - `api.py` - FastAPI app and WebSocket
  - `utils.py` - logging and alert queue
- `tests/` - pytest tests

Notes & Next steps
- Replace the placeholder 3D CNN weights with a trained model.
- Add tracking (e.g., ByteTrack) to associate clips with person IDs.
- Harden error handling and performance optimizations for production.
