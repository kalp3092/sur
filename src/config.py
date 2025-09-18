"""Configuration for the project.

This module prefers using pydantic BaseSettings (v1 or v2 via pydantic-settings)
but falls back to a small dataclass-based Settings object that reads from
environment variables when the pydantic settings package isn't available.
"""

from typing import Optional
import os


# Try to use pydantic BaseSettings when available (pydantic v1 or with
# pydantic-settings installed). Fall back to a simple dataclass reading env
# variables when not available.
try:
    from pydantic import BaseSettings, Field


    class Settings(BaseSettings):
        # Video source: 0 (webcam), path, or RTSP URL
        video_source: str = Field("0", description="Video capture source")

        # YOLO model path or name (ultralytics)
        yolo_model: str = Field("shoplifting_weights.pt", description="YOLOv8 model path or name")

        # Behavior model path
        behavior_model_path: Optional[str] = Field(None, description="Path to 3D CNN weights")

        # Confidence thresholds
        detection_confidence: float = Field(0.4, ge=0.0, le=1.0)
        alert_confidence: float = Field(0.7, ge=0.0, le=1.0)

        # Clip parameters for 3D CNN
        clip_len: int = Field(16, description="Number of frames per clip for behavior analysis")
        clip_fps: int = Field(8, description="Target FPS for clip sampling")

        # Server
        host: str = Field("0.0.0.0")
        port: int = Field(8000)

        class Config:
            env_prefix = "SD_"

    def get_settings() -> Settings:
        return Settings()

except Exception:
    # pydantic BaseSettings not available; use a minimal fallback that reads
    # configuration from environment variables with sane defaults.
    from dataclasses import dataclass

    @dataclass
    class Settings:
        video_source: str = os.getenv("SD_VIDEO_SOURCE", "0")
        yolo_model: str = os.getenv("SD_YOLO_MODEL", "yolov8n.pt")
        behavior_model_path: Optional[str] = os.getenv("SD_BEHAVIOR_MODEL_PATH")
        detection_confidence: float = float(os.getenv("SD_DETECTION_CONFIDENCE", "0.4"))
        alert_confidence: float = float(os.getenv("SD_ALERT_CONFIDENCE", "0.7"))
        clip_len: int = int(os.getenv("SD_CLIP_LEN", "16"))
        clip_fps: int = int(os.getenv("SD_CLIP_FPS", "8"))
        host: str = os.getenv("SD_HOST", "0.0.0.0")
        port: int = int(os.getenv("SD_PORT", "8000"))

    def get_settings() -> Settings:
        return Settings()
