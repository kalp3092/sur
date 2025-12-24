from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any, Iterable
from urllib.parse import urlsplit, urlunsplit
import cv2
import time
import logging
import socket
import os

logger = logging.getLogger(__name__)


def _mask_url(u: str) -> str:
    try:
        p = urlsplit(str(u))
        net = p.netloc
        if "@" in net:
            net = f"***:***@{net.split('@', 1)[1]}"
        return f"{p.scheme}://{net}{p.path}{('?' + p.query) if p.query else ''}"
    except Exception:
        return str(u)


def _tcp_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_url_with_credentials(url: str, username: Optional[str], password: Optional[str]) -> str:
    """Inject credentials into RTSP/HTTP URL if provided; strip fragments like #live."""
    if not url:
        logger.debug("build_url_with_credentials: empty url")
        return url

    parts = urlsplit(url)
    scheme, netloc, path, query, _fragment = parts

    # If creds already present, just strip fragment
    if "@" in netloc:
        logger.debug("build_url_with_credentials: creds already present -> %s", _mask_url(url))
        return urlunsplit((scheme, netloc, path, query, ""))

    username = (username or "").strip()
    password = (password or "").strip()
    if not username or not password:
        logger.debug("build_url_with_credentials: missing username/password -> leaving url unchanged")
        return urlunsplit((scheme, netloc, path, query, ""))

    new_netloc = f"{username}:{password}@{netloc}"
    masked = f"{scheme}://***:***@{netloc}{path}{('?' + query) if query else ''}"
    logger.info("build_url_with_credentials: injected credentials -> %s", masked)
    return urlunsplit((scheme, new_netloc, path, query, ""))


def test_camera(source, attempts: int = 30, delay: float = 0.1) -> Tuple[bool, Optional[float], Optional[Tuple[int, int]], Optional[bytes], Optional[str]]:
    """Open a camera/stream and try to read one frame.

    Returns: (ok, fps, (w,h), jpeg_bytes, error)
    """
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|max_delay;5000000|stimeout;30000000",
    )

    masked = _mask_url(str(source))
    logger.info("test_camera: opening source=%s", masked)

    # RTSP preflight: reachability and basic path sanity
    try:
        parts = urlsplit(str(source))
        if parts.scheme.lower() == "rtsp":
            host = parts.hostname
            port = parts.port or 554
            if host and not _tcp_reachable(host, port, timeout=3.0):
                msg = f"RTSP TCP port not reachable ({host}:{port})"
                logger.error("test_camera: %s", msg)
                return False, None, None, None, msg
            if not parts.path or parts.path == "/":
                logger.warning(
                    "test_camera: RTSP path is empty; add the stream path (e.g., /Streaming/Channels/101)"
                )
    except Exception as e:
        logger.debug("test_camera: preflight parse error: %s", e)

    cap = None
    try:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        # Hint to reduce latency on some backends
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not cap.isOpened():
            logger.error("test_camera: failed to open source")
            return False, None, None, None, "Failed to open source"

        fps = cap.get(cv2.CAP_PROP_FPS) or None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        ok = False
        jpeg_bytes = None
        err = None
        for i in range(attempts):
            ret, frame = cap.read()
            if ret and frame is not None:
                enc_ok, buf = cv2.imencode(".jpg", frame)
                if enc_ok:
                    jpeg_bytes = buf.tobytes()
                ok = True
                logger.info(
                    "test_camera: first frame received (fps=%s, size=%sx%s, attempt=%s)", fps, w, h, i + 1
                )
                break
            time.sleep(delay)

        if not ok:
            err = "No frames received"
            logger.error("test_camera: %s", err)

        size = (w, h) if (w and h) else None
        return ok, fps, size, jpeg_bytes, err
    finally:
        if cap is not None:
            cap.release()
