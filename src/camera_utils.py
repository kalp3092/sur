from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any, Iterable
from urllib.parse import urlsplit, urlunsplit, unquote
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

    # Decode percent-encoded characters (e.g. %40 -> @) so urlsplit can
    # correctly identify embedded credentials in the netloc.
    url = unquote(url)

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


# ----------------- Discovery helpers -----------------

BRAND_TEMPLATES: Dict[str, List[str]] = {
    "Unknown/Generic": [
        "/Streaming/Channels/101",
        "/Streaming/Channels/102",
        "/h264Preview_01_main",
        "/h264Preview_01_sub",
        "/cam/realmonitor?channel=1&subtype=0",
        "/cam/realmonitor?channel=1&subtype=1",
        "/axis-media/media.amp",
        "/live.sdp",
        "/media/video1",
    ],
    "Hikvision": ["/Streaming/Channels/101", "/Streaming/Channels/102"],
    "Dahua/Amcrest": ["/cam/realmonitor?channel=1&subtype=0", "/cam/realmonitor?channel=1&subtype=1"],
    "Reolink": ["/h264Preview_01_main", "/h264Preview_01_sub"],
    "Axis": ["/axis-media/media.amp"],
    "Uniview": ["/media/video1", "/unicast/c1/s0/live"],
    "Hanwha/Wisenet": ["/profile2/media.smp", "/profile1/media.smp"],
    "Unifi Protect": ["/"],
}

DEFAULT_PORTS: List[int] = [554, 8554, 10554, 5544, 7447]


def _build_rtsp_url(host: str, port: int, path: str, username: Optional[str], password: Optional[str]) -> str:
    host = host.strip()
    auth = ""
    if (username or "").strip() and (password or "").strip():
        auth = f"{username.strip()}:{password.strip()}@"
    return f"rtsp://{auth}{host}:{port}{path}"


def generate_common_rtsp_urls(
    host: str,
    username: Optional[str],
    password: Optional[str],
    brand: str = "Unknown/Generic",
    ports: Optional[Iterable[int]] = None,
) -> List[str]:
    templates = BRAND_TEMPLATES.get(brand, BRAND_TEMPLATES["Unknown/Generic"]) or []
    ports_list = list(ports) if ports is not None else DEFAULT_PORTS
    urls: List[str] = []
    for p in ports_list:
        for path in templates:
            urls.append(_build_rtsp_url(host, p, path, username, password))
    seen = set()
    uniq: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def probe_common_rtsp(
    host: str,
    username: Optional[str],
    password: Optional[str],
    brand: str = "Unknown/Generic",
    ports: Optional[Iterable[int]] = None,
    max_to_try: int = 20,
    attempts: int = 8,
    delay: float = 0.1,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    candidates = generate_common_rtsp_urls(host, username, password, brand=brand, ports=ports)
    results: List[Dict[str, Any]] = []
    best_url: Optional[str] = None
    to_try = candidates[:max_to_try]
    logger.info("probe_common_rtsp: trying %s candidates host=%s brand=%s", len(to_try), host, brand)
    for url in to_try:
        masked = _mask_url(url)
        try:
            parts = urlsplit(url)
            if parts.hostname and parts.port:
                if not _tcp_reachable(parts.hostname, parts.port, timeout=1.5):
                    results.append({
                        "url": url,
                        "masked": masked,
                        "ok": False,
                        "fps": None,
                        "size": None,
                        "err": f"TCP not reachable {parts.hostname}:{parts.port}",
                    })
                    logger.debug("probe_common_rtsp: skip %s (port unreachable)", masked)
                    continue
        except Exception:
            pass
        ok, fps, size, _jpeg, err = test_camera(url, attempts=attempts, delay=delay)
        results.append({"url": url, "masked": masked, "ok": ok, "fps": fps, "size": size, "err": err})
        if ok and best_url is None:
            best_url = url
            logger.info("probe_common_rtsp: found working %s", masked)
            break
    if best_url is None:
        logger.warning("probe_common_rtsp: no working RTSP URL found (tried=%s)", len(to_try))
    return best_url, results