from pathlib import Path

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import threading
import time
from datetime import datetime
from collections import deque
from src.yolo_detector import YOLODetector
from src.camera_utils import build_url_with_credentials, test_camera
from urllib.parse import urlsplit, urlunsplit, quote, unquote
from src.logging_config import init_logging
from src.config import get_settings

logger = init_logging()
st.title("Shoplifting Detection - Batch Images & Video")

settings = get_settings()
detector = YOLODetector(model_name=settings.yolo_model, conf_thresh=settings.detection_confidence)


def _generate_rtsp_variants(url: str) -> list:
    """Return a small list of URL variants to try for RTSP auth/encoding issues.

    Variants include: original, fully-unquoted, and reassembled forms with
    different quoting of username/password when present.
    """
    variants = []
    try:
        parsed = urlsplit(str(url))
    except Exception:
        # fallback to simple attempts
        v = str(url)
        return [v, unquote(v)] if unquote(v) != v else [v]

    orig = str(url)
    if orig not in variants:
        variants.append(orig)

    unq_full = unquote(orig)
    if unq_full not in variants:
        variants.append(unq_full)

    # If creds present in netloc, try variants with different quoting
    if "@" in parsed.netloc:
        try:
            creds, hostrest = parsed.netloc.split("@", 1)
            if ":" in creds:
                user, pwd = creds.split(":", 1)
                user_unq = unquote(user)
                pwd_unq = unquote(pwd)
                attempts = [
                    (user, pwd),
                    (user_unq, pwd),
                    (user, pwd_unq),
                    (user_unq, pwd_unq),
                ]
                for u, p in attempts:
                    try:
                        new_net = f"{quote(u, safe='')}:{quote(p, safe='')}@{hostrest}"
                        candidate = urlunsplit((parsed.scheme, new_net, parsed.path, parsed.query, ""))
                        if candidate not in variants:
                            variants.append(candidate)
                    except Exception:
                        continue
        except Exception:
            pass

    return variants


def is_shoplifting_detection(d) -> bool:
    """Business rule: consider a detection as shoplifting when its confidence >= configured alert threshold."""
    try:
        conf = float(getattr(d, 'conf', 0.0))
    except Exception:
        conf = 0.0
    thresh = float(getattr(settings, 'alert_confidence', float(os.getenv('SD_ALERT_CONFIDENCE', '0.85'))))
    return conf >= thresh

# UI mode selector: Video (uploads) or Live Feed (camera inputs)
mode = st.sidebar.radio("Mode", ["Video", "Live Feed"], index=0)

# Alerts directory
alerts_dir = Path(os.getenv('SD_ALERT_DIR', 'alerts'))
alerts_dir.mkdir(exist_ok=True)

# Session state keys for live cameras
if 'live_threads' not in st.session_state:
    st.session_state.live_threads = {}  # name -> Thread
if 'live_stops' not in st.session_state:
    st.session_state.live_stops = {}  # name -> bool
if 'live_latest' not in st.session_state:
    st.session_state.live_latest = {}  # name -> (ts, jpeg_bytes)
if 'live_any_alert' not in st.session_state:
    st.session_state.live_any_alert = {}  # name -> bool
if 'live_last_alert_bytes' not in st.session_state:
    st.session_state.live_last_alert_bytes = {}  # name -> jpeg bytes


def _encode_jpeg_bgr(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.jpg', image)
    return buf.tobytes() if ok else b''


def cam_worker(source, name: str, pre_seconds: float = 2.0, post_seconds: float = 3.0):
    """Background worker that reads frames, runs detection, updates session_state.

    It saves the last-detected annotated frame as JPEG bytes to
    `st.session_state.live_last_alert_bytes[name]` and sets
    `st.session_state.live_any_alert[name] = True` when any detection crosses
    the alert threshold.
    """
    cap = None
    try:
        # Prefer FFMPEG backend and set sane FFmpeg options for RTSP
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|max_delay;5000000|stimeout;30000000",
        )

        # Try multiple URL encodings/quoting variants to mitigate auth/encoding issues
        candidates = _generate_rtsp_variants(str(source))
        cap = None
        for cand in candidates:
            # try FFMPEG backend first
            try:
                c = cv2.VideoCapture(cand, cv2.CAP_FFMPEG)
                try:
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if c.isOpened():
                    cap = c
                    logger.info("cam_worker: opened using candidate URL (ffmpeg): %s", cand)
                    break
                else:
                    try:
                        c.release()
                    except Exception:
                        pass
            except Exception:
                pass

            # fallback to default backend for this candidate
            try:
                c = cv2.VideoCapture(cand)
                try:
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if c.isOpened():
                    cap = c
                    logger.info("cam_worker: opened using candidate URL (default): %s", cand)
                    break
                else:
                    try:
                        c.release()
                    except Exception:
                        pass
            except Exception:
                pass

        if cap is None or not cap.isOpened():
            logger.error("cam_worker: failed to open camera: %s", str(source))
            try:
                if 'live_latest' not in st.session_state:
                    st.session_state['live_latest'] = {}
                st.session_state.live_latest[name] = ("__error__", None)
            except Exception:
                pass
            return

        logger.info("cam_worker: started name=%s source=%s", name, str(source))

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 120:
            fps = 15.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None

        pre_buffer = deque(maxlen=int(pre_seconds * fps) if fps else 30)

        while not st.session_state.live_stops.get(name, False):
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            # Run detection (may raise) — keep behavior unchanged
            try:
                dets = detector.detect(frame)
            except Exception as e:
                logger.exception("cam_worker: detection error for %s: %s", name, e)
                dets = []

            out = draw_boxes(frame, dets)

            now = time.time()
            # store latest as jpeg bytes to keep session small
            try:
                if 'live_latest' not in st.session_state:
                    st.session_state['live_latest'] = {}
                st.session_state.live_latest[name] = (now, _encode_jpeg_bgr(out))
            except Exception:
                pass

            # maintain pre-buffer
            pre_buffer.append(out.copy())

            # check for shoplifting
            found = False
            for d in dets:
                if is_shoplifting_detection(d):
                    found = True
                    break

            if found:
                try:
                    if 'live_any_alert' not in st.session_state:
                        st.session_state['live_any_alert'] = {}
                    if 'live_last_alert_bytes' not in st.session_state:
                        st.session_state['live_last_alert_bytes'] = {}
                    st.session_state.live_any_alert[name] = True
                    st.session_state.live_last_alert_bytes[name] = _encode_jpeg_bgr(out)
                except Exception:
                    pass
                # also write a timestamped JPEG file in alerts_dir for persistence
                try:
                    ts = datetime.fromtimestamp(now).strftime('%Y%m%d_%H%M%S')
                    fname = alerts_dir / f"{name}_alert_{ts}.jpg"
                    with open(fname, 'wb') as f:
                        try:
                            data = st.session_state.live_last_alert_bytes.get(name)
                        except Exception:
                            data = None
                        if not data:
                            data = _encode_jpeg_bgr(out)
                        f.write(data)
                    logger.info("cam_worker: saved alert frame name=%s file=%s", name, str(fname))
                except Exception as e:
                    logger.exception("cam_worker: failed to save alert frame for %s: %s", name, e)

            time.sleep(0.01)
    finally:
        if cap is not None:
            cap.release()
        logger.info("cam_worker: stopped name=%s", name)


def draw_boxes(img: np.ndarray, detections):
    out = img.copy()
    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        conf = d.conf
        cls = d.cls
        # If shoplifting and conf > 0.8, red and 'shoplifting', else green and 'normal'
        is_shoplifting = (cls == 'shoplifting' or cls == 0) and conf > 0.8  # adjust 0 if your class index is different
        color = (0, 0, 255) if is_shoplifting else (0, 255, 0)
        label = "shoplifting" if is_shoplifting else "normal"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{label} conf:{conf:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out

if mode == "Video":
    st.header("Batch Image Detection")
    image_files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if image_files:
        for uploaded_file in image_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            if img is not None:
                dets = detector.detect(img)
                st.write(f"{uploaded_file.name}: {len(dets)} detections")
                # Check for shoplifting using configured confidence threshold
                found_shoplifting = False
                for d in dets:
                    st.write(f"bbox={d.xyxy} conf={d.conf:.3f} cls={d.cls}")
                    if is_shoplifting_detection(d):
                        found_shoplifting = True
                        break
                out = draw_boxes(img, dets)
                st.image(out, channels="BGR", caption=f"{uploaded_file.name} - Detection Result")
                if found_shoplifting:
                    st.error("Shoplifting detected!")
                    # Offer download of the annotated frame for this image
                    ok, buf = cv2.imencode('.jpg', out)
                    if ok:
                        st.download_button(
                            label="Download detected frame",
                            data=buf.tobytes(),
                            file_name=f"{Path(uploaded_file.name).stem}_shoplifting.jpg",
                            mime="image/jpeg",
                        )
                        # Persist annotated image to alerts dir
                        try:
                            tsf = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                            img_fname = alerts_dir / f"image_{Path(uploaded_file.name).stem}_alert_{tsf}.jpg"
                            with open(img_fname, 'wb') as _f:
                                _f.write(buf.tobytes())
                            st.info(f"Saved alert image: {img_fname}")
                        except Exception as _e:
                            st.warning(f"Failed to save alert image: {_e}")
            else:
                st.error(f"Failed to read {uploaded_file.name}")

    st.header("Video Detection")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"]) 
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.close()
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = tempfile.mktemp(suffix='.mp4')
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"Processing...")
        progress = st.progress(0)
        idx = 0
        found_shoplifting = False
        shoplifting_frames = []  # Store (frame_idx, frame) where shoplifting detected
        shoplifting_images = []  # For thumbnails
        first_detected_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets = detector.detect(frame)
            shoplifting_in_this_frame = False
            for d in dets:
                if is_shoplifting_detection(d):
                    found_shoplifting = True
                    shoplifting_in_this_frame = True
            out_frame = draw_boxes(frame, dets)
            writer.write(out_frame)
            if shoplifting_in_this_frame:
                # Save frame for thumbnails and video
                shoplifting_frames.append(out_frame.copy())
                if first_detected_frame is None:
                    first_detected_frame = out_frame.copy()
                # Persist this detected frame as JPEG
                try:
                    tsf = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    frame_fname = alerts_dir / f"video_{Path(tfile.name).stem}_frame_{idx}_alert_{tsf}.jpg"
                    okf, buff = cv2.imencode('.jpg', out_frame)
                    if okf:
                        with open(frame_fname, 'wb') as _ff:
                            _ff.write(buff.tobytes())
                except Exception as _e:
                    print(f"Failed to save detected video frame: {_e}")
                # For thumbnails, resize to width 256 for display
                thumb = cv2.resize(out_frame, (256, int(256 * h / w))) if w > 0 else out_frame
                shoplifting_images.append(thumb)
            idx += 1
            if frame_count > 0:
                progress.progress(min(idx / frame_count, 1.0))
        cap.release()
        writer.release()
        if found_shoplifting:
            st.error("Shoplifting detected in video!")
            # Offer download of the first detected frame as a JPEG
            if first_detected_frame is not None:
                ok, buf = cv2.imencode('.jpg', first_detected_frame)
                if ok:
                    st.download_button(
                        label="Download first detected frame",
                        data=buf.tobytes(),
                        file_name="shoplifting_first_frame.jpg",
                        mime="image/jpeg",
                    )
        st.success(f"Processed {idx} frames. Download result below.")
        with open(out_path, "rb") as f:
            st.download_button("Download Annotated Video", f, file_name="annotated_video.mp4")
        os.remove(out_path)
        os.remove(tfile.name)
else:
    # Live Feed UI
    st.header("Live Feed")
    live_choice = st.radio("Live source", ["Local camera (attached)", "Remote camera (URL)"])
    col1, col2 = st.columns(2)
    with col1:
        pre_seconds = st.number_input("Pre-roll seconds", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    with col2:
        post_seconds = st.number_input("Post-roll seconds", min_value=1.0, max_value=20.0, value=3.0, step=0.5)

    if live_choice == "Local camera (attached)":
        cam_index = st.number_input("Webcam index", min_value=0, max_value=10, value=0, step=1)
        source_val = int(cam_index)
        cam_name = st.text_input("Camera name (unique)", value=f"webcam_{cam_index}")
    else:
        cam_url = st.text_input("Remote camera URL (rtsp:// or http:// ...)")
        colu, colp = st.columns(2)
        with colu:
            cam_user = st.text_input("Username", key="remote_user")
        with colp:
            cam_pass = st.text_input("Password", type="password", key="remote_pass")

        # Build final URL with credentials (if provided) and show masked
        final_url = build_url_with_credentials(cam_url.strip(), cam_user, cam_pass)
        masked = final_url
        if "@" in masked:
            try:
                prefix, rest = masked.split("://", 1)
                _creds, hostrest = rest.split("@", 1)
                masked = f"{prefix}://***:***@{hostrest}"
            except Exception:
                pass
        st.caption(f"Final URL: {masked}")
        logger.info("UI: final camera URL built (masked) %s", masked)

        # Hint for web page URLs
        if final_url.startswith("http") and not any(tok in final_url.lower() for tok in [".mjpg", ".mjpeg", ".m3u8", "?action=stream", "video.mjpg", "mjpegstream"]):
            st.info("This looks like a web page. Provide a direct RTSP or MJPEG stream URL.")

        if st.button("Test camera"):
            ok, fps, size, jpeg_bytes, err = test_camera(final_url)
            logger.info("UI: test_camera result ok=%s fps=%s size=%s err=%s", ok, fps, size, err)
            if ok:
                st.success(f"Connected. FPS ~ {fps or 'n/a'}, size: {size or 'n/a'}")
                if jpeg_bytes:
                    st.image(jpeg_bytes, caption="Test frame", use_column_width=True)
            else:
                st.error(f"Test failed: {err}")

        source_val = final_url
        cam_name = st.text_input("Camera name (unique)", value=(cam_user or "remote_cam"))

    start_btn = st.button("Start")
    stop_btn = st.button("Stop")

    if start_btn:
        # Auto-generate camera name when not provided by user
        if not cam_name:
            try:
                from urllib.parse import urlsplit

                parts = urlsplit(str(source_val))
                host = parts.hostname or parts.path or f"remote_cam_{len(st.session_state.live_threads) + 1}"
                cam_name = host.replace(".", "_").replace(":", "_")
            except Exception:
                cam_name = f"remote_cam_{len(st.session_state.live_threads) + 1}"

        # reset stop flag and spawn thread
        st.session_state.live_stops[cam_name] = False
        if cam_name in st.session_state.live_threads and st.session_state.live_threads[cam_name].is_alive():
            st.info("Camera already running")
            logger.info("UI: start requested but already running name=%s", cam_name)
        else:
            t = threading.Thread(target=cam_worker, args=(source_val, cam_name, pre_seconds, post_seconds), daemon=True)
            st.session_state.live_threads[cam_name] = t
            t.start()
            logger.info("UI: started camera worker name=%s", cam_name)
            st.success(f"Started camera worker: {cam_name}")

    if stop_btn:
        if cam_name in st.session_state.live_threads:
            st.session_state.live_stops[cam_name] = True
            logger.info("UI: stopping camera worker name=%s", cam_name)
            st.success(f"Stopping camera worker: {cam_name}")
        else:
            st.info("No worker found for that name")
            logger.info("UI: stop requested but no worker found name=%s", cam_name)

    # Display live previews for all running cameras
    st.subheader("Live previews")
    for name, val in list(st.session_state.live_latest.items()):
        st.markdown(f"### {name}")
        ts, jpeg_bytes = val
        if ts == "__error__":
            st.error("Failed to open camera (check URL/index and network)")
            continue
        if jpeg_bytes:
            st.image(jpeg_bytes, channels="RGB", use_column_width=True, caption=f"Last frame: {datetime.fromtimestamp(ts).isoformat()}")
        else:
            st.info("Waiting for frames...")

        # show alert status and allow downloading the last alert frame
        if st.session_state.live_any_alert.get(name, False) and st.session_state.live_last_alert_bytes.get(name):
            st.error("Shoplifting detected!")
            st.download_button("Download last detected frame", st.session_state.live_last_alert_bytes[name], file_name=f"{name}_last_alert.jpg", mime="image/jpeg")
        else:
            st.success("Nobody stole anything")
