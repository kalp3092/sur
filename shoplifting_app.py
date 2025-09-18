from pathlib import Path

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from src.yolo_detector import YOLODetector
from src.config import get_settings

st.title("Shoplifting Detection - Batch Images & Video")

settings = get_settings()
detector = YOLODetector(model_name=settings.yolo_model, conf_thresh=settings.detection_confidence)

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

st.header("Batch Image Detection")
image_files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if image_files:
    for uploaded_file in image_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is not None:
            dets = detector.detect(img)
            st.write(f"{uploaded_file.name}: {len(dets)} detections")
            # Check for shoplifting class (by name or class id)
            found_shoplifting = False
            for d in dets:
                st.write(f"bbox={d.xyxy} conf={d.conf:.3f} cls={d.cls}")
                # If your model uses class name, replace 0 with the correct class id or check label
                if hasattr(d, 'cls') and (getattr(d, 'cls', None) == 'shoplifting' or getattr(d, 'cls', None) == 0):
                    found_shoplifting = True
            if found_shoplifting:
                st.error("Shoplifting detected!")
            out = draw_boxes(img, dets)
            st.image(out, channels="BGR", caption=f"{uploaded_file.name} - Detection Result")
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
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"Processing...")
    progress = st.progress(0)
    idx = 0
    found_shoplifting = False
    shoplifting_frames = []  # Store (frame_idx, frame) where shoplifting detected
    shoplifting_images = []  # For thumbnails
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        shoplifting_in_this_frame = False
        for d in dets:
            # If your model uses class name, replace 0 with the correct class id or check label
            if hasattr(d, 'cls') and (getattr(d, 'cls', None) == 'shoplifting' or getattr(d, 'cls', None) == 0):
                found_shoplifting = True
                shoplifting_in_this_frame = True
        out_frame = draw_boxes(frame, dets)
        writer.write(out_frame)
        if shoplifting_in_this_frame:
            # Save frame for thumbnails and video
            shoplifting_frames.append(out_frame.copy())
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
        # Write a video of only shoplifting frames if any
        if shoplifting_frames:
            shoplift_vid_path = tempfile.mktemp(suffix='.mp4')
            shoplift_writer = cv2.VideoWriter(shoplift_vid_path, fourcc, fps, (w, h))
            for f in shoplifting_frames:
                shoplift_writer.write(f)
            shoplift_writer.release()
            with open(shoplift_vid_path, "rb") as f:
                st.download_button("Download Shoplifting Frames Video", f, file_name="shoplifting_frames.mp4")
            os.remove(shoplift_vid_path)
    st.success(f"Processed {idx} frames. Download result below.")
    with open(out_path, "rb") as f:
        st.download_button("Download Annotated Video", f, file_name="annotated_video.mp4")
    os.remove(out_path)
    os.remove(tfile.name)
