import streamlit as st
import cv2
import tempfile
import time

from yolo_engine import YoloEngine

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "yolov8n-pose.pt"
TARGET_FPS = 15
FRAME_INTERVAL = 1.0 / TARGET_FPS
# ============================================================

st.set_page_config(page_title="AI Retail Theft Detection (Video Demo)", layout="wide")

st.title("🛍️ AI Retail Theft Detection – Video Analysis Demo")

st.markdown("""
### Offline Video-Based AI (Training & Demo Ready)
- Upload a retail video
- Analyze frame-by-frame
- Detect suspicious gestures
- No live camera required
""")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi"]
)

play = st.sidebar.toggle("▶ Run Analysis", value=False)

# ============================================================
# ENGINE (CACHED)
# ============================================================
@st.cache_resource
def load_engine():
    return YoloEngine(model_path=MODEL_PATH)

engine = load_engine()

# ============================================================
# UI PLACEHOLDERS
# ============================================================
frame_placeholder = st.empty()
status_placeholder = st.empty()
alert_placeholder = st.empty()

# ============================================================
# VIDEO HANDLING
# ============================================================
if uploaded_file is None:
    st.info("⬅️ Upload a video to start")
    st.stop()

# Save uploaded video to temp file
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_file.read())
video_path = tfile.name

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    st.error("❌ Failed to open uploaded video")
    st.stop()

# ============================================================
# FRAME PROCESSING LOOP (SAFE)
# ============================================================
last_time = 0.0

if play:
    while cap.isOpened():
        now = time.time()
        if now - last_time < FRAME_INTERVAL:
            continue
        last_time = now

        ret, frame = cap.read()
        if not ret:
            status_placeholder.success("✅ Video finished")
            break

        annotated, suspicious, _ = engine.process_frame(frame)


        if suspicious:
            alert_placeholder.error("⚠️ Suspicious gesture detected")
            status_placeholder.markdown("**Status:** 🚨 ALERT")
        else:
            alert_placeholder.success("✅ Normal behavior")
            status_placeholder.markdown("**Status:** 🟢 Monitoring")

        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, use_container_width=True)

    cap.release()
else:
    status_placeholder.markdown("**Status:** ⏸ Paused")
