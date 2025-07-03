import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

# Set page config
st.set_page_config(
    page_title="AI Object Detection - YOLOv5",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("AI Object Detection - YOLOv5")
st.write("Upload an image or video for object detection using a pretrained YOLOv5 model.")

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov5s.pt")  # You can also use yolov5m.pt or yolov5l.pt
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Function to process image with model
def process_image(image):
    try:
        results = model(image)
        annotated = results[0].plot()
        return annotated
    except Exception as e:
        st.error(f"Image processing failed: {e}")
        return None

# Sidebar controls
st.sidebar.title("Upload Options")
uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

# Image upload handling
if uploaded_image:
    st.subheader("Input Image")
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Processed Image")
    annotated_image = process_image(image)

    if annotated_image is not None:
        st.image(annotated_image, caption="Detected Objects", use_column_width=True)

        # Download button
        img_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            pil_img.save(tmp_file.name)
            with open(tmp_file.name, "rb") as f:
                st.download_button("Download Processed Image", f.read(), "processed_image.png", "image/png")
            os.unlink(tmp_file.name)

# Video upload handling
elif uploaded_video:
    st.subheader("Uploaded Video")
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("Unable to read video file.")
    else:
        stframe = st.empty()
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))

        progress = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()
            out.write(annotated)
            stframe.image(annotated, channels="BGR", use_column_width=True)

            count += 1
            progress.progress(min(count / frame_count, 1.0))

        cap.release()
        out.release()
        st.success("Video Processing Completed.")

        with open(temp_out.name, "rb") as f:
            st.download_button("Download Processed Video", f.read(), "processed_video.mp4", "video/mp4")

        os.unlink(temp_out.name)
        os.unlink(tfile.name)

# Webcam (Note: only works locally or with streamlit-webrtc, not on cloud)
else:
    st.info("Upload an image or video from the sidebar to begin detection.")
