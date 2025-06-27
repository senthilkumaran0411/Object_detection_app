import torch
from PIL import Image
import cv2
import numpy as np
import tempfile
import streamlit as st
import os
import subprocess
import sys

# Set page config
st.set_page_config(
    page_title="AI Object Detection - YOLOv5",
    page_icon="🤖",
    layout="wide"
)

# Load custom HTML if needed
try:
    with open("index.html", "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)
except FileNotFoundError:
    pass  # Continue without custom HTML if file doesn't exist

# Load the pretrained YOLOv5 model with proper error handling
@st.cache_resource
def load_model():
    try:
        # First try to import ultralytics directly
        import ultralytics
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        return model
    except ImportError:
        # If not found, install it properly
        st.warning("Installing required ultralytics package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        import ultralytics
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

def process_frame(frame):
    if model is None:
        return frame  # Return original frame if model failed to load
    
    try:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(image)
        results.render()
        processed_frame = results.ims[0]
        return cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        return frame  # Return original frame if processing fails

# Streamlit UI
st.title("AI Object Detection - YOLOv5")
st.write("Upload a video, use your webcam, or upload an image for real-time object detection.")

# Sidebar
st.sidebar.title("Control Panel")
use_webcam = st.sidebar.checkbox("Use Webcam")
uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if 'stop_webcam' not in st.session_state:
    st.session_state.stop_webcam = False
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Image Upload
if uploaded_image and not use_webcam and not uploaded_video:
    st.session_state.processing = True
    try:
        st.write("Processing uploaded image...")
        input_image = Image.open(uploaded_image)
        processed_image = process_frame(np.array(input_image))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Image")
            st.image(input_image, width=400)
        with col2:
            st.subheader("Processed Image")
            st.image(Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)), width=400)

        # Prepare download button
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(processed_image_rgb)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            temp_file_path = tmp_file.name
            img_pil.save(tmp_file.name, format="PNG")

            with open(temp_file_path, "rb") as f:
                file_content = f.read()

            st.download_button(
                label="Download Processed Image",
                data=file_content,
                file_name="processed_image.png",
                mime="image/png"
            )
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        st.session_state.processing = False

# Video Upload
elif uploaded_video and not use_webcam and not st.session_state.processing:
    st.session_state.processing = True
    temp_video_path = None
    temp_processed_video_path = None
    
    try:
        st.write("Processing uploaded video...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            temp_video_path = tmp_video.name
            tmp_video.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video_path)
        
        if not cap.isOpened():
            st.error("Error opening video file")
        else:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_processed:
                temp_processed_video_path = tmp_processed.name

            out = cv2.VideoWriter(temp_processed_video_path, fourcc, fps, (width, height))
            stframe = st.empty()
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_frames = 0

            while cap.isOpened() and not st.session_state.stop_webcam:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = process_frame(frame)
                out.write(processed_frame)
                stframe.image(Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)))
                processed_frames += 1
                progress_bar.progress(processed_frames / total_frames)

            cap.release()
            out.release()
            st.success("Video processing complete.")

            with open(temp_processed_video_path, "rb") as video_file:
                video_content = video_file.read()

            st.download_button(
                label="Download Processed Video",
                data=video_content,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    finally:
        for file_path in [temp_video_path, temp_processed_video_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass
        st.session_state.processing = False

# Webcam
elif use_webcam and not st.session_state.processing:
    st.session_state.processing = True
    try:
        st.write("Capturing from webcam...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not access the webcam. Please check your device.")
        else:
            stframe = st.empty()
            stop_button = st.button("Stop Webcam")

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame.")
                    break

                frame = cv2.flip(frame, 1)
                processed_frame = process_frame(frame)
                stframe.image(Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)))

                if stop_button:
                    st.session_state.stop_webcam = True
                    break

            cap.release()
            st.write("Webcam feed stopped.")
    except Exception as e:
        st.error(f"Error accessing webcam: {str(e)}")
    finally:
        st.session_state.stop_webcam = False
        st.session_state.processing = False

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
