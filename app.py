import streamlit as st
import cv2
import tempfile
import numpy as np
import os
from PIL import Image

from detection import load_model, detect_objects
from calibration import calibrate_camera, save_camera_matrix, load_camera_matrix
from utils import estimate_distance

st.set_page_config(page_title="Object Detection + Distance Estimation", layout="wide")
st.title("📸 Object Detection + Distance Estimation")

# Handle calibration matrix
if os.path.exists("camera_matrix.npy"):
    camera_matrix = load_camera_matrix()
    if st.button("🔁 Recalibrate Camera"):
        os.remove("camera_matrix.npy")
        st.success("Calibration reset. Please upload new chessboard images.")
        st.experimental_rerun()
else:
    st.subheader("📏 Camera Calibration Required")
    chessboard_images = st.file_uploader("Upload chessboard images (8x8)", type=["jpg"], accept_multiple_files=True, key="chess_upload")
    if chessboard_images:
        st.info("Calibrating camera for the first time...")
        camera_matrix = calibrate_camera(chessboard_images)
        if camera_matrix is not None:
            save_camera_matrix(camera_matrix)
            st.success("✅ Calibration successful and saved!")
            st.experimental_rerun()
        else:
            st.error("❌ Calibration failed. Please upload valid chessboard images.")
            st.stop()
    else:
        st.warning("📸 Please upload chessboard images to calibrate.")
        st.stop()

# --- Image Upload Handler ---
st.subheader("📤 Upload Image for Detection")

if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False

if not st.session_state.image_uploaded:
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="image_uploader")
    if uploaded_image:
        st.session_state.image_uploaded = True
        st.session_state.uploaded_image = uploaded_image
        st.experimental_rerun()

# Upload another image option
if st.session_state.image_uploaded:
    if st.button("📤 Upload Another Image"):
        st.session_state.image_uploaded = False
        st.session_state.uploaded_image = None
        st.experimental_rerun()

# --- Detection and Display ---
if st.session_state.image_uploaded:
    image_file = st.session_state.uploaded_image

    # Save image temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(image_file.read())
    image_path = tfile.name

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load model and detect
    model = load_model()
    detections, annotated_image = detect_objects(model, image)

    st.subheader("Camera Calibration Matrix:")
    st.write(camera_matrix)

    st.subheader("🖼 Original Image")
    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    st.subheader("📦 Annotated Image")
    st.image(annotated_image, caption="Detected Objects", channels="BGR", use_column_width=True)

    st.subheader("📏 Detected Objects and Estimated Distances")
    for det in detections:
        distance = estimate_distance(det["width"], camera_matrix)
        st.markdown(
            f"**{det['class_name']}** — Width: `{det['width']:.2f}px`, Distance: `{distance:.2f} cm`"
        )
