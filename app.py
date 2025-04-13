import streamlit as st
import cv2
import tempfile
import numpy as np
import os
from PIL import Image

from detection import load_model, detect_objects
from calibration import calibrate_camera, save_camera_matrix, load_camera_matrix
from utils import estimate_distance

st.title("📸 Object Detection + Distance Estimation")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
st.caption("Only upload a new chessboard set if calibrating for the first time.")

# Optional: Allow recalibration
if os.path.exists("camera_matrix.npy"):
    if st.button("🔁 Recalibrate Camera"):
        os.remove("camera_matrix.npy")
        st.success("Calibration reset. Please upload new chessboard images.")

# Calibrate only if needed
if not os.path.exists("camera_matrix.npy"):
    chessboard_images = st.file_uploader("Upload chessboard images (8x8)", type=["jpg"], accept_multiple_files=True, key="chess_upload")
    if chessboard_images:
        st.info("Calibrating camera for the first time...")
        camera_matrix = calibrate_camera(chessboard_images)
        if camera_matrix is not None:
            save_camera_matrix(camera_matrix)
            st.success("✅ Calibration successful and saved!")
        else:
            st.error("❌ Calibration failed. Please upload valid chessboard images.")
            st.stop()
    else:
        st.warning("📸 Please upload chessboard images for one-time calibration.")
        st.stop()
else:
    camera_matrix = load_camera_matrix()

# Proceed if image is uploaded
if uploaded_image:
    # Save image temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_image.read())
    image_path = tfile.name

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load YOLO and detect
    model = load_model()
    detections, annotated_image = detect_objects(model, image)

    # Show calibration matrix
    st.subheader("Camera Calibration Matrix:")
    st.write(camera_matrix)
    
    # Show original image
    st.subheader("Original Image:")
    st.image(image_rgb, caption="Original Image", channels="RGB", use_container_width=True)

    # Show annotated image with bounding boxes
    st.subheader("Annotated Image:")
    st.image(annotated_image, caption="Detected Objects", channels="BGR", use_container_width=True)

    # Display distances
    st.subheader("🔍 Detected Objects with Estimated Distances:")
    for det in detections:
        distance = estimate_distance(det["width"], camera_matrix)
        st.markdown(
            f"**{det['class_name']}** — Width: `{det['width']:.2f}px`, Distance: `{distance:.2f} cm`"
        )
else:
    st.info("Upload an image to start.")
