import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image

from detection import load_model, detect_objects
from calibration import calibrate_camera
from utils import estimate_distance

st.title("📸 Object Detection + Distance Estimation")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
chessboard_images = st.file_uploader("Upload chessboard images (8x8)", type=["jpg"], accept_multiple_files=True)

if uploaded_image and chessboard_images:
    # Save image temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_image.read())
    image_path = tfile.name

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load YOLO and detect
    model = load_model()
    detections, annotated_image = detect_objects(model, image)

    # Calibrate camera
    camera_matrix = calibrate_camera(chessboard_images)

    if camera_matrix is None:
        st.error("Calibration failed. Please use clear chessboard images.")
    else:
        #print the camera matrix
        st.subheader("Camera Calibration Matrix:")
        st.write(camera_matrix)
        # Show original image
        st.subheader("Original Image:")
        st.image(image_rgb, caption="Original Image", channels="RGB", use_container_width=True)
        # Show annotated image with bounding boxes
        st.subheader("Annotated Image:")
        st.image(annotated_image, caption="Detected Objects", channels="BGR", use_container_width=True)

        st.subheader("🔍 Detected Objects with Estimated Distances:")
        for det in detections:
            distance = estimate_distance(det["width"], camera_matrix)
            st.markdown(
                f"**{det['class_name']}** — Width: `{det['width']:.2f}px`, Distance: `{distance:.2f} cm`"
            )

else:
    st.info("Upload both an image and chessboard calibration images to start.")
