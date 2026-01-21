import streamlit as st
import numpy as np
import cv2
import os
import gdown
from PIL import Image
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# PAGE CONFIG
st.set_page_config(
    page_title="Concrete Crack Detection",
    layout="wide"
)
st.title("🛠️ Concrete Crack Detection System")

# -----------------------------
# LOAD CNN MODEL
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "crack_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading CNN model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)

# -----------------------------
# FUNCTIONS

def cnn_predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((150, 150))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    return float(model.predict(arr, verbose=0)[0][0])

def crack_severity(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_pixels = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 60:  # ignore tiny noise
            valid_pixels += area

    severity = (valid_pixels / thresh.size) * 100
    return round(severity, 3), thresh

def overlay_crack(img_path, thresh):
    img = cv2.imread(img_path)
    overlay = img.copy()
    overlay[thresh == 255] = [0, 0, 255]
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# -----------------------------
# IMAGE UPLOAD
uploaded_file = st.file_uploader(
    "Upload concrete image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    img = Image.open(uploaded_file)
    temp_path = "temp.jpg"
    img.save(temp_path)

    # CNN Prediction
    cnn_score = cnn_predict(temp_path)

    # CNN THRESHOLD (KEY FIX)
    CNN_THRESHOLD = 0.75

    if cnn_score < CNN_THRESHOLD:
        decision = "No Crack"
        severity_level = "None"
        recommendation = "Structure is safe"
        severity = 0.0
        show_overlay = False
        thresh = None
    else:
        decision = "Crack Detected"
        severity, thresh = crack_severity(temp_path)
        show_overlay = True

        if severity < 1:
            severity_level = "Low"
            recommendation = "Monitor periodically"
        elif severity < 5:
            severity_level = "Medium"
            recommendation = "Repair recommended"
        else:
            severity_level = "High"
            recommendation = "Immediate maintenance required"

    # -----------------------------
    # DISPLAY IMAGES
    col1, col2 = st.columns(2)

    col1.image(img, caption="Original Image", use_column_width=True)

    if show_overlay and thresh is not None:
        overlay_img = overlay_crack(temp_path, thresh)
        col2.image(
            overlay_img,
            caption="Crack Visualization",
            use_column_width=True
        )
    else:
        col2.image(
            img,
            caption="No Crack Detected",
            use_column_width=True
        )

    # -----------------------------
    # RESULTS
    if decision == "Crack Detected":
        st.error(f"Result: {decision}")
    else:
        st.success(f"Result: {decision}")

    st.info(f"Severity Level: {severity_level}")
    st.write(f"🧠 CNN Confidence Score: **{round(cnn_score, 3)}**")
    st.write(f"📏 Crack Area (%): **{severity}**")
    st.write(f"🛠 Recommendation: **{recommendation}**")
