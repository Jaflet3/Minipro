import streamlit as st
import numpy as np
import cv2
import os
import requests
from PIL import Image
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Concrete Crack Detection",
    layout="wide",
    page_icon="🛠️"
)

st.title("🛠️ Concrete Crack Detection System")
st.caption("Hybrid CNN + Image Processing based Structural Health Monitoring")
st.divider()

# -----------------------------
# GOOGLE DRIVE MODEL DOWNLOAD
# -----------------------------
MODEL_FILE = "crack_model.h5"
FILE_ID = "1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"

def download_from_gdrive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_model_local():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("📥 Downloading CNN model from Google Drive..."):
            download_from_gdrive(FILE_ID, MODEL_FILE)
    model = tf.keras.models.load_model(MODEL_FILE)
    return model

model = load_model_local()

# -----------------------------
# FUNCTIONS
# -----------------------------
def cnn_predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((150, 150))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    return float(model.predict(arr, verbose=0)[0][0])

def crack_severity(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

    crack_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    severity = (crack_pixels / total_pixels) * 100

    return round(severity, 3), thresh

def edge_ratio(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(gray, 100, 200)
    return round(np.sum(edges > 0) / edges.size, 4)

def overlay_crack(img_path, thresh):
    img = cv2.imread(img_path)
    overlay = img.copy()
    overlay[thresh == 255] = [0, 0, 255]
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Concrete Surface Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file)
    temp_path = "temp.jpg"
    img.save(temp_path)

    # -----------------------------
    # PREDICTIONS
    # -----------------------------
    cnn_score = cnn_predict(temp_path)
    severity, thresh = crack_severity(temp_path)
    edge_val = edge_ratio(temp_path)

    # -----------------------------
    # DECISION LOGIC
    # -----------------------------
    if severity < 0.2 and cnn_score < 0.65 and edge_val < 0.01:
        decision = "No Crack"
        severity_level = "None"
        recommendation = "Structure is safe"
        show_overlay = False
    else:
        decision = "Crack Detected"
        show_overlay = True

        if severity < 1.5:
            severity_level = "Low"
            recommendation = "Monitor periodically"
        elif severity < 5:
            severity_level = "Medium"
            recommendation = "Repair recommended"
        else:
            severity_level = "High"
            recommendation = "Immediate maintenance required"

    # -----------------------------
    # DISPLAY
    # -----------------------------
    col1, col2 = st.columns(2)

    col1.image(img, caption="Original Image", use_column_width=True)

    if show_overlay:
        col2.image(
            overlay_crack(temp_path, thresh),
            caption="Detected Crack Area",
            use_column_width=True
        )
    else:
        col2.image(
            img,
            caption="No Crack Found",
            use_column_width=True
        )

    st.divider()

    # -----------------------------
    # METRICS
    # -----------------------------
    st.subheader("📊 Analysis Results")
    m1, m2, m3 = st.columns(3)

    m1.metric("CNN Confidence", f"{cnn_score * 100:.2f}%")
    m2.metric("Crack Area", f"{severity} %")
    m3.metric("Edge Density", edge_val)

    # -----------------------------
    # RESULT
    # -----------------------------
    if decision == "Crack Detected":
        st.error("⚠️ Result: Crack Detected")
    else:
        st.success("✅ Result: No Crack Detected")

    st.info(f"🧱 Severity Level: **{severity_level}**")
    st.write(f"🛠 **Recommendation:** {recommendation}")

    with st.expander("ℹ️ How this decision was made"):
        st.write("""
        - CNN detects texture-level cracks
        - Crack Area measures visible damage
        - Edge Density validates discontinuities
        - Hybrid logic combines all signals
        """)
