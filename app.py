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
st.set_page_config(page_title="Concrete Crack Detection", layout="wide")
st.title("🛠️ Concrete Crack Detection System")
st.caption("AI-based structural crack analysis using CNN & image processing")

# -----------------------------
# LOAD MODEL
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
    return float(model.predict(arr)[0][0])

def cnn_level(score):
    if score < 0.4:
        return "Low"
    elif score < 0.7:
        return "Medium"
    else:
        return "High"

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
    return np.sum(edges > 0) / edges.size

def overlay_crack(img_path, thresh):
    img = cv2.imread(img_path)
    overlay = img.copy()
    overlay[thresh == 255] = [0, 0, 255]
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# -----------------------------
# UPLOAD IMAGE
uploaded_file = st.file_uploader(
    "📤 Upload concrete surface image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    img = Image.open(uploaded_file)
    temp_path = "temp.jpg"
    img.save(temp_path)

    # Predictions
    cnn_score = cnn_predict(temp_path)
    cnn_conf = cnn_level(cnn_score)
    severity, thresh = crack_severity(temp_path)
    edge_val = edge_ratio(temp_path)

    # -----------------------------
    # FINAL DECISION LOGIC

    if severity < 0.2 and cnn_score < 0.5:
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
    # DISPLAY IMAGES
    col1, col2 = st.columns(2)

    col1.image(img, caption="Original Image", use_column_width=True)

    if show_overlay:
        col2.image(
            overlay_crack(temp_path, thresh),
            caption="Detected Crack Area",
            use_column_width=True
        )
    else:
        col2.image(img, caption="No Crack Found", use_column_width=True)

    # -----------------------------
    # RESULTS DASHBOARD
    st.subheader("📊 Analysis Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("CNN Confidence", f"{round(cnn_score*100,2)}%", cnn_conf)
    m2.metric("Crack Area", f"{severity} %")
    m3.metric("Edge Density", f"{round(edge_val,4)}")

    st.progress(min(int(severity * 5), 100))

    # -----------------------------
    # FINAL STATUS
    if decision == "Crack Detected":
        st.error(f"⚠️ Result: {decision}")
    else:
        st.success(f"✅ Result: {decision}")

    st.info(f"🧱 Severity Level: {severity_level}")
    st.write(f"🛠 Recommendation: {recommendation}")

    with st.expander("📘 How the decision was made"):
        st.write(f"- CNN confidence level: **{cnn_conf}**")
        st.write(f"- Crack pixel ratio: **{severity}%**")
        st.write(f"- Edge density: **{round(edge_val,4)}**")
