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
    layout="wide",
    page_icon="🛠️"
)

st.title("🛠️ Concrete Crack Detection System")
st.caption("Hybrid CNN + Image Processing based Structural Health Monitoring")

st.divider()

# -----------------------------
# LOAD MODEL
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "crack_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading CNN model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)

# -----------------------------
# FUNCTIONS

def cnn_predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((150, 150))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    return float(model.predict(arr)[0][0])

def crack_severity(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 127, 255, cv2.THRESH_BINARY_INV
    )

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
    overlay[thresh == 255] = [0, 0, 255]  # red overlay
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# -----------------------------
# UPLOAD IMAGE
uploaded_file = st.file_uploader(
    "📤 Upload Concrete Surface Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    img = Image.open(uploaded_file)
    temp_path = "temp.jpg"
    img.save(temp_path)

    # -----------------------------
    # PREDICTIONS
    cnn_score = cnn_predict(temp_path)
    severity, thresh = crack_severity(temp_path)
    edge_val = edge_ratio(temp_path)

    # -----------------------------
    # FINAL DECISION LOGIC (UNCHANGED)
    if severity < 0.2:
        decision = "No Crack"
        severity_level = "None"
        recommendation = "Structure is safe"
        show_overlay = False

    elif cnn_score < 0.65 and edge_val < 0.01:
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
    # IMAGE DISPLAY
    col1, col2 = st.columns(2)

    col1.image(img, caption="Original Image", use_column_width=True)

    if show_overlay:
        overlay_img = overlay_crack(temp_path, thresh)
        col2.image(
            overlay_img,
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
    # ANALYSIS DASHBOARD
    st.subheader("📊 Analysis Results")

    m1, m2, m3 = st.columns(3)

    # CNN CONFIDENCE
    m1.metric(
        "CNN Confidence",
        f"{round(cnn_score * 100, 2)}%",
        "High" if cnn_score > 0.8 else "Moderate"
    )

    # CRACK AREA
    m2.metric(
        "Crack Area",
        f"{severity} %"
    )

    # EDGE DENSITY
    m3.metric(
        "Edge Density",
        edge_val
    )

    # -----------------------------
    # RESULT MESSAGE (UI FIX ONLY)
    if decision == "Crack Detected":

        if severity == 0.0:
            st.warning("⚠️ Result: Micro Crack Detected (CNN-based)")
            st.caption(
                "CNN detected texture-based micro cracks "
                "that are not measurable using pixel analysis."
            )
        else:
            st.error("⚠️ Result: Crack Detected")

    else:
        st.success("✅ Result: No Crack Detected")

    # -----------------------------
    # SEVERITY & RECOMMENDATION
    st.info(f"🧱 Severity Level: **{severity_level}**")
    st.write(f"🛠 **Recommendation:** {recommendation}")

    # -----------------------------
    # TECHNICAL EXPLANATION (ADVANCED FEATURE)
    with st.expander("ℹ️ How this decision was made"):
        st.write("""
        - **CNN Model** detects texture-level cracks including micro-cracks  
        - **Crack Area** measures visible pixel-level crack coverage  
        - **Edge Density** validates structural discontinuities  
        - Final decision is based on **hybrid intelligence**
        """)

