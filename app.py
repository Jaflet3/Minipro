import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# PAGE SETUP
st.set_page_config(page_title="Crack Detection System", layout="wide")
st.title("🛠️ Concrete Crack Detection System")

# -------------------------
# DOWNLOAD & LOAD MODEL
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "crack_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading CNN model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)

# -------------------------
# FUNCTIONS

def cnn_predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((150, 150))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    return float(model.predict(arr, verbose=0)[0][0])

def crack_severity(img_path, thresh_val):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, thresh_val, 255, cv2.THRESH_BINARY_INV
    )

    crack_pixels = np.sum(thresh == 255)
    severity = (crack_pixels / thresh.size) * 100

    return round(severity, 3), thresh

def edge_strength(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    return edge_ratio

def overlay_crack(img_path, thresh):
    img = cv2.imread(img_path)
    overlay = img.copy()
    overlay[thresh == 255] = [0, 0, 255]
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# -------------------------
# USER INPUT
threshold_val = st.slider("Binary Threshold", 80, 200, 130)
uploaded_files = st.file_uploader(
    "Upload Image(s)", type=["jpg", "png", "jpeg"], accept_multiple_files=True
)

results = []

# -------------------------
# PROCESS IMAGES
if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file)
        path = f"temp_{file.name}"
        img.save(path)

        cnn_score = cnn_predict(path)
        severity, thresh = crack_severity(path, threshold_val)
        edge_ratio = edge_strength(path)

        # -------------------------
        # FINAL CORRECT DECISION LOGIC
        if cnn_score < 0.65 and severity < 0.8 and edge_ratio < 0.01:
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

        # -------------------------
        # DISPLAY
        st.subheader(file.name)
        col1, col2 = st.columns(2)

        col1.image(img, caption="Original Image", use_column_width=True)

        if show_overlay:
            overlay = overlay_crack(path, thresh)
            col2.image(overlay, caption="Crack Visualization", use_column_width=True)
        else:
            col2.image(img, caption="No Crack Detected", use_column_width=True)

        if decision == "Crack Detected":
            st.error(f"Result: {decision}")
        else:
            st.success(f"Result: {decision}")

        st.info(f"Severity Level: {severity_level}")
        st.write(f"🔍 CNN Score: **{round(cnn_score, 3)}**")
        st.write(f"📏 Crack Area (%): **{severity}**")
        st.write(f"📐 Edge Strength: **{round(edge_ratio, 4)}**")
        st.write(f"🛠 Recommendation: **{recommendation}**")
        st.divider()

        results.append({
            "Image": file.name,
            "Result": decision,
            "CNN Score": round(cnn_score, 3),
            "Crack Area (%)": severity,
            "Edge Ratio": round(edge_ratio, 4),
            "Severity Level": severity_level,
            "Recommendation": recommendation
        })

# -------------------------
# SUMMARY REPORT
if results:
    df = pd.DataFrame(results)
    st.subheader("📊 Detection Summary")
    st.dataframe(df)

    st.download_button(
        "📥 Download CSV Report",
        df.to_csv(index=False).encode("utf-8"),
        file_name="crack_detection_report.csv",
        mime="text/csv"
    )
