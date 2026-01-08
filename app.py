import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import gdown
from tensorflow.keras.models import load_model
import pandas as pd

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Concrete Crack Detection", layout="wide")
st.title("🧠 AI-Based Concrete Crack Detection")

st.markdown("""
This system combines **Deep Learning (CNN)** and **Image Processing**  
to accurately detect **thin and thick concrete cracks** while avoiding false alarms.
""")

# ---------------- LOAD MODEL ----------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "crack_model.h5"

@st.cache_resource
def load_crack_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH, compile=False)

model = load_crack_model()

# ---------------- IMAGE FUNCTIONS ----------------
def preprocess_for_cnn(img):
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def cnn_predict(img):
    pred = model.predict(preprocess_for_cnn(img))[0][0]
    return float(pred)

def calculate_severity(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    crack_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    severity = (crack_pixels / total_pixels) * 100

    return round(severity, 2), thresh

def overlay_cracks(img, mask):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    overlay = img_cv.copy()
    overlay[mask == 255] = [0, 0, 255]
    return cv2.cvtColor(cv2.addWeighted(img_cv, 0.7, overlay, 0.3, 0), cv2.COLOR_BGR2RGB)

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "📤 Upload concrete images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

results = []

# ---------------- PROCESS IMAGES ----------------
if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")

        severity, mask = calculate_severity(img)
        cnn_score = cnn_predict(img)

        # -------- FINAL DECISION LOGIC --------
        if severity < 0.15:
            decision = "No Crack ✅"
        elif severity < 1.0:
            decision = "Crack Detected ⚠️" if cnn_score >= 0.55 else "No Crack ✅"
        else:
            decision = "Crack Detected ⚠️"

        # Severity category
        if severity < 1:
            category = "Low 🟢"
        elif severity < 5:
            category = "Medium 🟡"
        else:
            category = "High 🔴"

        overlay = overlay_cracks(img, mask)

        results.append({
            "Image Name": file.name,
            "Prediction": decision,
            "Severity (%)": severity,
            "Category": category,
            "Original": img,
            "Overlay": overlay
        })

    # ---------------- RESULTS TABLE ----------------
    st.subheader("📋 Detection Results")
    df = pd.DataFrame(results)[["Image Name", "Prediction", "Severity (%)", "Category"]]
    st.dataframe(df, use_container_width=True)

    # ---------------- IMAGE DISPLAY ----------------
    st.subheader("📷 Image Overlays")
    for r in results:
        st.markdown(
            f"**{r['Image Name']} — {r['Prediction']} | Severity: {r['Severity (%)']}% ({r['Category']})**"
        )
        col1, col2 = st.columns(2)
        col1.image(r["Original"], caption="Original Image", use_column_width=True)
        col2.image(r["Overlay"], caption="Crack Highlighted", use_column_width=True)

    st.success("✅ Crack analysis completed successfully!")
