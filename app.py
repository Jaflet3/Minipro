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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crack Detection System", layout="wide")
st.title("🧠 AI-Based Crack Detection")

# ---------------- MODEL ----------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "crack_model.h5"

@st.cache_resource
def load_crack_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH, compile=False)

model = load_crack_model()

# ---------------- FUNCTIONS ----------------
def cnn_predict(image_path):
    img = Image.open(image_path).convert("RGB").resize((150,150))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    return float(model.predict(arr)[0][0])

def calculate_severity(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    severity = (np.sum(thresh == 255) / thresh.size) * 100
    return round(severity,2), thresh

def overlay_crack(image_path, mask):
    img = cv2.imread(image_path)
    overlay = img.copy()
    overlay[mask == 255] = [0,0,255]
    return cv2.addWeighted(img, 0.75, overlay, 0.25, 0)

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------- UI ----------------
files = st.file_uploader("Upload images", type=["jpg","jpeg","png"], accept_multiple_files=True)

results = []

if files:
    for f in files:
        img = Image.open(f)
        path = f"temp_{f.name}"
        img.save(path)

        pred = cnn_predict(path)
        severity, mask = calculate_severity(path)

        # 🔥 FINAL DECISION LOGIC (FIXED)
        if severity < 0.5 and pred < 0.7:
            decision = "No Crack ✅"
        elif severity >= 0.5:
            decision = "Crack Detected ⚠️"
        elif pred >= 0.7:
            decision = "Crack Detected ⚠️"
        else:
            decision = "No Crack ✅"

        # Category
        if severity < 0.5:
            category = "Low 🟢"
        elif severity < 10:
            category = "Medium 🟡"
        else:
            category = "High 🔴"

        results.append({
            "Image Name": f.name,
            "Prediction": decision,
            "Severity (%)": severity,
            "Category": category,
            "Original": img,
            "Overlay": overlay_crack(path, mask)
        })

    # ---------------- RESULTS TABLE ----------------
    st.subheader("📋 Detection Results")
    df = pd.DataFrame(results)[["Image Name","Prediction","Severity (%)","Category"]]
    st.dataframe(df)

    # ---------------- IMAGE DISPLAY ----------------
    st.subheader("📷 Image Overlays")
    for r in results:
        st.markdown(f"**{r['Image Name']} — {r['Prediction']} | Severity: {r['Severity (%)']}% ({r['Category']})**")
        c1, c2 = st.columns(2)
        c1.image(r["Original"], caption="Original", use_column_width=True)
        c2.image(to_rgb(r["Overlay"]), caption="Crack Highlighted", use_column_width=True)

    st.success("✅ Crack detection completed accurately")
