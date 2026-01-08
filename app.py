import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crack Detection", layout="wide")
st.title("🧠 AI-Based Crack Detection System")

# ---------------- MODEL ----------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "crack_model.h5"

@st.cache_resource
def load_model_safe():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH, compile=False)

model = load_model_safe()

# ---------------- FUNCTIONS ----------------
def cnn_predict(path):
    img = Image.open(path).convert("RGB").resize((150,150))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    return float(model.predict(arr)[0][0])

def crack_severity(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    severity = (np.sum(thresh==255)/thresh.size)*100
    return round(severity,2), thresh

def overlay(path, mask):
    img = cv2.imread(path)
    overlay = img.copy()
    overlay[mask==255] = [0,0,255]
    return cv2.addWeighted(img,0.7,overlay,0.3,0)

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------- UPLOAD ----------------
files = st.file_uploader("Upload images", type=["jpg","png","jpeg"], accept_multiple_files=True)

results = []

if files:
    for f in files:
        img = Image.open(f)
        path = f"temp_{f.name}"
        img.save(path)

        pred = cnn_predict(path)
        severity, mask = crack_severity(path)

        # 🔥 FINAL CORRECT DECISION
        if severity >= 2.0:
            decision = "Crack Detected ⚠️"
        elif pred >= 0.6:
            decision = "Crack Detected ⚠️"
        else:
            decision = "No Crack ✅"

        # Category
        if severity < 2:
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
            "Image": img,
            "Overlay": overlay(path, mask)
        })

    # ---------------- TABLE ----------------
    st.subheader("📋 Detection Results")
    df = pd.DataFrame(results)[["Image Name","Prediction","Severity (%)","Category"]]
    st.dataframe(df)

    # ---------------- IMAGES ----------------
    st.subheader("📷 Image Overlays")
    for r in results:
        st.markdown(f"**{r['Image Name']} — {r['Prediction']} | Severity: {r['Severity (%)']}% ({r['Category']})**")
        c1,c2 = st.columns(2)
        c1.image(r["Image"], caption="Original", use_column_width=True)
        c2.image(to_rgb(r["Overlay"]), caption="Crack Highlighted", use_column_width=True)

    st.success("✅ Detection completed successfully")
