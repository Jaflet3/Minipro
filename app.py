import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model
from fpdf import FPDF
import pandas as pd
import matplotlib.pyplot as plt
from gtts import gTTS
import warnings

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🛠️ Crack Detection Dashboard", layout="wide")
st.title("🛠️ AI-Based Concrete Crack Detection")
st.markdown("Upload images to accurately detect concrete cracks with reduced false detection.")

# ---------------- MODEL SETUP ----------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "crack_model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH, compile=False)

model = download_model()

# ---------------- UTILITY FUNCTIONS ----------------
def predict_crack(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((150,150))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    return float(model.predict(arr)[0][0])

def calculate_severity(image_path, thresh_val):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    crack_pixels = np.sum(thresh == 255)
    severity = (crack_pixels / thresh.size) * 100
    return round(severity,2), thresh

def overlay_cracks(image_path, thresh):
    img = cv2.imread(image_path)
    overlay = img.copy()
    overlay[thresh==255] = [0,0,255]
    return cv2.addWeighted(img,0.7,overlay,0.3,0)

def overlay_heatmap(image_path, thresh):
    img = cv2.imread(image_path)
    heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
    return cv2.addWeighted(img,0.6,heatmap,0.4,0)

def speak(text):
    tts = gTTS(text=text, lang="en")
    tts.save("alert.mp3")
    return "alert.mp3"

# ---------------- STREAMLIT UI ----------------
threshold_val = st.slider("Binary threshold (crack extraction)", 50, 200, 130)
uploaded_files = st.file_uploader("Upload image(s)", type=["jpg","png","jpeg"], accept_multiple_files=True)

results = []

if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file)
        temp_path = f"temp_{file.name}"
        img.save(temp_path)

        pred = predict_crack(temp_path)
        severity, thresh = calculate_severity(temp_path, threshold_val)

        # ✅ FINAL DECISION LOGIC (FIXED)
        if severity < 1:
            decision = "No Crack ✅"
        elif pred >= 0.75 and severity >= 3:
            decision = "Crack Detected ⚠️"
        else:
            decision = "No Crack ✅"

        # Severity category
        if severity < 5:
            category = "Low 🟢"
        elif severity < 20:
            category = "Medium 🟡"
        else:
            category = "High 🔴"

        # Voice alert
        audio = speak(decision.replace("⚠️","").replace("✅",""))
        st.audio(audio)

        overlay = overlay_cracks(temp_path, thresh)
        heatmap = overlay_heatmap(temp_path, thresh)

        results.append({
            "Image Name": file.name,
            "Prediction": decision,
            "Severity (%)": severity,
            "Category": category,
            "Original": img,
            "Overlay": overlay,
            "Heatmap": heatmap
        })

    # ---------------- RESULTS TABLE ----------------
    st.subheader("📋 Detection Results")
    df = pd.DataFrame(results)[["Image Name","Prediction","Severity (%)","Category"]]
    st.dataframe(df)

    # ---------------- IMAGE DISPLAY ----------------
    st.subheader("📷 Image Overlays")
    for r in results:
        st.markdown(f"**{r['Image Name']}** — {r['Prediction']} | Severity: {r['Severity (%)']}% ({r['Category']})")
        c1,c2,c3 = st.columns(3)
        c1.image(r["Original"], caption="Original", use_column_width=True)
        c2.image(cv2.cvtColor(r["Overlay"], cv2.COLOR_BGR2RGB), caption="Crack Overlay", use_column_width=True)
        c3.image(cv2.cvtColor(r["Heatmap"], cv2.COLOR_BGR2RGB), caption="Heatmap", use_column_width=True)

    # ---------------- SEVERITY CHART ----------------
    st.subheader("📊 Severity Analysis")
    fig, ax = plt.subplots()
    ax.bar(df["Image Name"], df["Severity (%)"])
    ax.set_ylabel("Severity (%)")
    ax.set_xlabel("Image")
    ax.set_title("Crack Severity per Image")
    st.pyplot(fig)

    st.success("✅ Crack detection completed accurately!")
