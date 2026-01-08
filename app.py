import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gtts import gTTS
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
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)

# -------------------------
# FUNCTIONS
def cnn_predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((150,150))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    return float(model.predict(arr)[0][0])

def crack_severity(img_path, thresh_val=127):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)
    severity = (np.sum(thresh==255) / thresh.size) * 100
    return round(severity,3), thresh

def overlay_crack(img_path, thresh):
    img = cv2.imread(img_path)
    overlay = img.copy()
    overlay[thresh==255] = [0,0,255]
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

def speak(text):
    tts = gTTS(text=text)
    tts.save("voice.mp3")
    return "voice.mp3"

# -------------------------
# UPLOAD
threshold_val = st.slider("Binary Threshold", 80, 200, 127)
uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg","png","jpeg"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file)
        path = f"temp_{file.name}"
        img.save(path)

        cnn_score = cnn_predict(path)
        severity, thresh = crack_severity(path, threshold_val)

        # -------------------------
        # FINAL DECISION LOGIC
        if severity < 0.3:
            decision = "No Crack ✅"
        elif severity < 1.5:
            decision = "Crack Detected ⚠️" if cnn_score >= 0.65 else "No Crack ✅"
        else:
            decision = "Crack Detected ⚠️"

        audio = speak(decision.replace("⚠️","").replace("✅",""))
        overlay = overlay_crack(path, thresh)

        # -------------------------
        # DISPLAY
        st.subheader(f"📷 {file.name}")
        col1, col2 = st.columns(2)

        col1.image(img, caption="Original Image", use_column_width=True)
        col2.image(overlay, caption="Detected Cracks", use_column_width=True)

        st.markdown(f"### 🧪 Result: **{decision}**")
        st.write(f"CNN Confidence: `{cnn_score:.2f}`")
        st.write(f"Crack Severity: `{severity}%`")

        st.audio(audio)

        st.divider()
