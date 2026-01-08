import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from gtts import gTTS
import warnings

warnings.filterwarnings("ignore")

# -------------------------------------------------
# Streamlit page setup
st.set_page_config(page_title="Crack Detection System", layout="wide")
st.title("🧠 AI-based Concrete Crack Detection")
st.markdown("Upload images to automatically detect structural cracks using Deep Learning.")

# -------------------------------------------------
# Utility functions
def download_model(url, path):
    if not os.path.exists(path):
        with st.spinner("📥 Downloading model..."):
            gdown.download(url, path, quiet=False)
    return path

def load_crack_model(path):
    return load_model(path, compile=False)

def predict_crack(image_path, model):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((150, 150))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    return float(model.predict(arr)[0][0])

def calculate_crack_severity(image_path, threshold_val):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5,5), 0)
    _, thresh = cv2.threshold(img_blur, threshold_val, 255, cv2.THRESH_BINARY_INV)
    severity = (np.sum(thresh == 255) / thresh.size) * 100
    return round(severity, 2), thresh

def overlay_cracks(image_path, thresh):
    img = cv2.imread(image_path)
    overlay = img.copy()
    overlay[thresh == 255] = [0, 0, 255]
    return cv2.cvtColor(cv2.addWeighted(img, 0.7, overlay, 0.3, 0), cv2.COLOR_BGR2RGB)

def speak(text):
    tts = gTTS(text=text, lang="en")
    tts.save("voice.mp3")
    return "voice.mp3"

# -------------------------------------------------
# Load model
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = download_model(MODEL_URL, "crack_model.h5")
model = load_crack_model(MODEL_PATH)

# -------------------------------------------------
# UI Controls
threshold_val = st.slider("Binary threshold for crack detection", 80, 180, 130)
uploaded_files = st.file_uploader(
    "Upload crack images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -------------------------------------------------
# Processing
if uploaded_files:
    results = []

    for file in uploaded_files:
        img = Image.open(file)
        temp_path = "temp.jpg"
        img.save(temp_path)

        # Predictions
        pred = predict_crack(temp_path, model)
        severity, thresh = calculate_crack_severity(temp_path, threshold_val)

        # FINAL CORRECT DECISION LOGIC
        if pred >= 0.7 and severity > 5:
            result = "Crack Detected ⚠️"
        else:
            result = "No Crack ✅"

        # Severity level
        if severity <= 10:
            level = "Low"
        elif severity <= 30:
            level = "Medium"
        else:
            level = "High"

        overlay = overlay_cracks(temp_path, thresh)

        results.append({
            "Image": img,
            "Image_Name": file.name,
            "Prediction": result,
            "CNN Score": round(pred, 3),
            "Severity (%)": severity,
            "Level": level,
            "Overlay": overlay
        })

        # Voice alert
        audio = speak(result.replace("⚠️","").replace("✅",""))
        st.audio(audio)

    # -------------------------------------------------
    # Results Table
    st.subheader("📋 Detection Results")
    df = pd.DataFrame(results)[[
        "Image_Name", "Prediction", "CNN Score", "Severity (%)", "Level"
    ]]
    st.dataframe(df)

    # -------------------------------------------------
    # Display images
    st.subheader("📷 Crack Visualization")
    for r in results:
        st.markdown(f"### {r['Image_Name']}")
        st.write(f"**{r['Prediction']}** | CNN Score: `{r['CNN Score']}` | Severity: `{r['Severity (%)']}%`")
        col1, col2 = st.columns(2)
        col1.image(r["Image"], caption="Original", use_column_width=True)
        col2.image(r["Overlay"], caption="Crack Overlay", use_column_width=True)

    # -------------------------------------------------
    # Severity Chart
    st.subheader("📊 Severity Analysis")
    fig, ax = plt.subplots()
    ax.bar(df["Image_Name"], df["Severity (%)"])
    ax.set_ylabel("Severity (%)")
    ax.set_xlabel("Image Name")
    ax.set_title("Crack Severity per Image")
    st.pyplot(fig)

    st.success("✅ Crack detection completed successfully!")
