import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model
import pandas as pd
from gtts import gTTS
from fpdf import FPDF
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# PAGE CONFIG
st.set_page_config(page_title="Concrete Crack Detection", layout="wide")
st.title("🧠 AI-Based Concrete Crack Detection System")

# -------------------------------
# MODEL DOWNLOAD & LOAD
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "crack_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading CNN model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)

# -------------------------------
# FUNCTIONS
def cnn_predict(img_path):
    img = Image.open(img_path).convert("RGB").resize((150,150))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    return float(model.predict(arr)[0][0])

def calculate_severity(img_path, threshold=127):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)
    severity = (np.sum(thresh == 255) / thresh.size) * 100
    return round(severity,2), thresh

def overlay_crack(img_path, mask):
    img = cv2.imread(img_path)
    overlay = img.copy()
    overlay[mask == 255] = [0,0,255]
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

def speak(text):
    tts = gTTS(text=text)
    tts.save("result.mp3")
    return "result.mp3"

def generate_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0,10,"Concrete Crack Detection Report", ln=True, align="C")
    pdf.ln(5)

    for _, row in df.iterrows():
        pdf.multi_cell(0,8,
            f"Image: {row['Image']}\n"
            f"Result: {row['Result']}\n"
            f"CNN Score: {row['CNN Score']}\n"
            f"Severity (%): {row['Severity (%)']}\n"
            f"Severity Level: {row['Severity Level']}\n"
            f"Recommendation: {row['Recommendation']}\n"
            "----------------------------------------"
        )

    pdf.output("crack_report.pdf")
    return "crack_report.pdf"

# -------------------------------
# USER INPUT
threshold = st.slider("Binary Threshold", 80, 200, 127)
files = st.file_uploader("Upload Crack Images", type=["jpg","png","jpeg"], accept_multiple_files=True)

results = []

# -------------------------------
# PROCESS IMAGES
if files:
    for file in files:
        img = Image.open(file)
        path = f"temp_{file.name}"
        img.save(path)

        cnn_score = cnn_predict(path)
        severity, mask = calculate_severity(path, threshold)

        # -------------------------------
        # HARD GATE (FIXED LOGIC)
        if cnn_score < 0.75 or severity < 0.3:
            decision = "No Crack"
            severity_level = "None"
            recommendation = "Structure is safe"
        else:
            decision = "Crack Detected"

            if severity < 1:
                severity_level = "Low"
                recommendation = "Monitor periodically"
            elif severity < 5:
                severity_level = "Medium"
                recommendation = "Repair recommended"
            else:
                severity_level = "High"
                recommendation = "Immediate maintenance required"

        audio = speak(decision)
        overlay = overlay_crack(path, mask)

        # SAVE RESULT
        results.append({
            "Image": file.name,
            "Result": decision,
            "CNN Score": round(cnn_score,2),
            "Severity (%)": severity,
            "Severity Level": severity_level,
            "Recommendation": recommendation
        })

        # DISPLAY
        st.subheader(file.name)
        col1, col2 = st.columns(2)
        col1.image(img, caption="Original Image", use_column_width=True)
        col2.image(overlay, caption="Crack Highlighted", use_column_width=True)

        st.success(f"Result: {decision}")
        st.info(f"Severity Level: {severity_level}")
        st.warning(f"Recommendation: {recommendation}")
        st.audio(audio)
        st.divider()

# -------------------------------
# SUMMARY & DOWNLOADS
if results:
    df = pd.DataFrame(results)
    st.subheader("📊 Detection Summary")
    st.dataframe(df)

    st.download_button(
        "📥 Download CSV Report",
        df.to_csv(index=False).encode("utf-8"),
        "crack_report.csv",
        "text/csv"
    )

    pdf_file = generate_pdf(df)
    with open(pdf_file, "rb") as f:
        st.download_button(
            "📄 Download PDF Report",
            f,
            "crack_report.pdf",
            "application/pdf"
        )
