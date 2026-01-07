# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model
from fpdf import FPDF
import warnings

# -------------------------
# Ignore TensorFlow compile warning
# -------------------------
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Streamlit UI setup
# -------------------------
st.set_page_config(page_title="🛠️ Crack Detection", layout="centered")
st.title("🛠️ Image-based Crack Detection")
st.markdown(
    """
    Upload an image to detect cracks, calculate severity, and download a PDF report.
    """
)

# -------------------------
# 1️⃣ Download model from Google Drive
# -------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the trained model
with st.spinner("🔄 Loading model..."):
    model = load_model(MODEL_PATH, compile=False)  # compile=False avoids warning

# -------------------------
# 2️⃣ Functions
# -------------------------
def calculate_crack_severity(image_path):
    """Calculate crack severity as % of image area covered by cracks."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    crack_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    severity_score = (crack_pixels / total_pixels) * 100
    return round(severity_score, 2)


def create_pdf_report(image_path, severity_score, report_path="report.pdf"):
    """Generate a PDF report with the image and severity score."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Crack Detection Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Severity Score: {severity_score}%", ln=True)
    pdf.ln(10)
    pdf.image(image_path, x=50, w=100)
    pdf.output(report_path)
    return report_path


def predict_crack(image_path):
    """Use the trained model to predict crack presence."""
    img = Image.open(image_path).convert("RGB")  # ensure 3 channels

    # Get model input shape dynamically
    input_shape = model.input_shape  # e.g., (None, 224, 224, 3)
    height, width = input_shape[1], input_shape[2]

    img = img.resize((width, height))
    img_array = np.array(img) / 255.0

    # Ensure batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, C)

    # Predict
    pred = model.predict(img_array)
    return float(pred[0][0])  # return as float


# -------------------------
# 3️⃣ Streamlit main app
# -------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    temp_path = "temp_image.png"
    image.save(temp_path)

    with st.spinner("🔍 Detecting cracks..."):
        prediction = predict_crack(temp_path)

    if prediction >= 0.5:
        st.error("⚠️ Crack Detected")
    else:
        st.success("✅ No Crack Detected")

    severity = calculate_crack_severity(temp_path)
    st.metric(label="Crack Severity Score", value=f"{severity}%")

    report_file = create_pdf_report(temp_path, severity)
    with open(report_file, "rb") as f:
        st.download_button(
            label="📄 Download Report",
            data=f,
            file_name="Crack_Report.pdf",
            mime="application/pdf",
        )

    st.success("Analysis complete!")
