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

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="🛠️ Crack Detection Dashboard", layout="wide")
st.title("🛠️ Image-based Crack Detection with Dashboard")
st.markdown(
    "Upload one or more images to detect cracks, calculate severity, and download a report."
)

# -------------------------
# Download model
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)

# -------------------------
# Functions
def calculate_crack_severity(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    crack_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    severity_score = (crack_pixels / total_pixels) * 100
    return round(severity_score, 2), thresh

def predict_crack(image_path):
    img = Image.open(image_path).convert("RGB")
    input_shape = model.input_shape
    height, width = input_shape[1], input_shape[2]
    img = img.resize((width, height))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return float(pred[0][0])

def create_pdf_report_batch(image_paths, severities, categories, report_path="report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Crack Detection Report", ln=True, align="C")
    pdf.ln(10)

    for img_path, severity, category in zip(image_paths, severities, categories):
        category_text = category.replace("🟢", "Low").replace("🟡", "Medium").replace("🔴", "High")
        pdf.cell(200, 10, txt=f"Severity Score: {severity}% ({category_text})", ln=True)
        pdf.ln(5)
        pdf.image(img_path, x=50, w=100)
        pdf.ln(10)
    pdf.output(report_path)
    return report_path

def overlay_cracks(image_path, thresh):
    img = cv2.imread(image_path)
    overlay = img.copy()
    # Slightly thicker red overlay for subtle cracks
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    overlay[dilated==255] = [0,0,255]  # Red overlay for cracks
    combined = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    return combined

# -------------------------
# Streamlit UI
uploaded_files = st.file_uploader(
    "Upload one or more images", type=["jpg","png","jpeg"], accept_multiple_files=True
)

if uploaded_files:
    results = []
    temp_image_paths = []
    severities = []
    categories = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        temp_path = f"temp_{uploaded_file.name}"
        image.save(temp_path)
        temp_image_paths.append(temp_path)

        # Prediction
        prediction = predict_crack(temp_path)
        severity, thresh = calculate_crack_severity(temp_path)

        # Adjust detection: small cracks now detected
        if prediction >= 0.5 or severity > 2.0:
            detect_text = "Crack Detected ⚠️"
        else:
            detect_text = "No Crack ✅"

        # Categorize severity
        if severity <= 10:
            sev_category = "Low 🟢"
        elif severity <= 30:
            sev_category = "Medium 🟡"
        else:
            sev_category = "High 🔴"

        severities.append(severity)
        categories.append(sev_category)

        # Overlay cracks
        overlay_img = overlay_cracks(temp_path, thresh)
        overlay_path = f"overlay_{uploaded_file.name}"
        cv2.imwrite(overlay_path, overlay_img)

        results.append({
            "Image": uploaded_file.name,
            "Prediction": detect_text,
            "Severity": f"{severity}%",
            "Category": sev_category,
            "Overlay": overlay_path
        })

    # -------------------------
    # Display results in dashboard
    st.subheader("📝 Results Dashboard")
    for res in results:
        st.write(f"**{res['Image']}** - {res['Prediction']} - Severity: {res['Severity']} ({res['Category']})")
        st.image(res['Overlay'], width=400)
        st.progress(min(int(float(res['Severity'].replace('%',''))), 100))

    # -------------------------
    # Download combined PDF report
    report_file = create_pdf_report_batch(temp_image_paths, severities, categories)
    with open(report_file, "rb") as f:
        st.download_button(
            label="📄 Download Combined Report",
            data=f,
            file_name="Crack_Report.pdf",
            mime="application/pdf"
        )

    st.success("✅ Analysis Complete!")
