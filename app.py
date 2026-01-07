# app.py
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
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="🛠️ Crack Detection Dashboard", layout="wide")
st.title("🛠️ Enhanced Image-based Crack Detection Dashboard")
st.markdown(
    "Upload images to detect cracks, calculate severity, length, and download a detailed report."
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
def calculate_crack_severity(image_path, threshold_val=127):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY_INV)
    crack_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    severity_score = (crack_pixels / total_pixels) * 100
    return round(severity_score, 2), thresh

def crack_length_count(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_length = sum([cv2.arcLength(c, False) for c in contours])
    return len(contours), round(total_length, 2)

def predict_crack(image_path):
    img = Image.open(image_path).convert("RGB")
    input_shape = model.input_shape
    height, width = input_shape[1], input_shape[2]
    img = img.resize((width, height))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return float(pred[0][0])

def overlay_cracks(image_path, thresh):
    img = cv2.imread(image_path)
    overlay = img.copy()
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    overlay[dilated==255] = [0,0,255]  # Red overlay
    combined = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    return combined

def overlay_heatmap(image_path, thresh):
    img = cv2.imread(image_path)
    heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
    combined = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
    return combined

# Convert OpenCV BGR -> RGB
def cv2_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------
# PDF Report
def create_pdf_report_batch(images, severities, categories, counts, lengths, detections, report_path="report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Summary page
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Crack Detection Report - Summary", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    total_images = len(images)
    avg_severity = round(sum(severities)/total_images, 2)
    pdf.cell(200, 8, txt=f"Total Images: {total_images}", ln=True)
    pdf.cell(200, 8, txt=f"Average Severity: {avg_severity}%", ln=True)
    pdf.ln(5)

    # Per image details
    for idx, img in enumerate(images):
        temp_path = f"pdf_temp_{idx}.jpg"
        img.save(temp_path)
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=f"Image {idx+1}", ln=True)
        pdf.set_font("Arial", size=12)

        # Replace emojis for PDF
        category_text = categories[idx].replace("🟢", "Low").replace("🟡", "Medium").replace("🔴", "High")
        detect_text_pdf = detections[idx].replace("⚠️", "Crack Detected").replace("✅", "No Crack")

        pdf.cell(200, 8, txt=f"Severity: {severities[idx]}% ({category_text})", ln=True)
        pdf.cell(200, 8, txt=f"Detection: {detect_text_pdf}", ln=True)
        pdf.cell(200, 8, txt=f"Crack Count: {counts[idx]}, Total Length: {lengths[idx]}", ln=True)
        pdf.ln(5)
        pdf.image(temp_path, x=50, w=100)
        os.remove(temp_path)

    pdf.output(report_path)
    return report_path

# -------------------------
# Streamlit UI
uploaded_files = st.file_uploader(
    "Upload one or more images", type=["jpg","png","jpeg"], accept_multiple_files=True
)

threshold_val = st.slider("Adjust binary threshold for crack detection:", 50, 200, 127)

if uploaded_files:
    results = []
    temp_image_paths = []
    severities = []
    categories = []
    crack_counts = []
    crack_lengths = []
    detections = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        temp_path = f"temp_{uploaded_file.name}"
        image.save(temp_path)
        temp_image_paths.append(temp_path)

        # Prediction + severity
        prediction = predict_crack(temp_path)
        severity, thresh = calculate_crack_severity(temp_path, threshold_val)
        count, total_length = crack_length_count(thresh)

        # Detection text
        detect_text = "Crack Detected ⚠️" if prediction >= 0.5 or severity > 2.0 else "No Crack ✅"

        # Categorize severity
        if severity <= 10:
            sev_category = "Low 🟢"
        elif severity <= 30:
            sev_category = "Medium 🟡"
        else:
            sev_category = "High 🔴"

        severities.append(severity)
        categories.append(sev_category)
        crack_counts.append(count)
        crack_lengths.append(total_length)
        detections.append(detect_text)

        # Overlays
        overlay_img = overlay_cracks(temp_path, thresh)
        heatmap_img = overlay_heatmap(temp_path, thresh)

        results.append({
            "Image": image,  # PIL Image
            "Image_Name": uploaded_file.name,
            "Prediction": detect_text,
            "Severity": severity,
            "Category": sev_category,
            "Crack Count": count,
            "Total Length": total_length,
            "Overlay": overlay_img,
            "Heatmap": heatmap_img
        })

    # -------------------------
    # Display interactive table
    st.subheader("📝 Results Dashboard")
    df_display = pd.DataFrame(results)[['Image_Name','Prediction','Severity','Category','Crack Count','Total Length']]
    st.dataframe(df_display)

    # Display images side by side
    st.subheader("📷 Image Overlays")
    for res in results:
        st.write(f"**{res['Image_Name']}** - {res['Prediction']} - Severity: {res['Severity']}% ({res['Category']})")
        col1, col2, col3 = st.columns(3)
        col1.image(res['Image'], caption="Original", use_column_width=True)
        col2.image(cv2_to_rgb(res['Overlay']), caption="Overlay", use_column_width=True)
        col3.image(cv2_to_rgb(res['Heatmap']), caption="Heatmap", use_column_width=True)

    # -------------------------
    # Severity distribution chart
    st.subheader("📊 Severity Analysis")
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].bar([r['Image_Name'] for r in results], severities, color='orange')
    ax[0].set_ylabel("Severity %")
    ax[0].set_xticklabels([r['Image_Name'] for r in results], rotation=45, ha='right')
    ax[0].set_title("Severity per Image")

    cat_counts = pd.Series(categories).value_counts()
    ax[1].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=140)
    ax[1].set_title("Severity Category Distribution")
    st.pyplot(fig)

    # -------------------------
    # Download CSV
    df_csv = pd.DataFrame(results)[['Image_Name','Prediction','Severity','Category','Crack Count','Total Length']]
    csv = df_csv.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📊 Download CSV Report",
        data=csv,
        file_name="Crack_Report.csv",
        mime="text/csv"
    )

    # -------------------------
    # Download PDF report
    report_file = create_pdf_report_batch(
        [r['Image'] for r in results],
        severities,
        categories,
        crack_counts,
        crack_lengths,
        detections
    )
    with open(report_file, "rb") as f:
        st.download_button(
            label="📄 Download PDF Report",
            data=f,
            file_name="Crack_Report.pdf",
            mime="application/pdf"
        )

    st.success("✅ Analysis Complete!")
