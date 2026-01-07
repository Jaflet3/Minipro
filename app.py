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
import seaborn as sns
from gtts import gTTS
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# Streamlit page setup
st.set_page_config(page_title="🛠️ Crack Detection Dashboard", layout="wide")
st.title("🛠️ Image-based Crack Detection ")
st.markdown(
    "Upload images to detect cracks, calculate severity, overlay heatmaps, get voice alerts, and download reports."
)

# -------------------------
# Utility functions
def download_model(url: str, path: str):
    if not os.path.exists(path):
        with st.spinner("📥 Downloading model..."):
            gdown.download(url, path, quiet=False)
    return path

def load_crack_model(path: str):
    return load_model(path, compile=False)

def predict_crack(image_path: str, model) -> float:
    img = Image.open(image_path).convert("RGB")
    h, w = model.input_shape[1], model.input_shape[2]
    img = img.resize((w, h))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    return float(model.predict(arr)[0][0])

def calculate_crack_severity(image_path: str, threshold_val=127):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY_INV)
    severity = (np.sum(thresh == 255) / thresh.size) * 100
    return round(severity, 2), thresh

def crack_count_length(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_length = sum(cv2.arcLength(c, False) for c in contours)
    return len(contours), round(total_length, 2)

def overlay_cracks(image_path: str, thresh):
    img = cv2.imread(image_path)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    overlay = img.copy()
    overlay[dilated==255] = [0,0,255]
    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

def overlay_heatmap(image_path: str, thresh):
    img = cv2.imread(image_path)
    heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)

def cv2_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def speak(text: str, filename="detection.mp3") -> str:
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

def generate_pdf_report(images, severities, categories, counts, lengths, detections, path="report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Summary page
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Crack Detection Report - Summary", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, f"Total Images: {len(images)}", ln=True)
    pdf.cell(200, 8, f"Average Severity: {round(sum(severities)/len(images),2)}%", ln=True)
    pdf.ln(5)

    for idx, img in enumerate(images):
        temp_path = f"pdf_temp_{idx}.jpg"
        img.save(temp_path)
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, f"Image {idx+1}", ln=True)
        pdf.set_font("Arial", size=12)

        category_text = categories[idx].replace("🟢","Low").replace("🟡","Medium").replace("🔴","High")
        detect_text_pdf = detections[idx].replace("⚠️","Crack Detected").replace("✅","No Crack")

        pdf.cell(200, 8, f"Severity: {severities[idx]}% ({category_text})", ln=True)
        pdf.cell(200, 8, f"Detection: {detect_text_pdf}", ln=True)
        pdf.cell(200, 8, f"Crack Count: {counts[idx]}, Total Length: {lengths[idx]}", ln=True)
        pdf.ln(5)
        pdf.image(temp_path, x=50, w=100)
        os.remove(temp_path)

    pdf.output(path)
    return path

# -------------------------
# Load model
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = download_model(MODEL_URL, "model.h5")
model = load_crack_model(MODEL_PATH)

# -------------------------
# Streamlit UI
threshold_val = st.slider("Adjust binary threshold for crack detection:", 50, 200, 127)
uploaded_files = st.file_uploader("Upload one or more images", type=["jpg","png","jpeg"], accept_multiple_files=True)

if uploaded_files:
    results = []

    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        temp_path = f"temp_{uploaded_file.name}"
        img.save(temp_path)

        # Predictions
        pred = predict_crack(temp_path, model)
        severity, thresh = calculate_crack_severity(temp_path, threshold_val)
        count, length = crack_count_length(thresh)
        detect_text = "Crack Detected ⚠️" if pred >=0.5 or severity >2 else "No Crack ✅"

        # Voice feedback
        audio_file = speak(detect_text.replace("⚠️","").replace("✅",""))
        st.audio(audio_file, format='audio/mp3')

        # Severity category
        if severity <= 10:
            category = "Low 🟢"
        elif severity <= 30:
            category = "Medium 🟡"
        else:
            category = "High 🔴"

        overlay_img = overlay_cracks(temp_path, thresh)
        heatmap_img = overlay_heatmap(temp_path, thresh)

        results.append({
            "Image": img,
            "Image_Name": uploaded_file.name,
            "Prediction": detect_text,
            "Severity": severity,
            "Category": category,
            "Crack Count": count,
            "Total Length": length,
            "Overlay": overlay_img,
            "Heatmap": heatmap_img
        })

    # -------------------------
    # Display results table
    st.subheader("📝 Results Dashboard")
    df_display = pd.DataFrame(results)[['Image_Name','Prediction','Severity','Category','Crack Count','Total Length']]
    st.dataframe(df_display)

    # Display image overlays
    st.subheader("📷 Image Overlays")
    for res in results:
        st.write(f"**{res['Image_Name']}** - {res['Prediction']} - Severity: {res['Severity']}% ({res['Category']})")
        col1, col2, col3 = st.columns(3)
        col1.image(res['Image'], caption="Original", use_column_width=True)
        col2.image(cv2_to_rgb(res['Overlay']), caption="Overlay", use_column_width=True)
        col3.image(cv2_to_rgb(res['Heatmap']), caption="Heatmap", use_column_width=True)

    # -------------------------
    # Improved severity charts
    st.subheader("📊 Severity Analysis")
    severity_values = [r['Severity'] for r in results]
    image_names = [r['Image_Name'] for r in results]
    categories_clean = [r['Category'].replace("🟢","Low").replace("🟡","Medium").replace("🔴","High") for r in results]

    fig, ax = plt.subplots(1,2, figsize=(14,5))

    # Bar chart
    sns.barplot(x=image_names, y=severity_values, palette="Oranges", ax=ax[0])
    ax[0].set_ylabel("Severity (%)")
    ax[0].set_xlabel("Image Name")
    ax[0].set_title("Severity per Image")
    ax[0].set_ylim(0, max(severity_values)*1.2 if severity_values else 10)
    for i, v in enumerate(severity_values):
        ax[0].text(i, v + 0.1, f"{v:.2f}%", ha='center', fontweight='bold')

    # Pie chart
    cat_counts = pd.Series(categories_clean).value_counts()
    colors = sns.color_palette("pastel")[0:len(cat_counts)]
    ax[1].pie(cat_counts, labels=[f"{c} ({cat_counts[c]})" for c in cat_counts.index],
              autopct='%1.1f%%', startangle=90, colors=colors)
    ax[1].set_title("Severity Category Distribution")
    st.pyplot(fig)

    # -------------------------
    # CSV download
    df_csv = pd.DataFrame(results)[['Image_Name','Prediction','Severity','Category','Crack Count','Total Length']]
    st.download_button(
        label="📊 Download CSV Report",
        data=df_csv.to_csv(index=False).encode('utf-8'),
        file_name="Crack_Report.csv",
        mime="text/csv"
    )

    # -------------------------
    # PDF download
    report_file = generate_pdf_report(
        [r['Image'] for r in results],
        [r['Severity'] for r in results],
        [r['Category'] for r in results],
        [r['Crack Count'] for r in results],
        [r['Total Length'] for r in results],
        [r['Prediction'] for r in results]
    )
    with open(report_file, "rb") as f:
        st.download_button(
            label="📄 Download PDF Report",
            data=f,
            file_name="Crack_Report.pdf",
            mime="application/pdf"
        )

    st.success("✅ Analysis Complete! Voice alerts played for each image.")
