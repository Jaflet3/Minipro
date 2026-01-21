import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import os
import warnings
import io

# Hugging Face Hub
from huggingface_hub import hf_hub_download

# -----------------------------
# SUPPRESS WARNINGS & TF LOGS
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -----------------------------
# PAGE CONFIG
st.set_page_config(
    page_title="Concrete Crack Detection",
    layout="wide",
    page_icon="🛠️"
)

st.title("🛠️ Concrete Crack Detection System")
st.caption("Hybrid CNN + Image Processing based Structural Health Monitoring")
st.divider()

# -----------------------------
# LOAD MODEL FROM HUGGING FACE
MODEL_PATH = "crack_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading CNN model from Hugging Face..."):
        # Replace 'your-username/model-repo' with your Hugging Face repo and filename
        MODEL_PATH = hf_hub_download(
            repo_id="your-username/model-repo",
            filename="crack_model.h5"
        )

model = load_model(MODEL_PATH, compile=False)

# -----------------------------
# FUNCTIONS
def cnn_predict(pil_image):
    img = pil_image.convert("RGB").resize((150, 150))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    return float(model.predict(arr, verbose=0)[0][0])

def crack_severity(opencv_img):
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    crack_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    severity = (crack_pixels / total_pixels) * 100
    return round(severity, 3), thresh

def edge_ratio(opencv_img):
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return round(np.sum(edges > 0) / edges.size, 4)

def overlay_crack(opencv_img, thresh):
    overlay = opencv_img.copy()
    overlay[thresh == 255] = [0, 0, 255]  # red overlay
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# -----------------------------
# UPLOAD IMAGE
uploaded_file = st.file_uploader(
    "📤 Upload Concrete Surface Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(image_bytes))
    opencv_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # -----------------------------
    # PREDICTIONS
    cnn_score = cnn_predict(pil_img)
    severity, thresh = crack_severity(opencv_img)
    edge_val = edge_ratio(opencv_img)

    # -----------------------------
    # FINAL DECISION LOGIC
    if severity < 0.2:
        decision = "No Crack"
        severity_level = "None"
        recommendation = "Structure is safe"
        show_overlay = False
    elif cnn_score < 0.65 and edge_val < 0.01:
        decision = "No Crack"
        severity_level = "None"
        recommendation = "Structure is safe"
        show_overlay = False
    else:
        decision = "Crack Detected"
        show_overlay = True
        if severity < 1.5:
            severity_level = "Low"
            recommendation = "Monitor periodically"
        elif severity < 5:
            severity_level = "Medium"
            recommendation = "Repair recommended"
        else:
            severity_level = "High"
            recommendation = "Immediate maintenance required"

    # -----------------------------
    # IMAGE DISPLAY
    col1, col2 = st.columns(2)
    col1.image(pil_img, caption="Original Image", use_column_width=True)
    if show_overlay:
        overlay_img = overlay_crack(opencv_img, thresh)
        col2.image(overlay_img, caption="Detected Crack Area", use_column_width=True)
    else:
        col2.image(pil_img, caption="No Crack Found", use_column_width=True)

    st.divider()

    # -----------------------------
    # ANALYSIS DASHBOARD
    st.subheader("📊 Analysis Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("CNN Confidence", f"{round(cnn_score*100, 2)}%", "High" if cnn_score > 0.8 else "Moderate")
    m2.metric("Crack Area", f"{severity} %")
    m3.metric("Edge Density", edge_val)

    # -----------------------------
    # RESULT MESSAGE
    if decision == "Crack Detected":
        if severity == 0.0:
            st.warning("⚠️ Result: Micro Crack Detected (CNN-based)")
            st.caption("CNN detected texture-based micro cracks not measurable using pixel analysis.")
        else:
            st.error("⚠️ Result: Crack Detected")
    else:
        st.success("✅ Result: No Crack Detected")

    # -----------------------------
    st.info(f"🧱 Severity Level: **{severity_level}**")
    st.write(f"🛠 **Recommendation:** {recommendation}")

    # -----------------------------
    # TECHNICAL EXPLANATION
    with st.expander("ℹ️ How this decision was made"):
        st.write("""
        - **CNN Model** detects texture-level cracks including micro-cracks  
        - **Crack Area** measures visible pixel-level crack coverage  
        - **Edge Density** validates structural discontinuities  
        - Final decision is based on **hybrid intelligence**
        """)
