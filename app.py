import streamlit as st
import numpy as np
import cv2
import os
import gdown
from PIL import Image
from tensorflow.keras.models import load_model
from gtts import gTTS
import tempfile
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Concrete Crack Detection",
    layout="wide",
    page_icon="🛠️"
)

st.title("🛠️ Concrete Crack Detection System")

# -----------------------------
# LOAD CNN MODEL
# -----------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"
MODEL_PATH = "crack_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading CNN model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)

# -----------------------------
# FUNCTIONS
# -----------------------------
def cnn_predict(img_path):
    """
    Returns CNN probability score (0–1)
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((150, 150))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    score = model.predict(arr, verbose=0)[0][0]
    return float(score)


def crack_severity(img_path):
    """
    Calculates crack area percentage using thresholding
    """
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    crack_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    severity = (crack_pixels / total_pixels) * 100

    return round(severity, 3), thresh


def edge_ratio(img_path):
    """
    Edge density for additional validation
    """
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size


def overlay_crack(img_path, thresh):
    """
    Overlays detected crack in red color
    """
    img = cv2.imread(img_path)
    overlay = img.copy()
    overlay[thresh == 255] = [0, 0, 255]
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def speak(text):
    """
    Browser-based text-to-speech (Cloud safe)
    """
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Settings")
voice_on = st.sidebar.checkbox("🔊 Enable Voice Alert", value=True)
cnn_threshold = st.sidebar.slider("CNN Crack Threshold", 0.0, 1.0, 0.65)

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload concrete surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file)
    temp_path = "temp.jpg"
    img.save(temp_path)

    # -----------------------------
    # PREDICTIONS
    # -----------------------------
    cnn_score = cnn_predict(temp_path)
    severity, thresh = crack_severity(temp_path)
    edge_val = edge_ratio(temp_path)

    # -----------------------------
    # FINAL DECISION LOGIC
    # -----------------------------
    if severity < 0.2:
        decision = "No Crack"
        severity_level = "None"
        recommendation = "Structure is safe"
        show_overlay = False

    elif cnn_score < cnn_threshold and edge_val < 0.01:
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
    # DISPLAY IMAGES
    # -----------------------------
    col1, col2 = st.columns(2)

    col1.image(img, caption="Original Image", use_column_width=True)

    if show_overlay:
        overlay_img = overlay_crack(temp_path, thresh)
        col2.image(overlay_img, caption="Crack Visualization", use_column_width=True)
    else:
        col2.image(img, caption="No Crack Found", use_column_width=True)

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    if decision == "Crack Detected":
        st.error(f"Result: {decision}")
    else:
        st.success(f"Result: {decision}")

    st.info(f"Severity Level: {severity_level}")
    st.write(f"🔍 CNN Score: **{cnn_score:.3f}**")
    st.write(f"📏 Crack Area (%): **{severity}**")
    st.write(f"🛠 Recommendation: **{recommendation}**")

    # -----------------------------
    # VOICE ALERT
    # -----------------------------
    if voice_on:
        if decision == "Crack Detected":
            speak(
                f"Warning. Crack detected. Severity level is {severity_level}. "
                f"{recommendation}"
            )
        else:
            speak("No crack detected. Structure is safe.")
