import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
import tensorflow as tf

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Concrete Crack Detection",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 Concrete Crack Detection System")
st.caption("CNN-based Structural Health Monitoring")

# -----------------------------
# GOOGLE DRIVE MODEL DOWNLOAD
# -----------------------------
MODEL_PATH = "crack_model.h5"
FILE_ID = "1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"

def download_model(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value

    if token:
        response = session.get(
            URL,
            params={"id": file_id, "confirm": token},
            stream=True,
        )

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_crack_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading CNN model..."):
            download_model(FILE_ID, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_crack_model()

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Concrete Surface Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = image.resize((150, 150))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Prediction
    prediction = model.predict(arr, verbose=0)[0][0]

    st.divider()
    st.subheader("📊 Result")

    if prediction >= 0.5:
        st.error(f"⚠️ Crack Detected\n\nConfidence: {prediction*100:.2f}%")
        st.info("🛠 Recommendation: Inspection and repair advised")
    else:
        st.success(f"✅ No Crack Detected\n\nConfidence: {(1-prediction)*100:.2f}%")
        st.info("🧱 Structure appears safe")
