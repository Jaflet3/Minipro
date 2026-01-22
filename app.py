import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Crack Detection", layout="wide")

# -----------------------------
# DOWNLOAD MODEL FROM DRIVE
# -----------------------------
MODEL_FILE = "crack_model.h5"
FILE_ID = "1nz82zuEBc0y5rcj9X7Uh5YDvv05VkZuc"

def download_from_gdrive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v

    if token:
        response = session.get(
            URL, params={'id': file_id, 'confirm': token}, stream=True
        )

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("📥 Downloading model..."):
            download_from_gdrive(FILE_ID, MODEL_FILE)
    return tf.keras.models.load_model(MODEL_FILE)

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🧱 Concrete Crack Detection System")

uploaded_file = st.file_uploader(
    "Upload concrete surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((150, 150))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr, verbose=0)[0][0]

    st.subheader("Result")

    if pred > 0.5:
        st.error(f"⚠️ Crack Detected ({pred*100:.2f}%)")
    else:
        st.success(f"✅ No Crack Detected ({(1-pred)*100:.2f}%)")
