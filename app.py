import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_crack_model():
    model_path = hf_hub_download(
        repo_id="Jaflet/crack_model",
        filename="crack_model.h5"   # MUST MATCH HF FILE NAME
    )
    model = tf.keras.models.load_model(model_path)
    return model


model = load_crack_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧱 Crack Detection App")
st.write("Upload an image to detect cracks")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    confidence = float(prediction[0][0])

    st.subheader("Result")

    if confidence > 0.5:
        st.error(f"🚨 Crack Detected (Confidence: {confidence:.2f})")
    else:
        st.success(f"✅ No Crack Detected (Confidence: {1 - confidence:.2f})")
