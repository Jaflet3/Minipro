import streamlit as st
import cv2
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Crack Detection", layout="wide")
st.title("🛠️ Concrete Crack Detection System")

# --------------------------
def crack_severity(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 100, 200)

    crack_pixels = np.sum(edges > 0)
    total_pixels = edges.size
    severity = (crack_pixels / total_pixels) * 100

    return round(severity, 3), edges

def overlay_crack(img_path, edges):
    img = cv2.imread(img_path)
    overlay = img.copy()
    overlay[edges > 0] = [0, 0, 255]
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --------------------------
uploaded_file = st.file_uploader("Upload concrete image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    temp_path = "temp.jpg"
    img.save(temp_path)

    severity, edges = crack_severity(temp_path)

    # DECISION LOGIC
    if severity < 0.3:
        result = "No Crack"
        level = "None"
        show_overlay = False
    elif severity < 1.5:
        result = "Crack Detected"
        level = "Low"
        show_overlay = True
    elif severity < 5:
        result = "Crack Detected"
        level = "Medium"
        show_overlay = True
    else:
        result = "Crack Detected"
        level = "High"
        show_overlay = True

    col1, col2 = st.columns(2)
    col1.image(img, caption="Original Image", use_column_width=True)

    if show_overlay:
        col2.image(overlay_crack(temp_path, edges), caption="Crack Visualization", use_column_width=True)
    else:
        col2.image(img, caption="No Crack Found", use_column_width=True)

    if result == "Crack Detected":
        st.error(f"Result: {result}")
    else:
        st.success(f"Result: {result}")

    st.info(f"Severity Level: {level}")
    st.write(f"Crack Area (%): {severity}")
