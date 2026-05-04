import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "skin_model.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "model" / "class_names.json"
IMG_SIZE = 224

DESCRIPTIONS = {
    "Acne": "Pimples and oily skin caused by clogged pores.",
    "Tinea": "Fungal infection affecting the skin.",
    "Psoriasis": "Chronic condition causing red, scaly patches.",
    "Vitiligo": "Loss of skin color resulting in white patches.",
}


@st.cache_resource
def load_model_and_labels():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError("Model not found. Train the model first.")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("Class labels not found. Train the model first.")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return model, class_names


st.set_page_config(page_title="Skin Disease Detection")
st.title("Skin Disease Detection System")
st.warning("This AI demo is not a medical diagnosis. Please consult a doctor.")

try:
    model, class_names = load_model_and_labels()
except Exception as exc:
    st.error(str(exc))
    st.stop()

uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Invalid image file.")
        st.stop()

    st.image(image, caption="Uploaded Image", use_container_width=True)

    resized = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.asarray(resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    scores = model.predict(image_array, verbose=0)[0]
    best_index = int(np.argmax(scores))
    confidence = float(scores[best_index]) * 100
    predicted_class = class_names[best_index]

    st.subheader("Prediction Result")
    if confidence < 60:
        st.error("Low confidence. Please upload a clearer image.")
    else:
        st.success(f"Disease: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.info(DESCRIPTIONS.get(predicted_class, "No description available."))

    st.subheader("All Predictions")
    for index in np.argsort(scores)[::-1]:
        st.write(f"{class_names[index]}: {float(scores[index]) * 100:.2f}%")
