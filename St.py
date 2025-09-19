import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(layout="wide", page_title="COVID-19 X-Ray Detection")

# Model Loading
@st.cache_resource
def load_my_model():
    """Loads the pre-trained Keras model."""
    try:
        model = keras.models.load_model('Model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}", icon="🚨")
        return None

model = load_my_model()

# Class Labels
CLASS_LABELS = ['COVID', 'Normal', 'Viral Pneumonia']

# Image Preprocessing Function
def preprocess_image(image):
    """Preprocesses the uploaded image to match the model's input requirements."""
    img = image.resize((224, 224))
    img_array = keras.utils.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    # ResNet50's specific preprocessing
    return resnet_preprocess(img_array_expanded)

# Web App Interface
st.title("COVID-19 Detection from Chest X-Rays")
st.write(
    "Upload a chest X-ray image, and the application will predict if it indicates"
    " COVID-19, Viral Pneumonia, or is Normal."
)
st.divider()

# Create columns for a side by side layout
col1, col2 = st.columns(2)

with col1:
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an X-ray image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None and model is not None:
        # display the uploaded image
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        st.image(image, caption='Uploaded X-ray', use_column_width=True)
        st.info("Processing and predicting...")

        # Preprocess the image and get the prediction
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img)

        # Prediction details
        predicted_index = np.argmax(predictions[0])
        predicted_label = CLASS_LABELS[predicted_index]
        confidence = np.max(predictions[0]) * 100

        # Prediction result in the second column
        with col2:
            st.header("Prediction Result")
            if predicted_label == 'COVID':
                st.error(f"Predicted Class: **{predicted_label}**", icon="🦠")
            elif predicted_label == 'Normal':
                st.success(f"Predicted Class: **{predicted_label}**", icon="😇")
            else:
                st.warning(f"Predicted Class: **{predicted_label}**", icon="🫁")

            st.write(f"Confidence: **{confidence:.2f}%**")
            st.divider()

            # Confidence scores
            st.subheader("Confidence Scores:")
            for i, label in enumerate(CLASS_LABELS):
                st.text(f"{label}: {predictions[0][i]*100:.2f}%")

# Warning if the model file isn't found
if model is None:
    st.warning("Model file 'Model.keras' not found. Please place it in the same folder as this app.")