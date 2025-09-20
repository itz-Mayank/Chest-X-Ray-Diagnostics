import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(layout="wide", page_title="COVID-19 X-Ray Detection")

@st.cache_resource
def load_tflite_model():
    """Loads the TFLite model and allocates tensors."""
    try:
        interpreter = tf.lite.Interpreter(model_path='model.tflite')
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model when the app starts
interpreter = load_tflite_model()

# Class Labels
CLASS_LABELS = ['COVID', 'Normal', 'Viral Pneumonia']

def preprocess_image(image):
    """Preprocesses the uploaded image to match the model's input requirements."""
    img = image.resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = keras.utils.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return resnet_preprocess(img_array_expanded)

# Web App Interface
st.title("COVID-19 Detection from Chest X-Rays")
st.write(
    "Upload a chest X-ray image, and the application will predict if it indicates"
    " COVID-19, Viral Pneumonia, or is Normal."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Choose an X-ray image...", type=["jpg", "jpeg", "png"]
    )

if uploaded_file is not None and interpreter is not None:
    # Open and display the uploaded image in the first column
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption='Uploaded X-ray', use_column_width=True)

    # Process and predict
    processed_img = preprocess_image(image)

    # Prediction logic for TFLite Interpreter
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    processed_img = processed_img.astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Get the top prediction details
    predicted_index = np.argmax(predictions[0])
    predicted_label = CLASS_LABELS[predicted_index]
    confidence = np.max(predictions[0]) * 100

    # Display the prediction result in the second column
    with col2:
        st.header("Prediction Result")
        if predicted_label == 'COVID':
            st.error(f"Predicted Class: **{predicted_label}**", icon="🦠")
        elif predicted_label == 'Normal':
            st.success(f"Predicted Class: **{predicted_label}**", icon="😇")
        else: # Viral Pneumonia
            st.warning(f"Predicted Class: **{predicted_label}**", icon="🫁")
        
        # Display confidence scores for all classes
        st.subheader("Confidence Scores:")
        for i, label in enumerate(CLASS_LABELS):
            st.text(f"{label}: {predictions[0][i]*100:.2f}%")
            
        st.subheader("Confidence Scores:")
        st.bar_chart(data={label: pred for label, pred in zip(CLASS_LABELS, predictions[0])})

        # st.header("Prediction Result")
        # st.info(f"Predicted Class: **{predicted_label}**")
        # st.write(f"Confidence: **{confidence:.2f}%**")
        # st.divider()


# Display a warning if the model file isn't found
if interpreter is None:
    st.warning("Model file 'model.tflite' not found. Please place it in the same folder as this app.")