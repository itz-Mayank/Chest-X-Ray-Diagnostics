import os
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- TFLite Model Loading --- # <-- CHANGE: Using TFLite Interpreter
TFLITE_MODEL_PATH = 'model.tflite'
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded successfully.")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None

CLASS_LABELS = ['COVID', 'Normal', 'Viral Pneumonia']

# --- Helper Functions ---
def preprocess_image(image_path):
    """Loads and preprocesses an image for the model."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    # Important: TFLite models still need the same preprocessing
    return tf.keras.applications.resnet50.preprocess_input(img_array_expanded)

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles file upload and prediction."""
    if interpreter is None:
        return "Model not loaded. Please check server logs.", 500

    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess and predict
        processed_img = preprocess_image(filepath)

        # --- Prediction with TFLite Interpreter --- # <-- CHANGE: New prediction logic
        # Set the value of the input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        # Run the inference
        interpreter.invoke()
        # Get the result
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Format results
        predicted_index = np.argmax(predictions)
        result = {
            "label": CLASS_LABELS[predicted_index],
            "confidence": f"{np.max(predictions) * 100:.2f}",
            "scores": {label: score * 100 for label, score in zip(CLASS_LABELS, predictions)}
        }
        
        return render_template('result.html', filename=filename, result=result)

    return "Something went wrong", 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)