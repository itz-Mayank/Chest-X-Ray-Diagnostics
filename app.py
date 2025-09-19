import os
from flask import Flask, request, render_template, url_for, flash, redirect # <-- ADDED flash and redirect
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
# ADDED: A secret key is required for flashing messages
app.config['SECRET_KEY'] = 'a_super_secret_key_change_me'
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- TFLite Model Loading ---
TFLITE_MODEL_PATH = 'model.tflite'
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")
    interpreter = None

CLASS_LABELS = ['COVID', 'Normal', 'Viral Pneumonia']

# --- Helper Functions ---
def preprocess_image(image_path):
    """Loads and preprocesses an image for the model."""
    # MODIFIED: Added error handling for non-image files
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.resnet50.preprocess_input(img_array_expanded)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles file upload and prediction."""
    if interpreter is None:
        flash("Model is not available. Please check server logs.", "error")
        return redirect(url_for('index'))

    if 'file' not in request.files or request.files['file'].filename == '':
        # MODIFIED: Use flash to show an error message on the main page
        flash("No file selected. Please choose an image to upload.", "warning")
        return redirect(url_for('index'))
    
    file = request.files['file']

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess and predict
        processed_img = preprocess_image(filepath)
        
        # MODIFIED: Check if preprocessing was successful
        if processed_img is None:
            flash("Invalid file format. Please upload a valid image file (JPG, PNG, etc.).", "error")
            return redirect(url_for('index'))

        # --- Prediction with TFLite Interpreter ---
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Format results
        predicted_index = np.argmax(predictions)
        result = {
            "label": CLASS_LABELS[predicted_index],
            "confidence": f"{np.max(predictions) * 100:.2f}",
            "scores": {label: score * 100 for label, score in zip(CLASS_LABELS, predictions)}
        }
        
        # MODIFIED: Pass the prediction to the single-page template
        return render_template('index.html', filename=filename, prediction=result)

    return redirect(url_for('index'))

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)