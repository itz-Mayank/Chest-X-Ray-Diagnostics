import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# --- TFLite Model Loading ---
TFLITE_MODEL_PATH = 'model.tflite'
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded successfully.")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None

CLASS_LABELS = ['COVID', 'Normal', 'Viral Pneumonia']

# --- Helper Function ---
def preprocess_image(image_stream):
    """Preprocesses an image stream for the model."""
    img = Image.open(image_stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(img_array_expanded)

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image prediction and returns JSON."""
    if interpreter is None:
        return jsonify({"error": "Model not loaded"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        processed_img = preprocess_image(file.stream)

        # Ensure input data type matches model's expectation
        input_type = input_details[0]['dtype']
        processed_img = processed_img.astype(input_type)

        # Run inference
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
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)