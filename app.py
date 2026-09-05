import os
import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load disease info
try:
    with open('disease_info/disease_data.json', 'r') as f:
        DISEASE_INFO = json.load(f)
except FileNotFoundError:
    DISEASE_INFO = {}

# Global variable for model to lazy load
model = None
class_names = []

def load_model_and_classes():
    global model, class_names
    if model is None:
        model_path = 'model/crop_disease_model.keras'
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e:
                print(f"Error loading model: {e}")
                model = None
    
    if not class_names:
        class_names_path = 'model/class_names.txt'
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            # Fallback if class names not found
            class_names = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"]

load_model_and_classes()

def preprocess_image(image_path):
    # Load image, resize it to the expected input size (224, 224)
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Expand dimensions to match batch size (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load model if not loaded yet (lazy loading)
        load_model_and_classes()
        
        if model is None:
            return jsonify({'error': 'Model not trained or not found. Please train the model first.'})
        
        try:
            # Preprocess the image
            processed_img = preprocess_image(filepath)
            
            # Predict
            predictions = model.predict(processed_img, verbose=0)
            
            # Get the index of the highest probability
            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[0][predicted_index])
            
            # Get disease info
            info = DISEASE_INFO.get(predicted_class, {
                "name": predicted_class.replace('___', ' ').replace('_', ' '),
                "description": "Description not found.",
                "treatment": "Treatment not found."
            })
            
            # For returning HTML template
            image_url = '/' + filepath.replace('\\', '/')
            
            return render_template('result.html', 
                                   disease=info, 
                                   confidence=f"{confidence*100:.2f}%", 
                                   image_url=image_url)
            
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
