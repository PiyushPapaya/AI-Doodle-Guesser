from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load model and labels
try:
    model = tf.keras.models.load_model('sketch_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

try:
    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"Labels loaded: {labels}")
    if len(labels) != 20:
        print(f"Warning: Expected 20 labels, got {len(labels)}")
except Exception as e:
    print(f"Error loading labels: {e}")
    raise

@app.route('/')
def index():
    try:
        if os.path.exists(os.path.join(app.static_folder, 'index.html')):
            print("Serving index.html")
            return send_from_directory(app.static_folder, 'index.html')
        else:
            print("index.html not found in static folder")
            return jsonify({'error': 'index.html not found'}), 404
    except Exception as e:
        print(f"Error serving index.html: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data
        data = request.get_json()
        if not data or 'image' not in data:
            raise ValueError("No image data in request")
        
        image_data = data['image'].split(',')[1]
        print("Received image data (first 50 chars):", image_data[:50])
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            raise ValueError(f"Base64 decode failed: {e}")
        
        # Open image
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Image open failed: {e}")
        
        # Preprocess image
        try:
            image = image.resize((28, 28))
            image = image.convert('L')  # Grayscale
            img_array = np.array(image)
            img_array = 1.0 - img_array / 255.0  # Invert (white-on-black to black-on-white)
            img_array = img_array.reshape(1, 28, 28, 1)  # Shape for model
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {e}")
        
        # Log preprocessed image
        print("Preprocessed image shape:", img_array.shape)
        print("Preprocessed image sample:", img_array[0, :5, :5, 0])
        
        # Predict
        try:
            predictions = model.predict(img_array, verbose=0)
            top_pred_idx = np.argmax(predictions[0])
            top_pred_prob = predictions[0][top_pred_idx]
            if top_pred_idx >= len(labels):
                raise ValueError(f"Prediction index {top_pred_idx} out of bounds for {len(labels)} labels")
        except Exception as e:
            raise ValueError(f"Model prediction failed: {e}")
        
        print(f"Prediction: {labels[top_pred_idx]} ({top_pred_prob:.4f})")
        
        # Return top prediction and all probabilities
        return jsonify({
            'predictions': [{
                'class': labels[top_pred_idx],
                'probability': float(top_pred_prob),
                'probabilities': [float(p) for p in predictions[0]]
            }],
            'categories': labels
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)