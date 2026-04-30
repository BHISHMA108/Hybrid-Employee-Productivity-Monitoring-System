import os
import io
import json
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS

# Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

# PyTorch
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)

# --- Path Configurations ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KERAS_MODEL_PATH = os.path.join(BASE_DIR, "Model B (Keras Tensorflow)", "keras_model_with_5k_images.h5")
PYTORCH_MODEL_PATH = os.path.join(BASE_DIR, "Model A (MobileNet)", "mobilenet_model.pth")
STUDENT_MODEL_PATH = os.path.join(BASE_DIR, "baseStudentModel", "student_model.h5")

CLASS_NAMES = ['Face', 'Non-Face']

# --- Load Models ---
print("Loading Keras Model...")
keras_model = load_model(KERAS_MODEL_PATH)

print("Loading Hybrid Student Model...")
student_model = load_model(STUDENT_MODEL_PATH)

print("Loading PyTorch Model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pytorch_model = models.mobilenet_v2(weights=None)
pytorch_model.classifier[1] = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(pytorch_model.last_channel, 2)
)
pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device, weights_only=False), strict=True)
pytorch_model.to(device)
pytorch_model.eval()

# --- Preprocessing Functions ---
# Keras / Student preprocess
def preprocess_keras(image):
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    # Ensure 3 channels
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# PyTorch preprocess
pytorch_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_pytorch(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    tensor = pytorch_transform(image)
    return tensor.unsqueeze(0).to(device)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # --- Inference ---
        
        # 1. Keras Model
        keras_input = preprocess_keras(image)
        keras_pred = keras_model.predict(keras_input)
        keras_class_idx = np.argmax(keras_pred[0])
        keras_confidence = float(keras_pred[0][keras_class_idx])

        # 2. Student Model
        student_pred = student_model.predict(keras_input)
        student_class_idx = np.argmax(student_pred[0])
        student_confidence = float(student_pred[0][student_class_idx])

        # 3. PyTorch Model
        pytorch_input = preprocess_pytorch(image)
        with torch.no_grad():
            pytorch_outputs = pytorch_model(pytorch_input)
            # Apply softmax to get probabilities
            pytorch_probs = torch.nn.functional.softmax(pytorch_outputs, dim=1)
            pytorch_confidence, pytorch_class_idx = torch.max(pytorch_probs, 1)
            pytorch_confidence = float(pytorch_confidence.cpu().numpy()[0])
            pytorch_class_idx = int(pytorch_class_idx.cpu().numpy()[0])

        results = {
            "keras": {
                "class": CLASS_NAMES[keras_class_idx],
                "confidence": keras_confidence
            },
            "student": {
                "class": CLASS_NAMES[student_class_idx],
                "confidence": student_confidence
            },
            "pytorch": {
                "class": CLASS_NAMES[pytorch_class_idx],
                "confidence": pytorch_confidence
            }
        }

        return jsonify(results)

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
