import os
import io
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

# TensorFlow for friend's ResNet50
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== MODEL TANIMLARI (PyTorch) =====

class HemoCTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 20 * 20, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ===== GÖRÜNTÜ ÖN İŞLEME =====

# HemoCTNet için 
hemoctnet_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ===== MODEL YÜKLEME =====
device = torch.device('cpu') 

# --- HemoCTNet (PyTorch) ---
custom_model = None
custom_model_error = None
try:
    custom_model = HemoCTNet().to(device)
    ckpt_path = os.path.join(BASE_DIR, 'ozgun_cnn_final.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError('ozgun_cnn_final.pt bulunamadi.')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    custom_model.load_state_dict(state_dict)
    custom_model.eval()
    print("[OK] Ozgun CNN (HemoCTNet) basariyla yuklendi (PyTorch).")
except Exception as e:
    custom_model = None
    custom_model_error = str(e)
    print(f"[!] HemoCTNet yüklenemedi: {e}")

# --- ResNet50 (Keras) ---
pretrained_resnet = None
try:
    keras_candidates = [
        os.path.join(BASE_DIR, 'head_ct_resnet50_v1.keras'),
        os.path.join(BASE_DIR, 'en_iyi_resnet50_modeli.keras'),
    ]
    keras_path = next((p for p in keras_candidates if os.path.exists(p)), None)
    if keras_path is None:
        raise FileNotFoundError('head_ct_resnet50_v1.keras veya en_iyi_resnet50_modeli.keras bulunamadi.')
    pretrained_resnet = load_model(
        keras_path,
        safe_mode=False,
        custom_objects={'preprocess_input': preprocess_input},
    )
    print("[OK] ResNet50 (Transfer Learning) basariyla yuklendi (TensorFlow/Keras).")
except Exception as e:
    print(f"[!] ResNet50 Keras modeli yüklenemedi: {e}")


# ===== ROUTES =====
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    model_type = (request.form.get('model', 'custom') or 'custom').strip().lower()
    if model_type in {'ozgun', 'özgün', 'ozgun_cnn'}:
        model_type = 'custom'
    elif model_type in {'resnet', 'resnet50'}:
        model_type = 'pretrained'

    try:
        if model_type == 'custom' and custom_model is not None:
            # ---- PyTorch (HemoCTNet) ----
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            tensor = hemoctnet_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = custom_model(tensor)
                prob_hemorrhage = torch.sigmoid(logit).item()
                pred_class = 1 if prob_hemorrhage >= 0.5 else 0
                confidence = prob_hemorrhage if pred_class == 1 else 1.0 - prob_hemorrhage
                
        elif model_type == 'pretrained' and pretrained_resnet is not None:
            # ---- TensorFlow (ResNet50 Keras) ----
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            image = image.resize((320, 320))
            img_array = img_to_array(image) / 255.0 # Keras standardization
            img_array = np.expand_dims(img_array, axis=0) # Batch definition
            
            predictions = pretrained_resnet.predict(img_array)
            # Binary classification shape
            if getattr(predictions, 'shape', (1, 1))[1] == 2:
                pred_class = np.argmax(predictions[0])
                confidence = predictions[0][pred_class]
            else:
                prob_hemorrhage = predictions[0][0]
                pred_class = 1 if prob_hemorrhage >= 0.5 else 0
                confidence = prob_hemorrhage if pred_class == 1 else 1.0 - prob_hemorrhage

        else:
            if model_type == 'custom':
                detail = f" Detay: {custom_model_error}" if custom_model_error else ""
                return jsonify({'error': f'PyTorch model yüklenemedi.{detail}'}), 503
            return jsonify({'error': 'Model yüklenmemiş veya bulunamadı.'}), 503

        class_labels = {0: 'normal', 1: 'hemorrhage'}
        result = class_labels[pred_class]

        return jsonify({
            'prediction': result,
            'confidence': round(float(confidence) * 100, 2),
            'model_used': 'Özgün CNN (HemoCTNet)' if model_type == 'custom' else 'ResNet50 (Transfer Learning)',
            'label_tr':   'Kanama Tespit Edildi' if result == 'hemorrhage' else 'Kanama Tespit Edilmedi',
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
