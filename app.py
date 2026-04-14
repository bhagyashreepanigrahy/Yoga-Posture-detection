from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from collections import OrderedDict

import timm

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(r"C:\Users\HP\OneDrive\Documents\Desktop\project\resNet_18_model.pth")
LABELS_PATH = BASE_DIR / "labels.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_CLASSES = 23


app = Flask(__name__)
_model = None


# LOAD LABELS FROM JSON FILE
def load_labels():
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    return labels


# LOAD MODEL
def get_model():
    global _model
    if _model is None:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        
        model = timm.create_model(
            "resnet18",
            pretrained=False,
            num_classes=NUM_CLASSES,
        )
        
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        _model = model
    return _model


# PREPROCESSING FOR RESNET (MATCHES TRAINING)
def preprocess(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0  # Only divide by 255, NO normalization
    
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    return tensor



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    labels = load_labels()

    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"]
    img = Image.open(file.stream)

    x = preprocess(img)

    model = get_model()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    # Check if confidence is too low (model not trained for this posture)
    CONFIDENCE_THRESHOLD = 0.4  # 40% confidence threshold
    
    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify(
            {
                "error": "model is not trained with this posture currently",
                "prediction": None,
                "top": [],
            }
        ), 200

    top5_idx = np.argsort(-probs)[:5]

    top = [
        {
            "index": int(i),
            "label": labels[i],
            "confidence": float(probs[i]),
        }
        for i in top5_idx
    ]

    return jsonify(
        {
            "prediction": top[0],
            "top": top,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)