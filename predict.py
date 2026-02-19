# predict.py
import tensorflow as tf
import numpy as np
import cv2
import os

# ---------------- CONFIG ---------------- #
MODEL_PATH = "healthy_vs_rotten.h5"
IMG_SIZE = 224

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    """
    Returns:
        label: "Fresh" | "Rotten" | "Invalid Image"
        confidence: float (0–1)
    """

    # Read image
    img = cv2.imread(img_path)

    if img is None:
        return "Invalid Image", 0.0

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img, verbose=0)

    # Sigmoid output (binary model)
    score = float(pred[0][0])   # 0–1

    # ---------------- LOGIC ---------------- #
    THRESHOLD = 0.75   # confidence gate (non-fruit rejection)

    if score >= 0.5:
        label = "Rotten"
        confidence = score
    else:
        label = "Fresh"
        confidence = 1 - score

    # Reject non-fruit / non-vegetable
    if confidence < THRESHOLD:
        return "Invalid Image", confidence

    return label, confidence
