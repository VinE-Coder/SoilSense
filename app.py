from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="soilsense_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image = Image.open(file).convert("RGB")
    image = image.resize((224, 224))
    input_array = np.array(image, dtype=np.float32) / 255.0
    input_array = np.expand_dims(input_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], input_array)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    # Return only labels with > 0.5 confidence
    threshold = 0.5
    results = [
        {"label": labels[i], "confidence": float(conf)}
        for i, conf in enumerate(output)
        if conf > threshold and i < len(labels)
    ]

    return jsonify({"predictions": results})


@app.route("/", methods=["GET"])
def home():
    return "SoilSense API is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
