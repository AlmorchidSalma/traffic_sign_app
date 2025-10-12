from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2

app = Flask(__name__)
model = YOLO("best.pt")  # ton modèle entraîné

@app.route('/')
def home():
    return " Traffic Sign Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_path = "temp.jpg"
    file.save(img_path)

    results = model(img_path)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        detections.append({
            "class_id": cls_id,
            "confidence": conf
        })
    return jsonify(detections)

if __name__ == '__main__':
    app.run(debug=True)
