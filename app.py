# app.py
import io
import os
import time
import base64
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Use ultralytics YOLO (v8); install with `pip install ultralytics`
from ultralytics import YOLO

# Config
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "best.pt")  # allow override via env var
CONF_THRESH = float(os.environ.get("CONF_THRESH", 0.25))  # default confidence threshold

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load model once at startup
print("Loading YOLO model from", MODEL_PATH)
model = YOLO(MODEL_PATH)
print("Model loaded. Names:", getattr(model, "names", None))

# Utility: convert OpenCV image (BGR) to base64 PNG
def imencode_to_base64(img_bgr):
    _, im_png = cv2.imencode(".png", img_bgr)
    b64 = base64.b64encode(im_png.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# Utility: draw boxes using OpenCV
def draw_boxes_cv2(img_bgr, boxes, scores, class_ids, class_names):
    img = img_bgr.copy()
    h, w = img.shape[:2]
    for (x1, y1, x2, y2), conf, cid in zip(boxes, scores, class_ids):
        # convert to int
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = class_names[int(cid)] if (class_names and int(cid) in class_names) or (class_names and int(cid) < len(class_names)) else str(int(cid))
        text = f"{label} {conf:.2f}"
        # box color depends on class
        color = tuple(int(c) for c in np.random.RandomState(int(cid)).randint(0,255,3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # put text background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return img

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts multipart form-data with key 'file'.
    Returns JSON:
    {
      "detections": [
        {"label":"Stop","confidence":0.92,"box":[x1,y1,x2,y2]},
        ...
      ],
      "image": "data:image/png;base64,...."
    }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = f.filename.lower()
    raw = f.read()
    npimg = np.frombuffer(raw, np.uint8)

    # Determine if image or video by extension
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    try:
        if ext in image_exts:
            # decode image
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image")
            # ultralytics model can accept numpy array directly
            results = model.predict(source=img, conf=CONF_THRESH, device="cpu", verbose=False)  # single-image inference
            r = results[0]
            # extract boxes (xyxy), confidences, class ids
            boxes = []
            scores = []
            class_ids = []
            class_names = model.names if hasattr(model, "names") else None

            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                # r.boxes.xyxy is a tensor-like; convert to numpy
                xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else np.array(r.boxes.xyxy)
                confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, "cpu") else np.array(r.boxes.conf)
                cls = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, "cpu") else np.array(r.boxes.cls)
                for b, c, cl in zip(xyxy, confs, cls):
                    boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                    scores.append(float(c))
                    class_ids.append(int(cl))
            detections = []
            for box, score, cid in zip(boxes, scores, class_ids):
                label = class_names[cid] if class_names and cid < len(class_names) else str(cid)
                detections.append({"label": label, "confidence": float(score), "box": [float(x) for x in box], "class_id": int(cid)})

            # annotated image
            annotated = draw_boxes_cv2(img, boxes, scores, class_ids, class_names)
            img_b64 = imencode_to_base64(annotated)
            return jsonify({"detections": detections, "image": img_b64})
        else:
            # treat as video - save to temp file then annotate frames (synchronous)
            # WARNING: processing large videos will be slow on free hosts; consider streaming/CV pipeline for production
            tmp_in = f"/tmp/input_{int(time.time())}{ext}"
            tmp_out = f"/tmp/output_{int(time.time())}.mp4"
            with open(tmp_in, "wb") as fh:
                fh.write(raw)

            cap = cv2.VideoCapture(tmp_in)
            if not cap.isOpened():
                return jsonify({"error": "Could not open video"}), 400
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp_out, fourcc, fps, (w, h))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            processed = 0
            detections_summary = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame, conf=CONF_THRESH, device="cpu", verbose=False)
                r = results[0]
                boxes = []
                scores = []
                class_ids = []
                class_names = model.names if hasattr(model, "names") else None
                if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else np.array(r.boxes.xyxy)
                    confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, "cpu") else np.array(r.boxes.conf)
                    cls = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, "cpu") else np.array(r.boxes.cls)
                    for b, c, cl in zip(xyxy, confs, cls):
                        boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                        scores.append(float(c))
                        class_ids.append(int(cl))
                        detections_summary.append({"frame": processed, "label": class_names[int(cl)] if class_names and int(cl) < len(class_names) else str(int(cl)), "conf": float(c)})

                annotated = draw_boxes_cv2(frame, boxes, scores, class_ids, class_names)
                out.write(annotated)
                processed += 1

            cap.release()
            out.release()
            # return the annotated video as base64 (can be large) or a link. We'll return a small metadata + endpoint to download.
            with open(tmp_out, "rb") as fh:
                vid_b = fh.read()
            vid_b64 = base64.b64encode(vid_b).decode("utf-8")
            # cleanup
            try:
                os.remove(tmp_in)
                os.remove(tmp_out)
            except Exception:
                pass
            return jsonify({"detections_summary": detections_summary, "video_base64": vid_b64})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # for local development
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
