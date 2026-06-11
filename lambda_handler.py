import json
import base64
import boto3
import numpy as np
import cv2
import os
import tempfile
from ultralytics import YOLO

S3_BUCKET  = os.environ.get("MODEL_BUCKET", "your-model-bucket")
MODEL_KEY  = os.environ.get("MODEL_KEY",    "models/yolov8_best.pt")
CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.25"))

_model = None

def load_model():
    global _model
    if _model is None:
        s3 = boto3.client("s3")
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        s3.download_fileobj(S3_BUCKET, MODEL_KEY, tmp)
        tmp.close()
        _model = YOLO(tmp.name)
    return _model


def handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        img_b64 = body.get("image")
        if not img_b64:
            return {"statusCode": 400, "body": json.dumps({"error": "No image provided"})}

        img_bytes = base64.b64decode(img_b64)
        np_arr   = np.frombuffer(img_bytes, np.uint8)
        img      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        model   = load_model()
        results = model(img, conf=CONF_THRESH)[0]

        detections = []
        for box in results.boxes:
            detections.append({
                "bbox":       box.xyxy[0].tolist(),
                "class_id":   int(box.cls[0]),
                "confidence": round(float(box.conf[0]), 4),
            })

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "detections": detections,
                "count":      len(detections),
            }),
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
