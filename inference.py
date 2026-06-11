import argparse
import cv2
import time
from ultralytics import YOLO

CLASSES = [
    "person", "car", "truck", "bus", "motorcycle",
    "bicycle", "traffic light", "stop sign",
    "dog", "cat", "bird", "chair"
]

def run_inference(source, weights, conf_thresh=0.25, save=True):
    model = YOLO(weights)
    img = cv2.imread(source)
    if img is None:
        raise FileNotFoundError(f"Image not found: {source}")

    start = time.time()
    results = model(img, conf=conf_thresh)[0]
    latency = (time.time() - start) * 1000

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{CLASSES[cls] if cls < len(CLASSES) else cls}: {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"Detections: {len(results.boxes)} | Latency: {latency:.1f}ms")

    if save:
        out_path = source.replace(".", "_out.")
        cv2.imwrite(out_path, img)
        print(f"Saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  type=str, required=True)
    parser.add_argument("--weights", type=str, default="runs/train/yolov8_coco_subset/weights/best.pt")
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--save",    action="store_true", default=True)
    args = parser.parse_args()

    run_inference(args.source, args.weights, args.conf, args.save)
