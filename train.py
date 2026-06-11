import argparse
from ultralytics import YOLO
import mlflow
import mlflow.pytorch

def train(data_config, epochs, img_size, batch_size, weights):
    mlflow.start_run()

    model = YOLO(weights)

    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        project="runs/train",
        name="yolov8_coco_subset",
        save=True,
        val=True,
    )

    metrics = {
        "mAP50":      results.results_dict.get("metrics/mAP50(B)", 0),
        "mAP50_95":   results.results_dict.get("metrics/mAP50-95(B)", 0),
        "precision":  results.results_dict.get("metrics/precision(B)", 0),
        "recall":     results.results_dict.get("metrics/recall(B)", 0),
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.log_param("epochs",     epochs)
    mlflow.log_param("img_size",   img_size)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("weights",    weights)

    mlflow.pytorch.log_model(model.model, "yolov8_model")
    mlflow.end_run()

    print(f"\nTraining complete. mAP@0.5: {metrics['mAP50']:.4f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    type=str,   default="coco_subset.yaml")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--img",     type=int,   default=640)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--weights", type=str,   default="yolov8n.pt")
    args = parser.parse_args()

    train(args.data, args.epochs, args.img, args.batch, args.weights)
