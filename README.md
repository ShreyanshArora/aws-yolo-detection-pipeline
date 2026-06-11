# AWS YOLOv8 Object Detection Pipeline

An end-to-end MLOps pipeline for real-time object detection using YOLOv8, fine-tuned on a 12-class COCO subset and deployed serverlessly on AWS.

## Architecture
COCO Dataset (S3)
│
▼
Data Preprocessing & Augmentation
(mosaic, mixup, random affine)
│
▼
YOLOv8 Fine-Tuning
(SageMaker Training Job — GPU)
│
▼
Model Registry (S3 + ECR)
│
▼
Serverless Inference
(API Gateway → Lambda → YOLOv8)
│
▼
Monitoring & Drift Detection
(CloudWatch Metrics → SageMaker Pipelines retraining trigger)

## Results

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.87 |
| Inference latency | <200ms |
| Deployment time | <8 minutes |
| Dataset size | ~8,400 images (12-class COCO subset) |

## Stack

- **Model:** YOLOv8 (Ultralytics)
- **Training:** AWS SageMaker Training Jobs
- **Inference:** AWS Lambda + API Gateway
- **Storage:** Amazon S3, Amazon ECR
- **Monitoring:** Amazon CloudWatch + SageMaker Pipelines
- **Framework:** PyTorch, OpenCV

## Project Structure
├── train.py                  # YOLOv8 fine-tuning script
├── inference.py              # Local inference script
├── lambda_handler.py         # AWS Lambda inference handler
├── sagemaker_pipeline.py     # SageMaker training job config
├── requirements.txt
└── README.md
## Setup

```bash
pip install -r requirements.txt
python train.py --data coco_subset.yaml --epochs 50 --img 640
```

## Inference

```bash
python inference.py --source image.jpg --weights runs/train/best.pt
```
