import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.parameters import ParameterString

ROLE       = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerRole"
BUCKET     = "your-sagemaker-bucket"
REGION     = "us-east-1"
IMAGE_URI  = "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/yolov8-training:latest"

session = sagemaker.Session(boto3.Session(region_name=REGION))

epochs     = ParameterString(name="Epochs",    default_value="50")
img_size   = ParameterString(name="ImgSize",   default_value="640")
batch_size = ParameterString(name="BatchSize", default_value="16")

estimator = Estimator(
    image_uri=IMAGE_URI,
    role=ROLE,
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    volume_size=50,
    output_path=f"s3://{BUCKET}/output",
    sagemaker_session=session,
    hyperparameters={
        "epochs":     epochs,
        "img_size":   img_size,
        "batch_size": batch_size,
        "data":       "/opt/ml/input/data/training/coco_subset.yaml",
    },
)

training_step = TrainingStep(
    name="YOLOv8TrainingStep",
    estimator=estimator,
    inputs={
        "training": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{BUCKET}/datasets/coco_subset/",
            content_type="application/x-image",
        )
    },
)

pipeline = Pipeline(
    name="YOLOv8DetectionPipeline",
    parameters=[epochs, img_size, batch_size],
    steps=[training_step],
    sagemaker_session=session,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=ROLE)
    execution = pipeline.start()
    print(f"Pipeline started: {execution.arn}")
