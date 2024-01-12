from mlflow.sagemaker import SageMakerDeploymentClient

experiment_id = "545391055719976846"
run_id = "cecfdfa6977441d0b6ebc4818c1503a9"
region = "ap-south-1"
aws_id = "754134281902"
arn = "arn:aws:iam::754134281902:role/aws-sagemaker-for-deploy-ml-model" # remote connection to aws sagemaker role
app_name = "model-application"
model_uri = f"mlruns/{experiment_id}/{run_id}/artifacts/random-forest-model"

tag_id = "2.9.2"

image_url = aws_id + ".dkr.ecr." + region + ".amazonaws.com/mlflow-pyfunc:" + tag_id

client = SageMakerDeploymentClient(
    target_uri="sagemaker:/" + region
)

client.create_deployment(
    name=app_name,
    model_uri=model_uri,
    config={"instance_type": "ml.m5.large", "region_name": region, "mode": "replace", "image_url": image_url, "execution_role_arn": arn},
)
