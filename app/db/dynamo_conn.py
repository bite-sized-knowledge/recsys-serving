import boto3
import os

_dynamo_resource = None

def get_dynamo():
    global _dynamo_resource
    if _dynamo_resource is None:
        endpoint_url = os.getenv("DYNAMODB_ENDPOINT_URL")
        region = os.getenv("DYNAMODB_REGION", "ap-northeast-2")
        access_key = os.getenv("DYNAMODB_ACCESS_KEY_ID")
        secret_key = os.getenv("DYNAMODB_SECRET_ACCESS_KEY")

        boto3_params = {"region_name": region}
        if endpoint_url:
            boto3_params.update({
                "endpoint_url": endpoint_url,
                "aws_access_key_id": access_key,
                "aws_secret_access_key": secret_key,
            })

        _dynamo_resource = boto3.resource("dynamodb", **boto3_params)
    return _dynamo_resource