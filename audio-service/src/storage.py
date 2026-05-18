import os
import boto3
from botocore.client import Config


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=Config(signature_version="s3v4"),
    )


def download_to_path(storage_key: str, dest_path: str) -> None:
    client = get_s3_client()
    bucket = os.environ["S3_BUCKET"]
    client.download_file(bucket, storage_key, dest_path)
