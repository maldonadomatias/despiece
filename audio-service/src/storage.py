import os
import boto3
from botocore.client import Config


def get_s3_client():
    endpoint = os.environ.get("S3_ENDPOINT")
    access_key = os.environ.get("S3_ACCESS_KEY")
    secret_key = os.environ.get("S3_SECRET_KEY")
    missing = [k for k, v in {"S3_ENDPOINT": endpoint, "S3_ACCESS_KEY": access_key, "S3_SECRET_KEY": secret_key}.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )


def download_to_path(storage_key: str, dest_path: str) -> None:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise EnvironmentError("Missing required env var: S3_BUCKET")
    client = get_s3_client()
    client.download_file(bucket, storage_key, dest_path)
