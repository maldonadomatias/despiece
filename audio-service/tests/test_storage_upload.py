import os
import tempfile
import pytest
from src.storage import upload_from_path, download_to_path, get_s3_client


@pytest.fixture(autouse=True)
def s3_env(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT", "http://localhost:9000")
    monkeypatch.setenv("S3_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("S3_SECRET_KEY", "minioadmin")
    monkeypatch.setenv("S3_BUCKET", "songs")
    monkeypatch.setenv("S3_REGION", "us-east-1")


@pytest.mark.integration
def test_upload_from_path_round_trip(tmp_path):
    client = get_s3_client()
    bucket = os.environ["S3_BUCKET"]
    try:
        client.create_bucket(Bucket=bucket)
    except Exception:
        pass

    src = tmp_path / "src.bin"
    src.write_bytes(b"hello despiece")
    key = "tests/upload_from_path.bin"
    upload_from_path(str(src), key)

    dst = tmp_path / "dst.bin"
    download_to_path(key, str(dst))
    assert dst.read_bytes() == b"hello despiece"
