import os
from pathlib import Path

from dotenv import load_dotenv
import boto3
from botocore.exceptions import BotoCoreError, ClientError

load_dotenv()

S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

print("[S3 CONFIG]")
print("S3_ENDPOINT:", S3_ENDPOINT)
print("S3_BUCKET_NAME:", S3_BUCKET_NAME)
print("S3_ACCESS_KEY:", bool(S3_ACCESS_KEY))
print("S3_SECRET_KEY:", bool(S3_SECRET_KEY))

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)


def upload_evidence(file_path, candidate_id):
    """
    Upload evidence directly into:

    cheating-engine/
        evidence_<candidate_id>/
            filename

    No test folder.
    No date folder.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        print(f"[S3] File not found: {file_path}")
        return None

    evidence_folder = f"evidence_{candidate_id}"

    s3_key = f"cheating-engine/{evidence_folder}/{file_path.name}"

    # Detect content type
    suffix = file_path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        content_type = "image/jpeg"
    elif suffix == ".png":
        content_type = "image/png"
    elif suffix == ".mp4":
        content_type = "video/mp4"
    elif suffix == ".wav":
        content_type = "audio/wav"
    else:
        content_type = "application/octet-stream"

    try:
        s3_client.upload_file(
            str(file_path),
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={
                "ContentType": content_type
            },
        )

        print(f"[S3] Uploaded: {s3_key}")

        return s3_key

    except (BotoCoreError, ClientError) as e:
        print(f"[S3] Upload failed: {e}")
        return None