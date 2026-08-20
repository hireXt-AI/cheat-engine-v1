import os
from pathlib import Path
import boto3
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not all([S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_NAME]):
  raise RuntimeError("S3 environment variables missing!")

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=os.getenv("S3_REGION", "us-east-1"),
)


def upload_evidence(file_path, candidate_id):
  file_path = Path(file_path)

  if not file_path.exists() or file_path.stat().st_size == 0:
    print(f"[S3] File missing or empty: {file_path}")
    return None

  # S3 Key: cheating-engine/evidence_<candidate_id>/<candidate_id>.mp4
  s3_key = f"cheating-engine/evidence_{candidate_id}/{candidate_id}{file_path.suffix}"

  content_type = {
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".png": "image/png",
      ".mp4": "video/mp4",
      ".wav": "audio/wav",
  }.get(file_path.suffix.lower(), "application/octet-stream")

  print(f"[S3] Uploading to Key: {s3_key}")

  try:
    s3_client.upload_file(
        str(file_path),
        S3_BUCKET_NAME,
        s3_key,
        ExtraArgs={"ContentType": content_type},
    )
    print(f"[S3] Uploaded successfully: {s3_key}")
    return s3_key
  except Exception as e:
    print(f"[S3 ERROR] {e}")
    return None