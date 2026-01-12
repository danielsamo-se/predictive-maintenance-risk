"""Upload active model artifacts to MinIO/S3"""

import argparse
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish seq model artifacts to S3")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("models/production/seq"),
        help="Artifact directory",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version to upload",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="pmrisk",
        help="S3 prefix",
    )
    args = parser.parse_args()

    s3_endpoint = os.environ.get("S3_ENDPOINT")
    s3_access_key = os.environ.get("S3_ACCESS_KEY")
    s3_secret_key = os.environ.get("S3_SECRET_KEY")
    s3_bucket = os.environ.get("S3_BUCKET")
    s3_region = os.environ.get("S3_REGION", "us-east-1")

    if not all([s3_endpoint, s3_access_key, s3_secret_key, s3_bucket]):
        raise ValueError(
            "Missing required env vars: S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET"
        )

    artifact_dir = args.artifact_dir
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

    active_file = artifact_dir / "ACTIVE"
    if args.version:
        version = args.version
    else:
        if not active_file.exists():
            raise FileNotFoundError(f"ACTIVE file not found: {active_file}")
        version = active_file.read_text(encoding="utf-8").strip()
        if not version:
            raise ValueError(f"ACTIVE file is empty: {active_file}")

    version_dir = artifact_dir / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Version directory not found: {version_dir}")

    s3_client = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        region_name=s3_region,
    )

    try:
        s3_client.head_bucket(Bucket=s3_bucket)
    except Exception:
        print(f"Bucket {s3_bucket} does not exist, creating")
        try:
            s3_client.create_bucket(Bucket=s3_bucket)
        except Exception:
            raise

    uploaded_files = 0

    for local_path in version_dir.rglob("*"):
        if local_path.is_dir():
            continue

        parts = local_path.relative_to(version_dir).parts
        if any(part.startswith(".") for part in parts):
            continue

        relative_path = local_path.relative_to(version_dir)
        s3_key = f"{args.prefix}/seq/{version}/{relative_path.as_posix()}"

        print(f"Uploading {local_path} -> s3://{s3_bucket}/{s3_key}")
        s3_client.upload_file(str(local_path), s3_bucket, s3_key)
        uploaded_files += 1

    if active_file.exists():
        active_s3_key = f"{args.prefix}/seq/ACTIVE"
        print(f"Uploading {active_file} -> s3://{s3_bucket}/{active_s3_key}")
        s3_client.upload_file(str(active_file), s3_bucket, active_s3_key)
        uploaded_files += 1

    print(f"\nUploaded {uploaded_files} files to s3://{s3_bucket}/{args.prefix}/seq/{version}/")


if __name__ == "__main__":
    main()
