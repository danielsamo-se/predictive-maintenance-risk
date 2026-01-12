# Cloud artifacts (MinIO / S3)

Create a local .env. Use .env.example as a template.

Publish the ACTIVE model
pip install -e ".[cloud]"
python scripts/publish_artifacts.py

Uploads:
s3://$S3_BUCKET/pmrisk/seq/ACTIVE
s3://$S3_BUCKET/pmrisk/seq/<version>/...

MinIO Console: http://localhost:9001

Verify
Open the bucket in the MinIO Console and check pmrisk/seq/<version>/.