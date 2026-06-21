import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

R2_ENDPOINT   = os.environ.get("R2_ENDPOINT_URL")
R2_BUCKET     = os.environ.get("R2_BUCKET_NAME")
R2_KEY_ID     = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET     = os.environ.get("R2_SECRET_ACCESS_KEY")

def main():
    if not all([R2_ENDPOINT, R2_BUCKET, R2_KEY_ID, R2_SECRET]):
        print("Error: Missing R2 credentials in .env")
        return

    print(f"Uploading handler.py to R2 bucket '{R2_BUCKET}'...")
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_KEY_ID,
        aws_secret_access_key=R2_SECRET,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    
    try:
        s3.upload_file("handler.py", R2_BUCKET, "handler.py")
        print("✅ Successfully uploaded handler.py to R2!")
        print("RunPod will instantly use this new code on the next cold start.")
    except Exception as e:
        print(f"Failed to upload: {e}")

if __name__ == "__main__":
    main()
