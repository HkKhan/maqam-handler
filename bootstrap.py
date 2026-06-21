import os
import sys
import boto3
import subprocess
from botocore.config import Config

# R2 Environment Variables must be available on the RunPod Endpoint
R2_ENDPOINT   = os.environ.get("R2_ENDPOINT_URL")
R2_BUCKET     = os.environ.get("R2_BUCKET_NAME")
R2_KEY_ID     = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET     = os.environ.get("R2_SECRET_ACCESS_KEY")

HANDLER_KEY   = "handler.py"
LOCAL_HANDLER = "/app/handler.py"

def main():
    print("Bootstrap: Initializing dynamic handler loading...")
    
    if not all([R2_ENDPOINT, R2_BUCKET, R2_KEY_ID, R2_SECRET]):
        print("Bootstrap Warning: Missing R2 credentials. Falling back to baked handler.py if it exists.")
    else:
        try:
            print(f"Bootstrap: Downloading {HANDLER_KEY} from R2 bucket {R2_BUCKET}...")
            s3 = boto3.client(
                "s3",
                endpoint_url=R2_ENDPOINT,
                aws_access_key_id=R2_KEY_ID,
                aws_secret_access_key=R2_SECRET,
                config=Config(signature_version="s3v4"),
                region_name="auto",
            )
            s3.download_file(R2_BUCKET, HANDLER_KEY, LOCAL_HANDLER)
            print("Bootstrap: Successfully downloaded latest handler.py")
        except Exception as e:
            print(f"Bootstrap Error: Failed to download handler from R2: {e}")
            print("Bootstrap: Falling back to baked handler.py if it exists.")
            
    if not os.path.exists(LOCAL_HANDLER):
        print("Bootstrap Fatal: No handler.py found locally or in R2. Exiting.")
        sys.exit(1)
        
    print("Bootstrap: Executing handler.py...")
    # Exec the handler so it replaces the bootstrap process completely
    os.execl(sys.executable, sys.executable, "-u", LOCAL_HANDLER)

if __name__ == "__main__":
    main()
