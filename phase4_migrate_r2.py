import os
from curl_cffi import requests
import boto3
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from botocore.config import Config
from fake_useragent import UserAgent

ua = UserAgent()

# Configuration
ACCOUNT_ID = "63aec1f75e296553a9580d0d75face77"
BUCKET_NAME = "everyayah-na-east"
BASE_URL = "https://everyayah.com/data"

# These must be set via environment variables
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4', max_pool_connections=50),
        region_name="auto"
    )

def scrape_directories():
    print("Scraping directories from EveryAyah...")
    response = requests.get(BASE_URL + "/", impersonate="chrome", headers={"User-Agent": ua.random})
    soup = BeautifulSoup(response.text, 'html.parser')
    
    directories = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.startswith("/data/") and href.endswith("/") and href != "/data/":
            dir_name = href.split("/data/")[1]
            directories.append(dir_name)
    
    print(f"Found {len(directories)} reciter directories.")
    return directories

def get_files_in_directory(dir_name):
    url = f"{BASE_URL}/{dir_name}"
    response = requests.get(url, impersonate="chrome", headers={"User-Agent": ua.random})
    soup = BeautifulSoup(response.text, 'html.parser')
    
    files = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith(".mp3"):
            filename = href.split("/")[-1]
            files.append(filename)
    return files

def upload_file(s3_client, dir_name, filename):
    import time
    s3_key = f"{dir_name}{filename}"
    
    # Check if already uploaded (Idempotency)
    try:
        s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        return True, s3_key, "Skipped (Already exists)"
    except Exception:
        pass # Doesn't exist, proceed to upload
        
    url = f"{BASE_URL}/{dir_name}{filename}"
    for attempt in range(5):
        try:
            response = requests.get(
                url, 
                impersonate="chrome", 
                headers={"User-Agent": ua.random},
                timeout=30
            )
            if response.status_code == 200:
                s3_client.put_object(
                    Bucket=BUCKET_NAME, 
                    Key=s3_key, 
                    Body=response.content,
                    ContentType='audio/mpeg'
                )
                return True, s3_key, "Uploaded"
            elif response.status_code == 404:
                return False, s3_key, "HTTP 404"
            elif response.status_code == 429:
                time.sleep(3 * (2 ** attempt)) # Exponential backoff for rate limits
            else:
                time.sleep(1)
        except Exception as e:
            if attempt == 4:
                return False, s3_key, str(e)
            time.sleep(2)
            
    return False, s3_key, "Max retries exceeded"

def main():
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("ERROR: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in the environment.")
        return

    s3 = get_s3_client()
    directories = scrape_directories()
    
    total_files_processed = 0
    
    # We process directory by directory to manage memory and progress
    for idx, dir_name in enumerate(directories):
        print(f"\n[{idx+1}/{len(directories)}] Processing {dir_name}")
        files = get_files_in_directory(dir_name)
        print(f"  Found {len(files)} files to upload.")
        
        # Concurrent uploads with rate limit handling
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(upload_file, s3, dir_name, f): f for f in files}
            
            for future in tqdm(as_completed(futures), total=len(files), desc=f"Uploading {dir_name}"):
                success, key, msg = future.result()
                if not success:
                    tqdm.write(f"Failed to upload {key}: {msg}")
                else:
                    total_files_processed += 1
                    
    print(f"\nMigration complete! Processed {total_files_processed} files.")

if __name__ == '__main__':
    main()
