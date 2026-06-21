"""
deploy_runpod.py  –  Create (or update) a RunPod Serverless endpoint for Maqam.

Usage:
    python deploy_runpod.py

This script:
  1. Reads credentials from .env
  2. Creates a RunPod Serverless endpoint pointing to your Docker image on GHCR/DockerHub
  3. Sets all required environment variables on the endpoint (R2 keys)
  4. Prints the endpoint ID and a sample curl command to test it

NOTE: You must push the Docker image to a registry first:
    docker build -t <your-registry>/maqam-handler:latest .
    docker push <your-registry>/maqam-handler:latest
Then set DOCKER_IMAGE below.
"""

import os
import runpod
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
DOCKER_IMAGE   = os.getenv("DOCKER_IMAGE", "ghcr.io/hkkhan/maqam-handler:latest")
ENDPOINT_NAME  = "maqam-voice-fingerprinting"

# GPU: RTX 3090 (24GB) is cheapest that comfortably fits all three models
# Options: NVIDIA RTX 3090, NVIDIA RTX A4000, NVIDIA RTX 4090
GPU_IDS        = ["NVIDIA RTX 3090"]   # RunPod GPU type string
MAX_WORKERS    = 3      # scale up to 3 parallel requests
MIN_WORKERS    = 0      # scale to zero when idle (pay nothing at rest)
IDLE_TIMEOUT   = 5      # seconds before scaling down an idle worker

# Env vars to inject into every worker (from .env)
WORKER_ENV = {
    "R2_ENDPOINT_URL":    os.getenv("R2_ENDPOINT_URL"),
    "R2_BUCKET_NAME":     os.getenv("R2_BUCKET_NAME"),
    "R2_ACCESS_KEY_ID":   os.getenv("R2_ACCESS_KEY_ID"),
    "R2_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY"),
    "HF_HOME":            "/tmp/hf_cache",   # writable path on RunPod
}
# ─────────────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set in .env")
        return

    runpod.api_key = api_key

    print(f"Creating/updating endpoint '{ENDPOINT_NAME}'...")
    print(f"  Image:      {DOCKER_IMAGE}")
    print(f"  GPU:        {GPU_IDS}")
    print(f"  Workers:    {MIN_WORKERS}–{MAX_WORKERS}")

    # 1. Create a Template first (required by latest RunPod SDK)
    print(f"Creating template...")
    template = runpod.create_template(
        name=ENDPOINT_NAME + "-template",
        image_name=DOCKER_IMAGE,
        env=WORKER_ENV,
        is_serverless=True
    )
    template_id = template["id"]
    print(f"  Template ID: {template_id}")

    # 2. Create the Endpoint using the template
    print(f"Creating endpoint...")
    endpoint = runpod.create_endpoint(
        name=ENDPOINT_NAME,
        template_id=template_id,
        gpu_ids="AMPERE_24",  # AMPERE_24 represents an RTX 3090 / 24GB GPU
        workers_max=MAX_WORKERS,
        workers_min=MIN_WORKERS,
        idle_timeout=IDLE_TIMEOUT,
    )

    endpoint_id = endpoint["id"]
    print(f"\n✅ Endpoint created!")
    print(f"   ID:  {endpoint_id}")
    print(f"   URL: https://api.runpod.ai/v2/{endpoint_id}/run")
    print(f"\nSample curl test (uses 003.mp3):")
    print(f"""
    python - <<'EOF'
import base64, requests, os
from dotenv import load_dotenv
load_dotenv()

with open("003.mp3", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "https://api.runpod.ai/v2/{endpoint_id}/run",
    headers={{"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]}},
    json={{"input": {{"audio_b64": b64, "top_k": 5}}}},
    timeout=120,
)
print(resp.json())
EOF
""")

    # Save the endpoint ID for future use
    with open(".env", "a") as f:
        f.write(f"\nRUNPOD_ENDPOINT_ID={endpoint_id}\n")
    print(f"   (Saved RUNPOD_ENDPOINT_ID to .env)")

if __name__ == "__main__":
    main()
