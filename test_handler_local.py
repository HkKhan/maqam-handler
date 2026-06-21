"""
test_handler_local.py  –  Test handler.py logic locally without Docker/RunPod.

Usage:
    source venv/bin/activate
    python test_handler_local.py 003.mp3
"""
import base64
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Make sure R2 env vars are available for handler
os.environ.setdefault("R2_ENDPOINT_URL", os.getenv("R2_ENDPOINT_URL", ""))
os.environ.setdefault("R2_BUCKET_NAME",  os.getenv("R2_BUCKET_NAME", ""))
os.environ.setdefault("R2_ACCESS_KEY_ID", os.getenv("R2_ACCESS_KEY_ID", ""))
os.environ.setdefault("R2_SECRET_ACCESS_KEY", os.getenv("R2_SECRET_ACCESS_KEY", ""))

# Remove any stale cached DB so we exercise the real R2 download path
if os.path.exists("/tmp/master_fatiha_db.json"):
    os.remove("/tmp/master_fatiha_db.json")
    print("Cleared stale /tmp/master_fatiha_db.json — will re-download from R2")

from handler import handler, get_models  # noqa: E402  (must be after env vars)

audio_path = sys.argv[1] if len(sys.argv) > 1 else "003.mp3"
with open(audio_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

print(f"Testing handler with: {audio_path}")
result = handler({"input": {"audio_b64": audio_b64, "top_k": 10}})

import json
print(json.dumps(result, ensure_ascii=False, indent=2))
