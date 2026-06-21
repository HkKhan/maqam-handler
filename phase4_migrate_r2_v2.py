"""
phase4_migrate_r2_v2.py  –  Async EveryAyah → Cloudflare R2 migration

Optimizations over v1:
  1. Full asyncio – no thread pool blocking on I/O
  2. Per-directory pre-fetch of existing R2 keys (one list call instead of
     one head_object call per file → saves ~500k API round-trips)
  3. Parallel directories – process 4 folders concurrently
  4. curl_cffi AsyncSession with Chrome impersonation + UA rotation
  5. Adaptive concurrency – semaphore-controlled, backs off on 429s
  6. aioboto3 async S3 client for non-blocking uploads
"""

import asyncio
import os
import sys
import time
import random
import logging

import aioboto3
from botocore.config import Config
from bs4 import BeautifulSoup
from curl_cffi.requests import AsyncSession
from fake_useragent import UserAgent
from tqdm.asyncio import tqdm as atqdm

# ── Config ────────────────────────────────────────────────────────────────────
ACCOUNT_ID        = "63aec1f75e296553a9580d0d75face77"
BUCKET_NAME       = "everyayah-na-east"
BASE_URL          = "https://everyayah.com/data"
R2_ENDPOINT       = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"

AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Tune these for speed vs. politeness
DOWNLOAD_CONCURRENCY  = 40   # concurrent downloads from EveryAyah per dir
DIR_PARALLELISM       = 3    # directories processed in parallel
CHROME_VERSIONS       = ["chrome99", "chrome100", "chrome101", "chrome110", "chrome120"]

ua = UserAgent()
logging.basicConfig(level=logging.WARNING)

# ── Helpers ───────────────────────────────────────────────────────────────────

async def list_r2_keys(s3, prefix: str) -> set[str]:
    """Return the set of object keys already in R2 under this prefix."""
    existing = set()
    paginator = s3.get_paginator("list_objects_v2")
    async for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            existing.add(obj["Key"])
    return existing

async def scrape_dirs(session: AsyncSession) -> list[str]:
    r = await session.get(BASE_URL + "/", impersonate=random.choice(CHROME_VERSIONS),
                          headers={"User-Agent": ua.random})
    soup = BeautifulSoup(r.text, "html.parser")
    dirs = []
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if href.startswith("/data/") and href.endswith("/") and href != "/data/":
            dirs.append(href.split("/data/")[1])  # e.g. "Husary_128kbps/"
    return dirs

async def scrape_files(session: AsyncSession, dir_name: str) -> list[str]:
    url = f"{BASE_URL}/{dir_name}"
    for attempt in range(3):
        try:
            r = await session.get(url, impersonate=random.choice(CHROME_VERSIONS),
                                  headers={"User-Agent": ua.random})
            soup = BeautifulSoup(r.text, "html.parser")
            return [
                a["href"].split("/")[-1]
                for a in soup.find_all("a")
                if a.get("href", "").endswith(".mp3")
            ]
        except Exception:
            await asyncio.sleep(1)
    return []

async def upload_file(
    session: AsyncSession,
    s3,
    sem: asyncio.Semaphore,
    dir_name: str,
    filename: str,
    existing: set[str],
    stats: dict,
):
    s3_key = f"{dir_name}{filename}"
    if s3_key in existing:
        stats["skipped"] += 1
        return

    url = f"{BASE_URL}/{dir_name}{filename}"
    async with sem:
        for attempt in range(5):
            try:
                r = await session.get(
                    url,
                    impersonate=random.choice(CHROME_VERSIONS),
                    headers={"User-Agent": ua.random},
                    timeout=30,
                )
                if r.status_code == 200:
                    await s3.put_object(
                        Bucket=BUCKET_NAME,
                        Key=s3_key,
                        Body=r.content,
                        ContentType="audio/mpeg",
                    )
                    stats["uploaded"] += 1
                    return
                elif r.status_code == 404:
                    stats["missing"] += 1
                    return
                elif r.status_code == 429:
                    wait = 3 * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait)
                else:
                    await asyncio.sleep(1)
            except Exception as e:
                if attempt == 4:
                    stats["failed"] += 1
                    tqdm_log(f"FAIL {s3_key}: {e}")
                    return
                await asyncio.sleep(2 ** attempt)

def tqdm_log(msg):
    print(f"\n  ⚠ {msg}", file=sys.stderr)

async def process_directory(
    session: AsyncSession,
    s3,
    dir_name: str,
    idx: int,
    total: int,
    global_stats: dict,
):
    files = await scrape_files(session, dir_name)
    if not files:
        tqdm_log(f"[{idx}/{total}] {dir_name} – no files found, skipping")
        return

    # Pre-fetch existing R2 keys for this directory (one list call)
    existing = await list_r2_keys(s3, dir_name)
    to_upload = [f for f in files if f"{dir_name}{f}" not in existing]

    stats = {"uploaded": 0, "skipped": len(files) - len(to_upload), "missing": 0, "failed": 0}

    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    tasks = [
        upload_file(session, s3, sem, dir_name, f, existing, stats)
        for f in to_upload
    ]

    desc = f"[{idx}/{total}] {dir_name.rstrip('/')[:35]}"
    for coro in atqdm.as_completed(tasks, desc=desc, total=len(tasks), leave=False):
        await coro

    global_stats["uploaded"] += stats["uploaded"]
    global_stats["skipped"]  += stats["skipped"]
    global_stats["failed"]   += stats["failed"]
    print(
        f"\n✓ {dir_name.rstrip('/')}  "
        f"↑{stats['uploaded']} skip={stats['skipped']} miss={stats['missing']} fail={stats['failed']}"
    )

async def main():
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("ERROR: Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars.")
        return

    boto_session = aioboto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )

    async with AsyncSession() as session, \
               boto_session.client(
                   "s3",
                   endpoint_url=R2_ENDPOINT,
                   config=Config(signature_version="s3v4", max_pool_connections=100),
               ) as s3:

        dirs = await scrape_dirs(session)
        total = len(dirs)
        print(f"Found {total} directories on EveryAyah.")

        global_stats = {"uploaded": 0, "skipped": 0, "failed": 0}
        t0 = time.time()

        # Process DIR_PARALLELISM directories at a time
        for i in range(0, total, DIR_PARALLELISM):
            batch = dirs[i: i + DIR_PARALLELISM]
            await asyncio.gather(*[
                process_directory(session, s3, d, i + j + 1, total, global_stats)
                for j, d in enumerate(batch)
            ])

        elapsed = time.time() - t0
        print(
            f"\n🏁 Done in {elapsed/3600:.1f}h  |  "
            f"↑{global_stats['uploaded']} uploaded  "
            f"⏩{global_stats['skipped']} skipped  "
            f"✗{global_stats['failed']} failed"
        )

if __name__ == "__main__":
    asyncio.run(main())
