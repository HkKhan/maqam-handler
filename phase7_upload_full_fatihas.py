import os, io, boto3, librosa, numpy as np, soundfile as sf
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

R2_ENDPOINT = os.environ["R2_ENDPOINT_URL"]
R2_BUCKET   = os.environ["R2_BUCKET_NAME"]
R2_KEY_ID   = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET   = os.environ["R2_SECRET_ACCESS_KEY"]

DOWNLOAD_WORKERS = 8
FATIHA_AYAHS = list(range(1, 8))

from phase6_build_reference_features import RECITERS

def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_KEY_ID,
        aws_secret_access_key=R2_SECRET,
        config=Config(signature_version="s3v4", max_pool_connections=DOWNLOAD_WORKERS + 4),
        region_name="auto",
    )

def download_ayah(s3, dir_slug, surah, ayah):
    key = f"{dir_slug}{surah:03d}{ayah:03d}.mp3"
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
        audio_bytes = obj["Body"].read()
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        return (ayah, audio)
    except Exception:
        return None

def main():
    s3 = make_s3()
    
    for dir_slug, display_name in tqdm(RECITERS.items(), desc="Uploading Full Fatihas"):
        ayah_audios = {}
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
            futures = {
                ex.submit(download_ayah, s3, dir_slug, 1, ayah): ayah
                for ayah in FATIHA_AYAHS
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    ayah_num, audio = result
                    ayah_audios[ayah_num] = audio

        if not ayah_audios:
            tqdm.write(f"  ✗ {display_name} – no audio found")
            continue

        # Concatenate ayahs with same 150ms gap as Phase 6
        silence = np.zeros(int(16000 * 0.15))
        full_audio = []
        for ayah_num in sorted(ayah_audios.keys()):
            full_audio.append(ayah_audios[ayah_num])
            full_audio.append(silence)

        full_fatiha = np.concatenate(full_audio)
        
        # Save to memory buffer as WAV
        wav_io = io.BytesIO()
        sf.write(wav_io, full_fatiha, 16000, format='WAV', subtype='PCM_16')
        wav_io.seek(0)
        
        # Upload to R2 under fatiha_full/
        # Remove trailing slash from dir_slug for filename
        slug_name = dir_slug.strip("/")
        r2_key = f"fatiha_full/{slug_name}.wav"
        
        try:
            s3.upload_fileobj(wav_io, R2_BUCKET, r2_key, ExtraArgs={"ContentType": "audio/wav"})
            tqdm.write(f"  ✓ {display_name} -> {r2_key}")
        except Exception as e:
            tqdm.write(f"  ⚠ {display_name} – upload failed: {e}")

if __name__ == "__main__":
    main()
