"""
phase6_build_reference_features.py  –  Build precomputed word-level reference features
                                       for all Sheikh Fatiha recitations.

For each sheikh in the R2 bucket, this script:
  1. Downloads their Fatiha (Surah 1, ayahs 1–7) audio files from Cloudflare R2.
  2. Concatenates them into a single continuous Fatiha audio.
  3. Runs Whisper (FaisaI/tadabur-Whisper-Small) with `return_timestamps="word"` to get
     per-word timestamps.
  4. Extracts pitch (F0) using `librosa.yin` for each word segment.
  5. Computes normalized duration and pitch statistics per word.
  6. Saves all reference features as `fatiha_reference_features.json` and uploads to R2.

The resulting JSON is keyed by reciter directory slug and contains:
  {
    "reciter_slug/": {
      "name": "Display Name",
      "words": [
        {
          "text": "الحمد",
          "start": 0.0,
          "end": 0.68,
          "duration_s": 0.68,
          "duration_norm": 0.045,  // fraction of total recitation
          "pitch_mean_hz": 180.5,
          "pitch_std_hz": 12.3,
          "pitch_mean_norm": 0.0,  // z-scored within this reciter
          "energy_db": -12.5
        },
        ...
      ],
      "total_duration_s": 35.0,
      "global_pitch_mean_hz": 175.0,
      "global_pitch_std_hz": 30.0
    }
  }
"""

import os, io, json, boto3, librosa, torch, numpy as np, warnings
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import pipeline
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ────────────────────────────────────────────────────────────────────
R2_ENDPOINT = os.environ["R2_ENDPOINT_URL"]
R2_BUCKET   = os.environ["R2_BUCKET_NAME"]
R2_KEY_ID   = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET   = os.environ["R2_SECRET_ACCESS_KEY"]

DB_LOCAL    = "fatiha_reference_features.json"
DB_R2_KEY   = "fatiha_reference_features.json"
DOWNLOAD_WORKERS = 8

# Fatiha = Surah 1, ayahs 1–7
FATIHA_AYAHS = list(range(1, 8))

# ── Reciter list (same as phase5) ─────────────────────────────────────────────
RECITERS = {
    "AbdulSamad_64kbps_QuranExplorer.Com/":       "Abdul Samad (QuranExplorer)",
    "Abdul_Basit_Mujawwad_128kbps/":              "Abdul Basit Abdul Samad (Mujawwad)",
    "Abdul_Basit_Murattal_192kbps/":              "Abdul Basit Abdul Samad (Murattal)",
    "Abdullaah_3awwaad_Al-Juhaynee_128kbps/":     "Abdullaah Al-Juhaynee",
    "Abdullah_Basfar_192kbps/":                   "Abdullah Basfar",
    "Abdullah_Matroud_128kbps/":                  "Abdullah Matroud",
    "Abdurrahmaan_As-Sudais_192kbps/":            "Abdurrahman Al-Sudais",
    "Abu_Bakr_Ash-Shaatree_128kbps/":             "Abu Bakr Ash-Shatri",
    "Ahmed_Neana_128kbps/":                       "Ahmed Neana",
    "Akram_AlAlaqimy_128kbps/":                   "Akram Al-Alaqimy",
    "Alafasy_128kbps/":                           "Mishary Rashid Al-Afasy",
    "Ali_Hajjaj_AlSuesy_128kbps/":                "Ali Hajjaj Al-Suesy",
    "Ali_Jaber_64kbps/":                          "Ali Jaber",
    "Ayman_Sowaid_64kbps/":                       "Ayman Sowaid",
    "Fares_Abbad_64kbps/":                        "Fares Abbad",
    "Ghamadi_40kbps/":                            "Saad Al-Ghamdi",
    "Hani_Rifai_192kbps/":                        "Hani Ar-Rifai",
    "Hudhaify_128kbps/":                          "Ali Al-Huthaify",
    "Husary_128kbps/":                            "Mahmoud Khalil Al-Husary (Murattal)",
    "Husary_128kbps_Mujawwad/":                   "Mahmoud Khalil Al-Husary (Mujawwad)",
    "Ibrahim_Akhdar_32kbps/":                     "Ibrahim Al-Akhdar",
    "Karim_Mansoori_40kbps/":                     "Karim Mansoori",
    "Khaalid_Abdullaah_al-Qahtaanee_192kbps/":   "Khalid Al-Qahtani",
    "MaherAlMuaiqly128kbps/":                     "Maher Al-Muaiqly",
    "Minshawy_Mujawwad_192kbps/":                 "Muhammad Siddiq Al-Minshawi (Mujawwad)",
    "Minshawy_Murattal_128kbps/":                 "Muhammad Siddiq Al-Minshawi (Murattal)",
    "Mohammad_al_Tablaway_128kbps/":              "Mohammad Al-Tablaway",
    "Muhammad_AbdulKareem_128kbps/":              "Muhammad Abdul Kareem",
    "Muhammad_Ayyoub_128kbps/":                   "Muhammad Ayyoub",
    "Muhammad_Jibreel_128kbps/":                  "Muhammad Jibreel",
    "Muhsin_Al_Qasim_192kbps/":                   "Muhsin Al-Qasim",
    "Mustafa_Ismail_48kbps/":                     "Mustafa Ismail",
    "Nasser_Alqatami_128kbps/":                   "Nasser Al-Qatami",
    "Saood_ash-Shuraym_128kbps/":                 "Saud Al-Shuraim",
    "Salah_Al_Budair_128kbps/":                   "Salah Al-Budair",
    "Salaah_AbdulRahman_Bukhatir_128kbps/":       "Salah Bukhatir",
    "Yasser_Ad-Dussary_128kbps/":                 "Yasser Al-Dosari",
    "ahmed_ibn_ali_al_ajamy_128kbps/":            "Ahmad ibn Ali Al-Ajamy",
    "aziz_alili_128kbps/":                        "Aziz Alili",
    "khalefa_al_tunaiji_64kbps/":                 "Khalefa Al-Tunaiji",
    "mahmoud_ali_al_banna_32kbps/":               "Mahmoud Ali Al-Banna",
    "Nabil_Rifa3i_48kbps/":                       "Nabil Rifai",
    "Sahl_Yassin_128kbps/":                       "Sahl Yassin",
    "Yaser_Salamah_128kbps/":                     "Yaser Salamah",
    "Parhizgar_48kbps/":                          "Parhizgar",
}

# ── S3 client ─────────────────────────────────────────────────────────────────
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
    """Download a single ayah from R2, return (ayah_num, numpy array at 16kHz) or None."""
    key = f"{dir_slug}{surah:03d}{ayah:03d}.mp3"
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
        audio_bytes = obj["Body"].read()
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        return (ayah, audio)
    except Exception:
        return None


def extract_word_features(audio_np, whisper_pipe, sr=16000):
    """
    Given a full Fatiha audio array, extract word-level features:
      - Word text and timestamps from Whisper
      - Pitch (F0) per word segment using librosa.yin
      - RMS energy per word segment
    Returns (words_list, global_pitch_mean, global_pitch_std)
    """
    # 1) Run Whisper with word timestamps
    result = whisper_pipe(audio_np, return_timestamps="word")
    chunks = result.get("chunks", [])

    if not chunks:
        return [], 0.0, 0.0

    total_duration = len(audio_np) / sr

    # 2) Compute global pitch for normalization
    f0_global = librosa.yin(
        audio_np.astype(np.float32),
        fmin=60, fmax=600, sr=sr,
        frame_length=2048, hop_length=512,
    )
    f0_voiced = f0_global[(f0_global > 60) & (f0_global < 600)]
    global_pitch_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    global_pitch_std  = float(np.std(f0_voiced))  if len(f0_voiced) > 0 else 1.0

    # 3) Per-word features
    words = []
    for chunk in chunks:
        text = chunk["text"].strip()
        ts   = chunk.get("timestamp", (None, None))
        start, end = ts

        if start is None or end is None or end <= start:
            continue

        start_sample = int(start * sr)
        end_sample   = min(int(end * sr), len(audio_np))
        segment      = audio_np[start_sample:end_sample]

        if len(segment) < 512:
            continue

        duration_s    = end - start
        duration_norm = duration_s / total_duration if total_duration > 0 else 0.0

        # Pitch for this word
        try:
            f0_word = librosa.yin(
                segment.astype(np.float32),
                fmin=60, fmax=600, sr=sr,
                frame_length=min(2048, len(segment)),
                hop_length=min(512, len(segment) // 4 or 1),
            )
            f0_voiced_word = f0_word[(f0_word > 60) & (f0_word < 600)]
            pitch_mean = float(np.mean(f0_voiced_word)) if len(f0_voiced_word) > 0 else global_pitch_mean
            pitch_std  = float(np.std(f0_voiced_word))  if len(f0_voiced_word) > 0 else 0.0
        except Exception:
            pitch_mean = global_pitch_mean
            pitch_std  = 0.0

        # Normalized pitch (z-score relative to speaker)
        pitch_mean_norm = (pitch_mean - global_pitch_mean) / global_pitch_std if global_pitch_std > 0 else 0.0

        # RMS energy in dB
        rms = float(np.sqrt(np.mean(segment ** 2)))
        energy_db = float(20 * np.log10(rms + 1e-10))

        words.append({
            "text":            text,
            "start":           round(start, 3),
            "end":             round(end, 3),
            "duration_s":      round(duration_s, 3),
            "duration_norm":   round(duration_norm, 4),
            "pitch_mean_hz":   round(pitch_mean, 1),
            "pitch_std_hz":    round(pitch_std, 1),
            "pitch_mean_norm": round(pitch_mean_norm, 3),
            "energy_db":       round(energy_db, 1),
        })

    return words, global_pitch_mean, global_pitch_std


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading Whisper (tadabur)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="FaisaI/tadabur-Whisper-Small",
        device=device if str(device) != "mps" else "cpu",  # Whisper works better on CPU for MPS
        chunk_length_s=30,
    )

    s3 = make_s3()
    db = {}

    for dir_slug, display_name in tqdm(RECITERS.items(), desc="Reciters"):
        # Download all 7 Fatiha ayahs in parallel
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
            tqdm.write(f"  ✗ {display_name} – no Fatiha audio found, skipping")
            continue

        # Concatenate ayahs in order (1–7) with small silence gap
        silence = np.zeros(int(16000 * 0.15))  # 150ms gap between ayahs
        full_audio = []
        for ayah_num in sorted(ayah_audios.keys()):
            full_audio.append(ayah_audios[ayah_num])
            full_audio.append(silence)

        full_fatiha = np.concatenate(full_audio)

        # Extract word-level features
        try:
            words, g_pitch_mean, g_pitch_std = extract_word_features(full_fatiha, whisper_pipe)
        except Exception as e:
            tqdm.write(f"  ⚠ {display_name} – feature extraction failed: {e}")
            continue

        if not words:
            tqdm.write(f"  ✗ {display_name} – no words detected, skipping")
            continue

        total_duration = len(full_fatiha) / 16000.0

        db[dir_slug] = {
            "name":                display_name,
            "words":               words,
            "total_duration_s":    round(total_duration, 2),
            "global_pitch_mean_hz": round(g_pitch_mean, 1),
            "global_pitch_std_hz":  round(g_pitch_std, 1),
        }
        tqdm.write(f"  ✓ {display_name}  ({len(words)} words, {total_duration:.1f}s)")

    # Save locally
    with open(DB_LOCAL, "w") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {len(db)} reciters to {DB_LOCAL}")

    # Upload to R2
    print(f"Uploading to R2 as '{DB_R2_KEY}'...")
    s3.upload_file(DB_LOCAL, R2_BUCKET, DB_R2_KEY,
                   ExtraArgs={"ContentType": "application/json"})
    print("✅ Uploaded to R2!")


if __name__ == "__main__":
    main()
