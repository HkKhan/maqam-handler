"""
phase5_build_db_r2_full.py  –  Build a comprehensive voice fingerprint DB
                               by streaming audio directly from Cloudflare R2.

Strategy:
  - For each unique reciter, download 1 ayah from EACH surah (up to 114 samples).
  - Mean-pool all WavLM embeddings → single 1024-dim speaker vector.
  - Averaging across 114 phonetically-diverse surahs causes phonetic content to
    cancel out, leaving pure speaker identity in the vector.
  - Saves master_fatiha_db_full.json and uploads it to R2.

Runtime estimate:  ~1.5–2h on MPS (Apple Silicon).
"""

import os, io, json, boto3, librosa, torch, numpy as np
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
R2_ENDPOINT   = os.environ["R2_ENDPOINT_URL"]
R2_BUCKET     = os.environ["R2_BUCKET_NAME"]
R2_KEY_ID     = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET     = os.environ["R2_SECRET_ACCESS_KEY"]

DB_LOCAL      = "master_db_full.json"
DB_R2_KEY     = "master_db_full.json"
DOWNLOAD_WORKERS = 12    # parallel R2 downloads

# ── Quran structure: surah → ayah count ──────────────────────────────────────
QURAN = {
     1:7,  2:286,  3:200,  4:176,  5:120,  6:165,  7:206,  8:75,  9:129,
    10:109, 11:123, 12:111, 13:43,  14:52,  15:99,  16:128, 17:111, 18:110,
    19:98,  20:135, 21:112, 22:78,  23:118, 24:64,  25:77,  26:227, 27:93,
    28:88,  29:69,  30:60,  31:34,  32:30,  33:73,  34:54,  35:45,  36:83,
    37:182, 38:88,  39:75,  40:85,  41:54,  42:53,  43:89,  44:59,  45:37,
    46:35,  47:38,  48:29,  49:18,  50:45,  51:60,  52:49,  53:62,  54:55,
    55:78,  56:96,  57:29,  58:22,  59:24,  60:13,  61:14,  62:11,  63:11,
    64:18,  65:12,  66:12,  67:30,  68:52,  69:52,  70:44,  71:28,  72:28,
    73:20,  74:56,  75:40,  76:31,  77:50,  78:40,  79:46,  80:42,  81:29,
    82:19,  83:36,  84:25,  85:22,  86:17,  87:19,  88:26,  89:30,  90:20,
    91:15,  92:21,  93:11,  94:8,   95:8,   96:19,  97:5,   98:8,   99:8,
   100:11, 101:11, 102:8,  103:3,  104:9,  105:5,  106:4,  107:7,  108:3,
   109:6,  110:3,  111:5,  112:4,  113:5,  114:6,
}

# Pick ayah 1 from each surah as the sample point (simplest, reproducible)
# Ayah 1 of every surah = Bismillah/opening, good phonetic anchor
SAMPLE_AYAHS = {s: 1 for s in QURAN}   # {surah: ayah_number}

# ── Reciter directory → display name mapping ──────────────────────────────────
# Best-quality (highest bitrate, no duplicates) per unique sheikh
RECITERS = {
    # (dir_slug, display_name)
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
    """Download a single ayah from R2, return numpy array at 16kHz or None."""
    key = f"{dir_slug}{surah:03d}{ayah:03d}.mp3"
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
        audio_bytes = obj["Body"].read()
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        return audio
    except Exception:
        return None

# ── WavLM embedding ───────────────────────────────────────────────────────────
def embed(audio_np, processor, model, device):
    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb / np.linalg.norm(emb)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading WavLM-Large...")
    processor   = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large", use_safetensors=True)
    wavlm       = WavLMModel.from_pretrained("microsoft/wavlm-large", use_safetensors=True)
    device    = torch.device("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    wavlm = wavlm.to(device).eval()

    s3  = make_s3()
    db  = {}

    for dir_slug, display_name in tqdm(RECITERS.items(), desc="Reciters"):
        # Build download tasks: 1 ayah per surah
        tasks = [(surah, ayah) for surah, ayah in SAMPLE_AYAHS.items()]

        # Download in parallel
        audios = []
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
            futures = {ex.submit(download_ayah, s3, dir_slug, s, a): (s, a)
                       for s, a in tasks}
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None and len(result) > 0:
                    audios.append(result)

        if not audios:
            tqdm.write(f"  ✗ {display_name} – no audio found, skipping")
            continue

        # Embed each downloaded clip and mean-pool
        embs = []
        for audio in audios:
            try:
                e = embed(audio, processor, wavlm, device)
                embs.append(e)
            except Exception as ex_err:
                tqdm.write(f"  ⚠ embed error: {ex_err}")

        if not embs:
            tqdm.write(f"  ✗ {display_name} – embedding failed")
            continue

        mean_emb = np.mean(np.stack(embs), axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)  # re-normalize after averaging

        db[dir_slug] = {
            "name":      display_name,
            "embedding": mean_emb.tolist(),
            "n_ayahs":   len(embs),
        }
        tqdm.write(f"  ✓ {display_name}  ({len(embs)}/114 ayahs)")

    # Save locally
    with open(DB_LOCAL, "w") as f:
        json.dump(db, f)
    print(f"\n✅ Saved {len(db)} reciters to {DB_LOCAL}")

    # Upload to R2
    print(f"Uploading to R2 as '{DB_R2_KEY}'...")
    s3.upload_file(DB_LOCAL, R2_BUCKET, DB_R2_KEY,
                   ExtraArgs={"ContentType": "application/json"})
    print("✅ Uploaded to R2!")

if __name__ == "__main__":
    main()
