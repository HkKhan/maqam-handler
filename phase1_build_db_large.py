"""
phase1_build_db_large.py  –  Downloads Fatiha from EveryAyah.com for famous reciters,
                              extracts WavLM embeddings, and saves master_fatiha_db.json.

No HF streaming needed. Downloads finish in < 1 minute.
"""

import os, json, io, requests
import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from tqdm import tqdm

DB_PATH = "master_fatiha_db.json"
FATIHA_AYAHS = [f"001{str(i).zfill(3)}.mp3" for i in range(1, 8)]  # 001001.mp3 .. 001007.mp3

# Famous reciters from EveryAyah.com - (name, folder_slug)
# Full list: https://everyayah.com/recitations.html
RECITERS = [
    ("Mahmoud Khalil Al-Husary (Murattal)",  "Husary_128kbps"),
    ("Mahmoud Khalil Al-Husary (Mujawwad)",  "Husary_128kbps_Mujawwad"),
    ("Mahmoud Khalil Al-Husary (Muallim)",   "Husary_Muallim_128kbps"),
    ("Abdul Basit Abdul Samad (Mujawwad)",   "Abdul_Basit_Mujawwad_128kbps"),
    ("Abdul Basit Abdul Samad (Murattal)",   "Abdul_Basit_Murattal_192kbps"),
    ("Mishary Rashid Al-Afasy",              "Alafasy_128kbps"),
    ("Abdurrahman Al-Sudais",                "Abdurrahmaan_As-Sudais_192kbps"),
    ("Saud Al-Shuraim",                      "Saood_ash-Shuraym_128kbps"),
    ("Mohamed Siddiq Al-Minshawi (Mujawwad)","Minshawy_Mujawwad_128kbps"),
    ("Mohamed Siddiq Al-Minshawi (Murattal)","Minshawy_Murattal_128kbps"),
    ("Nasser Al-Qatami",                     "Nasser_Alqatami_128kbps"),
    ("Maher Al-Muaiqly",                     "MaherAlMuaiqly128kbps"),
    ("Yasser Al-Dosari",                     "Yasser_Ad-Dussary_128kbps"),
    ("Saad Al-Ghamdi",                       "Ghamadi_40kbps"),
    ("Muhammad Ayyoub",                      "Muhammad_Ayyoub_128kbps"),
    ("Ali Al-Huthaify",                      "Hudhaify_128kbps"),
    ("Ibrahim Al-Akhdar",                    "Ibrahim_Akhdar_64kbps"),
    ("Salah Al-Budair",                      "Salah_Al_Budair_128kbps"),
    ("Hani Ar-Rifai",                        "Hani_Rifai_192kbps"),
    ("Abu Bakr Ash-Shatri",                  "Abu_Bakr_Ash-Shaatree_128kbps"),
    ("Fares Abbad",                          "Fares_Abbad_64kbps"),
    ("Abdullah Basfar",                      "Abdullah_Basfar_192kbps"),
    ("Abdullah Matroud",                     "Abdullah_Matroud_128kbps"),
    ("Ahmad ibn Ali Al-Ajamy",               "Ahmed_ibn_Ali_al-Ajamy_128kbps_ketaballah.net"),
    ("Abdullaah Al-Juhaynee",                "Abdullaah_3awwaad_Al-Juhaynee_128kbps"),
    ("Khalid Al-Qahtani",                    "Khaalid_Abdullaah_al-Qahtaanee_192kbps"),
    ("Ali Hajjaj Al-Suesy",                  "Ali_Hajjaj_AlSuesy_128kbps"),
    ("Ali Jaber",                            "Ali_Jaber_64kbps"),
    ("Ayman Sowaid",                         "Ayman_Sowaid_64kbps"),
    ("Karim Mansoori",                       "Karim_Mansoori_40kbps"),
]

BASE_URL = "https://everyayah.com/data"

import time

def download_ayah_audio(reciter_folder, ayah_file, session):
    url = f"{BASE_URL}/{reciter_folder}/{ayah_file}"
    for attempt in range(3):
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 1000:
                return r.content
            if r.status_code == 404:
                return None  # Definitely missing
            time.sleep(0.5)
        except Exception as e:
            if attempt == 2:
                print(f"  ✗ Connection error on {ayah_file}: {e}")
            time.sleep(1)
    return None

def main():
    # ── Models ────────────────────────────────────────────────────────────
    print("Loading WavLM-large...", flush=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    wavlm    = WavLMModel.from_pretrained("microsoft/wavlm-large")
    device   = ("mps" if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    wavlm = wavlm.to(device).eval()
    resamplers = {}

    def embed(audio_np, sr):
        if len(audio_np) < 800:
            return None
        t = torch.from_numpy(audio_np.astype(np.float32)).to(device)
        if t.ndim > 1:
            t = t.mean(0)
        if sr != 16000:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(sr, 16000).to(device)
            t = resamplers[sr](t)
        inp = processor(t.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = wavlm(**inp)
        return out.last_hidden_state.mean(1).squeeze().cpu().numpy()

    # ── Download & embed ──────────────────────────────────────────────────
    master_db = {}
    for name, folder in tqdm(RECITERS, desc="Reciters"):
        session = requests.Session()
        session.headers["User-Agent"] = "Mozilla/5.0"
        embs = []
        for ayah in FATIHA_AYAHS:
            raw = download_ayah_audio(folder, ayah, session)
            if raw is None:
                tqdm.write(f"  ⚠ {name} / {ayah} – not found, skipping")
                continue
            try:
                audio, sr = sf.read(io.BytesIO(raw))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
            except Exception:
                try:
                    audio, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
                except Exception as e:
                    tqdm.write(f"  ✗ decode error for {name}/{ayah}: {e}")
                    continue
            emb = embed(audio, sr)
            if emb is not None:
                embs.append(emb)

        if not embs:
            tqdm.write(f"  ✗ Skipping {name} – no audio obtained")
            continue

        mean_emb = np.mean(embs, axis=0)
        norm_emb = mean_emb / np.linalg.norm(mean_emb)
        # Use folder slug directly as key (no collision between variants)
        key = folder
        master_db[key] = {"embedding": norm_emb.tolist(), "name": name}
        tqdm.write(f"  ✓ {name} ({len(embs)}/7 ayahs)")

    with open(DB_PATH, "w") as f:
        json.dump(master_db, f)
    print(f"\n✅ Saved {len(master_db)} reciters to {DB_PATH}", flush=True)


if __name__ == "__main__":
    main()
