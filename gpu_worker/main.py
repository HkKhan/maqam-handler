import os
import json
import tempfile
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage, firestore
import torch
import librosa
import soundfile as sf
import difflib
from thefuzz import fuzz
import re

app = FastAPI()

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "maqam-chirp-backend-1782142443")
GCS_BUCKET = os.environ.get("GCS_AUDIO_BUCKET", "maqam-audio")

db = firestore.Client(project=PROJECT_ID)
gcs = storage.Client(project=PROJECT_ID)

# ── Load models at startup (cached in container memory while warm) ──────────

print("Loading faster-whisper...")
from faster_whisper import WhisperModel
# Use "small" for Arabic; runs on CPU or GPU automatically
whisper_model = WhisperModel(
    "small",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8",
)

print("Loading WavLM-large...")
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
wavlm_model.eval()
if torch.cuda.is_available():
    wavlm_model = wavlm_model.cuda()

print("Loading DeepFilterNet...")
from df.enhance import enhance, init_df, load_audio, save_audio
df_model, df_state, _ = init_df()

# ── Pre-load reciter embeddings from GCS ────────────────────────────────────

def load_databases():
    bucket = gcs.bucket(GCS_BUCKET)
    
    # Load master_fatiha_db.json
    blob = bucket.blob("db/master_fatiha_db.json")
    master_db_str = blob.download_as_string()
    master_db = json.loads(master_db_str)
    
    # Load fatiha_reference_features.json
    blob2 = bucket.blob("db/fatiha_reference_features.json")
    try:
        ref_features_str = blob2.download_as_string()
        ref_features = json.loads(ref_features_str)
    except Exception as e:
        print(f"Warning: could not load reference features from GCS: {e}")
        ref_features = {}
        
    return master_db, ref_features

master_db, ref_features = load_databases()
print(f"Loaded {len(master_db)} reciter embeddings and {len(ref_features)} reference features.")

# ── Audio and Word Feedback Utilities ───────────────────────────────────────

CLEAN_RMS_THRESHOLD = 0.01
DURATION_TOLERANCE = 0.25
PITCH_TOLERANCE = 0.5

def is_clean_audio(audio_np: np.ndarray) -> bool:
    return float(np.sqrt(np.mean(audio_np**2))) > CLEAN_RMS_THRESHOLD

def normalize_arabic(text):
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    return text

def pre_split_user_words(user_words):
    split_words = []
    for uw in user_words:
        text = uw["text"].strip()
        parts = text.split()
        if len(parts) <= 1:
            split_words.append(uw)
        else:
            total_dur = uw.get("duration_s", 1.0)
            part_dur = total_dur / len(parts)
            curr_start = uw.get("start_s", 0.0)
            for part in parts:
                new_uw = uw.copy()
                new_uw["text"] = part
                new_uw["start_s"] = round(curr_start, 3)
                new_uw["end_s"] = round(curr_start + part_dur, 3)
                new_uw["duration_s"] = round(part_dur, 3)
                new_uw["artificially_split"] = True
                split_words.append(new_uw)
                curr_start += part_dur
    return split_words

def align_words(user_words, ref_words):
    u_texts = [normalize_arabic(w["text"]) for w in user_words]
    r_texts = [normalize_arabic(w["text"]) for w in ref_words]
    
    sm = difflib.SequenceMatcher(None, u_texts, r_texts)
    aligned = []
    
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                aligned.append((user_words[i1 + k], ref_words[j1 + k]))
        elif tag == 'replace':
            u_seg = user_words[i1:i2]
            r_seg = ref_words[j1:j2][:]
            for u in u_seg:
                best_r, best_score = None, 0
                for r in r_seg:
                    score = fuzz.ratio(u["text"], r["text"])
                    if score > best_score:
                        best_score = score
                        best_r = r
                if best_score >= 60:
                    aligned.append((u, best_r))
                    r_seg.remove(best_r)
                else:
                    aligned.append((u, None))
        elif tag == 'delete':
            for k in range(i1, i2):
                aligned.append((user_words[k], None))
                
    return aligned

def extract_user_word_features(audio_np, user_words, sr=16000):
    if not user_words:
        return [], 0.0, 0.0

    total_duration = len(audio_np) / sr
    f0_global = librosa.yin(
        audio_np.astype(np.float32), fmin=60, fmax=600, sr=sr,
        frame_length=2048, hop_length=512,
    )
    f0_voiced = f0_global[(f0_global > 60) & (f0_global < 600)]
    g_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    g_std  = float(np.std(f0_voiced))  if len(f0_voiced) > 0 else 1.0

    words = []
    for w in user_words:
        start = w["start_s"]
        end = w["end_s"]
        text = w["text"]
        if start is None or end is None:
            continue
        if end <= start:
            end = start + 0.1

        start_sample = int(start * sr)
        end_sample   = min(int(end * sr), len(audio_np))
        segment      = audio_np[start_sample:end_sample]
        if len(segment) < 512:
            continue

        duration_s    = end - start
        duration_norm = duration_s / total_duration if total_duration > 0 else 0.0

        try:
            f0_word = librosa.yin(
                segment.astype(np.float32), fmin=60, fmax=600, sr=sr,
                frame_length=min(2048, len(segment)),
                hop_length=min(512, len(segment) // 4 or 1),
            )
            f0_v = f0_word[(f0_word > 60) & (f0_word < 600)]
            pitch_mean = float(np.mean(f0_v)) if len(f0_v) > 0 else g_mean
        except Exception:
            pitch_mean = g_mean

        pitch_norm = (pitch_mean - g_mean) / g_std if g_std > 0 else 0.0
        rms = float(np.sqrt(np.mean(segment ** 2)))
        energy_db = float(20 * np.log10(rms + 1e-10))

        words.append({
            "text": text,
            "start_s": round(start, 3),
            "end_s": round(end, 3),
            "duration_s": round(duration_s, 3),
            "duration_norm": round(duration_norm, 4),
            "pitch_mean_hz": round(pitch_mean, 1),
            "pitch_mean_norm": round(pitch_norm, 3),
            "energy_db": round(energy_db, 1),
        })

    return words, g_mean, g_std

def generate_word_feedback(user_words, ref_data):
    if not ref_data or "words" not in ref_data or not ref_data["words"]:
        return []

    ref_words = ref_data["words"]
    user_words = pre_split_user_words(user_words)
    aligned   = align_words(user_words, ref_words)
    feedback  = []

    for uw, rw in aligned:
        entry = {"word": uw["text"]}

        if rw is None:
            entry["duration_advice"] = "no reference"
            entry["pitch_advice"]    = "no reference"
            entry["detail"]          = "Could not match to a reference word."
            feedback.append(entry)
            continue

        if rw["duration_norm"] > 0:
            dur_ratio = uw["duration_norm"] / rw["duration_norm"]
        else:
            dur_ratio = 1.0

        if dur_ratio < (1.0 - DURATION_TOLERANCE):
            entry["duration_advice"] = "elongate"
            pct = round((1.0 - dur_ratio) * 100)
            entry["duration_detail"] = f"Your word is ~{pct}% shorter than {ref_data['name']}. Try holding it longer."
        elif dur_ratio > (1.0 + DURATION_TOLERANCE):
            entry["duration_advice"] = "shorten"
            pct = round((dur_ratio - 1.0) * 100)
            entry["duration_detail"] = f"Your word is ~{pct}% longer than {ref_data['name']}. Try shortening it."
        else:
            entry["duration_advice"] = "good"
            entry["duration_detail"] = "Duration matches well."

        pitch_diff = uw["pitch_mean_norm"] - rw["pitch_mean_norm"]

        if pitch_diff < -PITCH_TOLERANCE:
            entry["pitch_advice"] = "go higher"
            entry["pitch_detail"] = f"Your pitch is lower than {ref_data['name']} on this word. Try raising your tone."
        elif pitch_diff > PITCH_TOLERANCE:
            entry["pitch_advice"] = "go lower"
            entry["pitch_detail"] = f"Your pitch is higher than {ref_data['name']} on this word. Try lowering your tone."
        else:
            entry["pitch_advice"] = "good"
            entry["pitch_detail"] = "Pitch matches well."

        entry["user_start_s"]     = uw.get("start_s", 0)
        entry["user_end_s"]       = uw.get("end_s", 0)
        entry["ref_start_s"]      = rw.get("start", 0)
        entry["ref_end_s"]        = rw.get("end", 0)
        entry["user_duration_s"]  = uw["duration_s"]
        entry["ref_duration_s"]   = rw["duration_s"]
        entry["user_pitch_hz"]    = uw["pitch_mean_hz"]
        entry["ref_pitch_hz"]     = rw["pitch_mean_hz"]

        feedback.append(entry)

    return feedback


# ── Processing endpoint ──────────────────────────────────────────────────────

@app.post("/process")
async def process(request: Request):
    body = await request.json()
    job_id = body.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing job_id")

    job_ref = db.collection("jobs").document(job_id)
    job_ref.update({"status": "processing"})

    try:
        result = await run_pipeline(job_id)
        job_ref.update({"status": "done", "result": result})
        return {"ok": True}
    except Exception as e:
        job_ref.update({"status": "error", "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

async def run_pipeline(job_id: str) -> dict:
    # 1. Download audio from GCS
    job_doc = db.collection("jobs").document(job_id).get().to_dict()
    gcs_path = job_doc["gcs_path"]
    bucket = gcs.bucket(GCS_BUCKET)
    
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        bucket.blob(gcs_path).download_to_filename(tmp.name)
        raw_path = tmp.name

    # Load and clean audio
    audio_np, sr = librosa.load(raw_path, sr=16000, mono=True)
    audio_np, _ = librosa.effects.trim(audio_np, top_db=30)
    
    wav_path = raw_path.replace(".webm", ".wav")
    sf.write(wav_path, audio_np, int(sr))

    if is_clean_audio(audio_np):
        processed = audio_np
        enhanced_path = wav_path
    else:
        # DeepFilterNet enhancement
        audio_df, _ = load_audio(wav_path, sr=df_state.sr())
        enhanced = enhance(df_model, df_state, audio_df)
        enhanced_path = wav_path.replace(".wav", "_enhanced.wav")
        sf.write(enhanced_path, enhanced.squeeze().cpu().numpy(), df_state.sr())
        processed, _ = librosa.load(enhanced_path, sr=16000, mono=True)

    # 2. faster-whisper transcription + word timestamps
    segments, info = whisper_model.transcribe(
        enhanced_path,
        language="ar",
        word_timestamps=True,
        beam_size=5,
    )
    
    user_words = []
    full_text = ""
    for seg in segments:
        full_text += seg.text + " "
        if seg.words:
            for w in seg.words:
                user_words.append({
                    "text": w.word,
                    "start_s": round(w.start, 3),
                    "end_s": round(w.end, 3),
                    "duration_s": round(w.end - w.start, 3)
                })
    
    full_text = full_text.strip()
    fatiha_verified = fuzz.partial_ratio(full_text, "بسم الله الرحمن الرحيم") >= 60

    if not fatiha_verified:
        return {
            "transcription": full_text,
            "error": "Could not confidently detect Surah Al-Fatiha in this audio."
        }

    # 3. WavLM embedding (GPU)
    inputs = wavlm_processor(
        processed, sampling_rate=16000, return_tensors="pt", padding=True
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = wavlm_model(**inputs)
    user_emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    user_emb = user_emb / np.linalg.norm(user_emb)

    # 4. Reciter matching
    results = []
    for reciter_id, data in master_db.items():
        master_emb = np.array(data["embedding"])
        score = float(np.dot(user_emb, master_emb))
        results.append({"id": reciter_id, "name": data.get("name", reciter_id), "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    best_id = results[0]["id"]
    top_k = 10
    
    # 5. Word-level feedback
    word_feedback = []
    ref_audio_url = None
    
    ref_data = ref_features.get(best_id) or ref_features.get(best_id + "/")
    if ref_data:
        user_words_feat, _, _ = extract_user_word_features(processed, user_words)
        word_feedback = generate_word_feedback(user_words_feat, ref_data)
        
        # Optionally generate presigned URL for reference audio from GCS
        # Note: requires the reference audios to be uploaded to GCS. 
        # We will assume they are in gs://maqam-audio/fatiha_full/{slug}.wav later.
        try:
            slug_name = best_id.strip("/")
            ref_blob = bucket.blob(f"fatiha_full/{slug_name}.wav")
            if ref_blob.exists():
                ref_audio_url = ref_blob.generate_signed_url(version="v4", expiration=3600, method="GET")
        except Exception as e:
            print(f"Failed to generate presigned URL: {e}")

    return {
        "best_match": {"name": results[0]["name"], "score": round(results[0]["score"], 4)},
        "rankings": [{"name": r["name"], "score": round(r["score"], 4)} for r in results[:top_k]],
        "transcription": full_text,
        "fatiha_verified": fatiha_verified,
        "word_feedback": word_feedback,
        "ref_audio_url": ref_audio_url,
    }

@app.get("/health")
async def health():
    return {"ok": True, "gpu": torch.cuda.is_available()}
