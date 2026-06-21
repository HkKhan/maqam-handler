"""
handler.py  –  RunPod Serverless Handler for Maqam Voice Fingerprinting

Cold-start behavior:
  - Downloads master_fatiha_db.json from Cloudflare R2 once and caches in /tmp
  - Downloads fatiha_reference_features.json for word-level feedback
  - Loads WavLM-Large, Whisper, and DeepFilterNet once per worker lifecycle

Input JSON payload:
  {
    "audio_b64": "<base64-encoded audio bytes>",   // option A
    "audio_url": "https://...",                    // option B (public URL)
    "top_k": 10                                    // optional, default 10
  }

Output JSON:
  {
    "best_match": {"name": "...", "score": 0.91},
    "rankings": [{"name": "...", "score": 0.91}, ...],
    "transcription": "...",
    "fatiha_verified": true,
    "word_feedback": [
      {"word": "الحمد", "duration_advice": "good", "pitch_advice": "go higher", ...},
      ...
    ]
  } 
"""

import os, json, base64, tempfile, logging
import numpy as np
import torch
import librosa
import boto3
import soundfile as sf
import runpod
import requests as req

from botocore.config import Config
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, pipeline
from df.enhance import enhance, init_df, load_audio
from thefuzz import fuzz

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("maqam")

# ── Constants ─────────────────────────────────────────────────────────────────
CLEAN_RMS_THRESHOLD     = 0.01
DB_LOCAL_PATH           = "/tmp/master_fatiha_db.json"
R2_DB_KEY               = "master_fatiha_db.json"
REF_FEATURES_LOCAL_PATH = "/tmp/fatiha_reference_features.json"
R2_REF_FEATURES_KEY     = "fatiha_reference_features.json"

# Thresholds for feedback advice generation
DURATION_TOLERANCE = 0.25   # ±25% of reference word duration is "good"
PITCH_TOLERANCE    = 0.5    # ±0.5 z-score units is "good"

# ── Environment ───────────────────────────────────────────────────────────────
R2_ENDPOINT   = os.environ["R2_ENDPOINT_URL"]
R2_BUCKET     = os.environ["R2_BUCKET_NAME"]
R2_KEY_ID     = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET     = os.environ["R2_SECRET_ACCESS_KEY"]

# ── Module-level (cold-start) singletons ─────────────────────────────────────
_models = {}

def get_models():
    """Load all models once per worker. Called lazily on first request."""
    if _models:
        return _models

    log.info("Cold start: downloading DB from R2...")
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_KEY_ID,
        aws_secret_access_key=R2_SECRET,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    s3.download_file(R2_BUCKET, R2_DB_KEY, DB_LOCAL_PATH)
    log.info("DB downloaded.")

    with open(DB_LOCAL_PATH) as f:
        master_db = json.load(f)

    # Download reference features for word-level feedback
    log.info("Downloading reference features from R2...")
    try:
        s3.download_file(R2_BUCKET, R2_REF_FEATURES_KEY, REF_FEATURES_LOCAL_PATH)
        with open(REF_FEATURES_LOCAL_PATH) as f:
            ref_features = json.load(f)
        log.info(f"Reference features loaded for {len(ref_features)} reciters.")
    except Exception as e:
        log.warning(f"Could not load reference features: {e}. Word feedback will be unavailable.")
        ref_features = {}

    log.info("Loading DeepFilterNet...")
    df_model, df_state, _ = init_df()

    log.info("Loading WavLM-Large...")
    processor   = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    wavlm       = WavLMModel.from_pretrained("microsoft/wavlm-large")
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    wavlm       = wavlm.to(device).eval()

    log.info("Loading Whisper (tadabur)...")
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="FaisaI/tadabur-Whisper-Small",
        device=device,
        chunk_length_s=30,
    )

    _models.update({
        "df_model":      df_model,
        "df_state":      df_state,
        "processor":     processor,
        "wavlm":         wavlm,
        "device":        device,
        "whisper":       whisper_pipe,
        "master_db":     master_db,
        "ref_features":  ref_features,
    })
    log.info("All models loaded.")
    return _models

# ── Audio utilities ───────────────────────────────────────────────────────────

def load_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """Write bytes to a tmp file, load with librosa at 16 kHz."""
    suffix = ".mp3"  # librosa/soundfile will auto-detect
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        audio, _ = librosa.load(tmp_path, sr=16000, mono=True)
    finally:
        os.unlink(tmp_path)
    return audio

def is_clean_audio(audio_np: np.ndarray) -> bool:
    return float(np.sqrt(np.mean(audio_np**2))) > CLEAN_RMS_THRESHOLD

def enhance_audio(audio_np, df_model, df_state) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_np, 16000)
        tmp_path = tmp.name
    try:
        df_audio, _ = load_audio(tmp_path, sr=df_state.sr())
        cleaned = enhance(df_model, df_state, df_audio)
        out = cleaned.squeeze().cpu().numpy()
        if df_state.sr() != 16000:
            out = librosa.resample(out, orig_sr=df_state.sr(), target_sr=16000)
    finally:
        os.unlink(tmp_path)
    return out

def extract_embedding(audio_np, processor, wavlm, device) -> np.ndarray:
    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out   = wavlm(**inputs)
        emb   = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        emb   = emb / np.linalg.norm(emb)
    return emb

# ── Word-level feedback ───────────────────────────────────────────────────────

def extract_user_word_features(audio_np, whisper_pipe, sr=16000):
    """Extract word-level features from the user's audio (same method as phase6)."""
    result = whisper_pipe(audio_np, return_timestamps="word")
    chunks = result.get("chunks", [])
    if not chunks:
        return [], 0.0, 0.0

    total_duration = len(audio_np) / sr

    # Global pitch
    f0_global = librosa.yin(
        audio_np.astype(np.float32), fmin=60, fmax=600, sr=sr,
        frame_length=2048, hop_length=512,
    )
    f0_voiced = f0_global[(f0_global > 60) & (f0_global < 600)]
    g_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    g_std  = float(np.std(f0_voiced))  if len(f0_voiced) > 0 else 1.0

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


def align_words(user_words, ref_words):
    """Align user words to reference words using DP (Needleman-Wunsch) for global optimal alignment.
    This prevents desyncs when words are hallucinated or missing."""
    n = len(user_words)
    m = len(ref_words)
    
    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # Backtrack table: 0 = Match/Sub, 1 = Insert (skip ref), 2 = Delete (skip user)
    ptr = [[0] * (m + 1) for _ in range(n + 1)]
    
    GAP_PENALTY = -20
    
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + GAP_PENALTY
        ptr[i][0] = 2
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + GAP_PENALTY
        ptr[0][j] = 1
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = fuzz.ratio(user_words[i-1]["text"], ref_words[j-1]["text"])
            
            if score < 60:
                match = dp[i-1][j-1] - 30
            else:
                match = dp[i-1][j-1] + score
                
            delete = dp[i-1][j] + GAP_PENALTY
            insert = dp[i][j-1] + GAP_PENALTY
            
            best = max(match, delete, insert)
            dp[i][j] = best
            
            if best == match:
                ptr[i][j] = 0
            elif best == delete:
                ptr[i][j] = 2
            else:
                ptr[i][j] = 1
                
    i, j = n, m
    aligned = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ptr[i][j] == 0:
            score = fuzz.ratio(user_words[i-1]["text"], ref_words[j-1]["text"])
            if score >= 60:
                aligned.append((user_words[i-1], ref_words[j-1]))
            else:
                aligned.append((user_words[i-1], None))
            i -= 1
            j -= 1
        elif i > 0 and ptr[i][j] == 2:
            aligned.append((user_words[i-1], None))
            i -= 1
        else:
            j -= 1
            
    return aligned[::-1]


def generate_word_feedback(user_words, ref_data):
    """Compare user word features against the matched reciter's reference.
    Returns a list of feedback dicts, one per user word."""
    if not ref_data or "words" not in ref_data or not ref_data["words"]:
        return []

    ref_words = ref_data["words"]
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

        # ── Duration comparison ──────────────────────────────────────────
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

        # ── Pitch comparison (z-score normalized) ────────────────────────
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

        # ── Numeric deltas for the frontend ──────────────────────────────
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


# ── RunPod handler ────────────────────────────────────────────────────────────

def handler(job):
    inp = job.get("input", {})

    # ── 1. Get audio bytes ────────────────────────────────────────────────────
    if "audio_b64" in inp:
        audio_bytes = base64.b64decode(inp["audio_b64"])
    elif "audio_url" in inp:
        resp = req.get(inp["audio_url"], timeout=30)
        resp.raise_for_status()
        audio_bytes = resp.content
    else:
        return {"error": "Provide 'audio_b64' (base64) or 'audio_url' in input."}

    top_k = int(inp.get("top_k", 10))

    # ── 2. Load models (cached after first call) ──────────────────────────────
    m = get_models()

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    audio = load_audio_bytes(audio_bytes)
    audio, _ = librosa.effects.trim(audio, top_db=30)

    if is_clean_audio(audio):
        log.info("Studio audio — skipping DeepFilterNet")
        processed = audio
    else:
        log.info("Noisy audio — applying DeepFilterNet")
        processed = enhance_audio(audio, m["df_model"], m["df_state"])

    # ── 4. Transcribe & verify ────────────────────────────────────────────────
    transcription   = m["whisper"](processed)["text"]
    fatiha_verified = fuzz.partial_ratio(transcription, "بسم الله الرحمن الرحيم") >= 60

    # ── 5. WavLM embedding ────────────────────────────────────────────────────
    user_emb = extract_embedding(processed, m["processor"], m["wavlm"], m["device"])

    # ── 6. Cosine similarity ranking ──────────────────────────────────────────
    results = []
    for reciter_id, data in m["master_db"].items():
        master_emb = np.array(data["embedding"])
        score      = float(np.dot(user_emb, master_emb))
        results.append({"id": reciter_id, "name": data.get("name", reciter_id), "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    best_id = results[0]["id"]

    # ── 7. Word-level feedback & Reference Audio URL ──────────────────────────
    word_feedback = []
    ref_audio_url = None
    try:
        ref_features = m.get("ref_features", {})
        
        # master_db keys often lack trailing slash, but ref_features keys might have it
        ref_data = ref_features.get(best_id) or ref_features.get(best_id + "/")

        if ref_data:
            log.info(f"Generating word-level feedback against {ref_data['name']}...")
            user_words, _, _ = extract_user_word_features(processed, m["whisper"])
            word_feedback    = generate_word_feedback(user_words, ref_data)
            log.info(f"Generated feedback for {len(word_feedback)} words.")
            
            # Generate presigned URL for the reference Fatiha audio
            s3 = boto3.client(
                "s3",
                endpoint_url=R2_ENDPOINT,
                aws_access_key_id=R2_KEY_ID,
                aws_secret_access_key=R2_SECRET,
                config=Config(signature_version="s3v4"),
                region_name="auto",
            )
            slug_name = best_id.strip("/")
            ref_audio_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': R2_BUCKET, 'Key': f"fatiha_full/{slug_name}.wav"},
                ExpiresIn=3600
            )
        else:
            log.info(f"No reference features for {best_id}, skipping word feedback.")
    except Exception as e:
        log.warning(f"Word feedback generation failed: {e}")

    return {
        "best_match":       {"name": results[0]["name"], "score": round(results[0]["score"], 4)},
        "rankings":         [{"name": r["name"], "score": round(r["score"], 4)} for r in results[:top_k]],
        "transcription":    transcription,
        "fatiha_verified":  fatiha_verified,
        "word_feedback":    word_feedback,
        "ref_audio_url":    ref_audio_url,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
