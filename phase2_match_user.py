"""
phase2_match_user.py  –  Identify the reciter of a user-supplied audio file.

Key fixes:
- Uses `name` field stored directly in master_fatiha_db.json (no separate name_map needed).
- Skips DeepFilterNet if input looks like studio/clean audio (avg RMS > threshold).
- Falls back to DeepFilterNet for noisy/ambient recordings.
"""

import os
import sys
import json
import torch
import numpy as np
import librosa
from df.enhance import enhance, init_df, load_audio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, pipeline
import soundfile as sf
from thefuzz import fuzz

CLEAN_RMS_THRESHOLD = 0.01   # if RMS > this, audio is likely studio-quality → skip DeepFilter

def fuzzy_match(text, target, threshold=60):
    return fuzz.partial_ratio(text, target) >= threshold

def is_clean_audio(audio_np):
    rms = np.sqrt(np.mean(audio_np**2))
    return rms > CLEAN_RMS_THRESHOLD

def main():
    print("Loading models...")
    df_model, df_state, _ = init_df()
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    wavlm_model = wavlm_model.to(device).eval()

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="FaisaI/tadabur-Whisper-Small",
        device=device,
        chunk_length_s=30,
    )

    if not os.path.exists("master_fatiha_db.json"):
        print("Master DB not found. Run phase1_build_db_large.py first.")
        return

    with open("master_fatiha_db.json", "r") as f:
        master_db = json.load(f)

    user_audio_path = sys.argv[1] if len(sys.argv) > 1 else "Me surah Fatiha.wav"
    if not os.path.exists(user_audio_path):
        print(f"Cannot find test file: {user_audio_path}")
        return

    print(f"\nProcessing: {user_audio_path}")

    # ── Load & trim silence ───────────────────────────────────────────────
    audio, sr = librosa.load(user_audio_path, sr=16000, mono=True)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)

    # ── Decide whether to apply DeepFilterNet ────────────────────────────
    if is_clean_audio(audio_trimmed):
        print("🎙  Studio-quality audio detected — skipping DeepFilterNet")
        processed_audio = audio_trimmed
    else:
        print("🔧 Noisy audio detected — applying DeepFilterNet...")
        sf.write("temp_user.wav", audio_trimmed, 16000)
        df_audio, _ = load_audio("temp_user.wav", sr=df_state.sr())
        cleaned = enhance(df_model, df_state, df_audio)
        processed_audio = cleaned.squeeze().cpu().numpy()
        df_sr = df_state.sr()
        if df_sr != 16000:
            processed_audio = librosa.resample(processed_audio, orig_sr=df_sr, target_sr=16000)
        if os.path.exists("temp_user.wav"):
            os.remove("temp_user.wav")

    # ── ASR verification ─────────────────────────────────────────────────
    print("Transcribing to verify Fatiha content...")
    transcription = whisper_pipe(processed_audio)["text"]
    print(f"Transcription: {transcription}")

    if fuzzy_match(transcription, "بسم الله الرحمن الرحيم"):
        print("✅ Fatiha verified!")
    else:
        print("⚠️  Warning: Fatiha not clearly detected in transcription")

    # ── Extract WavLM embedding ───────────────────────────────────────────
    print("Extracting voice embedding...")
    inputs = processor(processed_audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = wavlm_model(**inputs)
        user_emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        user_emb = user_emb / np.linalg.norm(user_emb)

    # ── Compare against master DB ─────────────────────────────────────────
    print("Computing similarities...")
    results = []
    for reciter_id, data in master_db.items():
        master_emb = np.array(data["embedding"])
        similarity = float(np.dot(user_emb, master_emb))
        # Name is stored directly in the DB from phase1
        name = data.get("name", f"Reciter {reciter_id}")
        results.append({"id": reciter_id, "name": name, "score": similarity})

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]

    print(f"\n🏆 Best match: {best['name']}")
    print(f"   Score: {best['score']:.4f}")
    print("\n📊 Top-10 rankings:")
    for r in results[:10]:
        bar = "█" * int(r["score"] * 20)
        print(f"   {r['score']:.4f} {bar:20s}  {r['name']}")

if __name__ == "__main__":
    main()
