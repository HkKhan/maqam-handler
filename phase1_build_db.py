import os
import json
import torch
import numpy as np
import librosa
from df.enhance import enhance, init_df, load_audio, save_audio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
import datasets
import soundfile as sf
from tqdm import tqdm

import shutil

def download_samples():
    print("Copying local sample files from tadabur repository...", flush=True)
    os.makedirs("local_data", exist_ok=True)
    src_dir = "tadabur/samples/audio"
    for file in os.listdir(src_dir):
        if file.endswith(".wav"):
            shutil.copy(os.path.join(src_dir, file), os.path.join("local_data", file))
    print("Copied local files successfully!", flush=True)

def main():
    if not os.path.exists("local_data") or len(os.listdir("local_data")) == 0:
        download_samples()
    else:
        print("local_data already exists and is not empty. Skipping download.", flush=True)
    
    print("Loading pre-trained DeepFilterNet...", flush=True)
    df_model, df_state, _ = init_df()
    
    print("Loading Wav2Vec2 Feature Extractor...", flush=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    
    print("Loading WavLM Model (microsoft/wavlm-large, ~1.2 GB)...", flush=True)
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    wavlm_model = wavlm_model.to(device)
    wavlm_model.eval()
    
    master_db = {}
    
    files = [f for f in os.listdir("local_data") if f.endswith(".wav")]
    print(f"Processing {len(files)} local files to extract embeddings...", flush=True)
    
    for file in tqdm(files, desc="Extracting embeddings"):
        # Expect file name format: reciter_<id>_fatiha.wav
        parts = file.split("_")
        if len(parts) >= 2:
            rid = parts[1]
        else:
            rid = file
            
        audio_path = os.path.join("local_data", file)
        
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
        
        sf.write("temp_trimmed.wav", audio_trimmed, 16000)
        
        df_audio, _ = load_audio("temp_trimmed.wav", sr=df_state.sr())
        cleaned_audio = enhance(df_model, df_state, df_audio)
        
        cleaned_audio_np = cleaned_audio.squeeze().cpu().numpy()
        
        df_sr = df_state.sr()
        if df_sr != 16000:
            cleaned_audio_np = librosa.resample(cleaned_audio_np, orig_sr=df_sr, target_sr=16000)
        
        inputs = processor(cleaned_audio_np, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = wavlm_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
        master_db[str(rid)] = {
            "embedding": embedding.tolist(),
            "name": str(rid)
        }
        
    with open("master_fatiha_db.json", "w") as f:
        json.dump(master_db, f, indent=2)
        
    if os.path.exists("temp_trimmed.wav"):
        os.remove("temp_trimmed.wav")
        
    print("Master database built locally!", flush=True)

if __name__ == "__main__":
    main()
