import os
import json
import torch
import numpy as np
import librosa
from df.enhance import enhance, init_df, load_audio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
import datasets
import soundfile as sf

def main():
    print("Loading pre-trained models for RunPod streaming...")
    df_model, df_state, _ = init_df()
    
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    wavlm_model = wavlm_model.to(device)
    wavlm_model.eval()
    
    # 1. Stream the massive Tadabur dataset on-the-fly (NO DOWNLOAD)
    print("Connecting to Hugging Face datasets stream...")
    dataset = datasets.load_dataset("FaisaI/tadabur", split="train", streaming=True)
    
    # 2. Build the master_db dynamically
    master_db = {}
    
    print("Streaming and processing master reciters...")
    count = 0
    # Process up to 100 reciters
    MAX_RECITERS = 100
    
    for sample in dataset:
        surah_id = sample.get("surah_id", sample.get("surah_number"))
        if surah_id == 1:
            reciter_name = str(sample.get("reciter_id"))
            if reciter_name not in master_db:
                print(f"Processing streamed Fatiha for reciter: {reciter_name}")
                
                audio_array = sample["audio"]["array"]
                sr = sample["audio"]["sampling_rate"]
                
                # Trim silence
                audio_trimmed, _ = librosa.effects.trim(audio_array, top_db=30)
                
                # We save to a temp file to feed DeepFilterNet
                temp_file = "temp_runpod_stream.wav"
                sf.write(temp_file, audio_trimmed, sr)
                
                df_audio, df_sr = load_audio(temp_file, sr=df_state.sr())
                cleaned_audio = enhance(df_model, df_state, df_audio)
                
                cleaned_audio_np = cleaned_audio.squeeze().cpu().numpy()
                
                if df_sr != 16000:
                    cleaned_audio_np = librosa.resample(cleaned_audio_np, orig_sr=df_sr, target_sr=16000)
                
                inputs = processor(cleaned_audio_np, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = wavlm_model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                    embedding = embedding / np.linalg.norm(embedding)
                    
                master_db[reciter_name] = {
                    "embedding": embedding.tolist(),
                    "name": reciter_name
                }
                
                count += 1
                if count >= MAX_RECITERS:
                    break

    # Save to local SSD
    with open("runpod_master_db.json", "w") as f:
        json.dump(master_db, f)
        
    if os.path.exists("temp_runpod_stream.wav"):
        os.remove("temp_runpod_stream.wav")
        
    print(f"✅ Successfully extracted embeddings for {count} reciters!")
    print("Saved to runpod_master_db.json")

if __name__ == "__main__":
    main()
