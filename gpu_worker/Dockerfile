# Base image with CUDA 12.1 for L4 compatibility
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Pre-download model weights during build so cold start is faster
# RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"
# RUN python3 -c "from transformers import WavLMModel, Wav2Vec2FeatureExtractor; WavLMModel.from_pretrained('microsoft/wavlm-large'); Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')"
# RUN python3 -c "from df.enhance import init_df; init_df()"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "600"]
