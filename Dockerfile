# ── Base: CUDA 12.1 + Python 3.10 (official RunPod starter) ──────────────────
FROM runpod/base:0.4.0-cuda11.8.0

# System deps for librosa / soundfile / mpg123
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libmpg123-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements_handler.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements_handler.txt

# Patch DeepFilterNet for torchaudio >= 2.1.0 compatibility (AudioMetaData was removed, it's just a type hint)
RUN sed -i 's/from torchaudio.backend.common import AudioMetaData/AudioMetaData = object/g' /usr/local/lib/python3.11/dist-packages/df/io.py


# ── Pre-download HuggingFace models into the image layer ─────────────────────
# This avoids cold-start model downloads; R2 DB is still fetched at runtime.
RUN python3.11 -c "\
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, pipeline; \
Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large'); \
WavLMModel.from_pretrained('microsoft/wavlm-large', use_safetensors=True); \
pipeline('automatic-speech-recognition', model='FaisaI/tadabur-Whisper-Small'); \
print('Models cached.')"

# DeepFilterNet downloads its own weights on first init — trigger it here too
RUN python3.11 -c "from df.enhance import init_df; init_df(); print('DeepFilterNet cached.')"

# ── Copy handler and bootstrap loader ───────────────────────────────────────
COPY handler.py .
COPY bootstrap.py .

# ── RunPod entrypoint ─────────────────────────────────────────────────────────
CMD ["python3.11", "-u", "bootstrap.py"]

LABEL org.opencontainers.image.source=https://github.com/HkKhan/tarteel
