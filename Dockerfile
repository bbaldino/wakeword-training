FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    ffmpeg libspeexdsp-dev espeak-ng git wget \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Pin numpy<2 early — many dependencies (pyarrow, tensorflow, speechbrain) are incompatible with numpy 2.x
RUN pip install --no-cache-dir "numpy<2"

# Install PyTorch with CUDA 12.1 support first (before anything else pulls in CPU-only torch)
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone piper-sample-generator and download TTS model
RUN git clone https://github.com/dscripka/piper-sample-generator \
    && mkdir -p piper-sample-generator/models \
    && wget -q -O piper-sample-generator/models/en-us-libritts-high.pt \
       'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'

# Install piper TTS dependencies
RUN pip install --no-cache-dir piper-phonemize espeak-phonemizer webrtcvad

# Clone and install openwakeword (base only, not [full] to avoid dep conflicts)
RUN git clone https://github.com/dscripka/openwakeword \
    && pip install --no-cache-dir -e ./openwakeword

# Install training dependencies (order follows the reference notebook)
RUN pip install --no-cache-dir \
    mutagen==1.47.0 \
    torchinfo==1.8.0 \
    torchmetrics==1.2.0 \
    speechbrain==0.5.14 \
    audiomentations==0.33.0 \
    torch-audiomentations==0.11.0 \
    acoustics==0.2.6

# TFLite conversion dependencies
# Note: tensorflow-cpu 2.8.1 is old but required by onnx_tf 1.10.0.
# If conversion fails at runtime, the .onnx model is still usable.
RUN pip install --no-cache-dir \
    onnx==1.14.0 \
    tensorflow-cpu==2.8.1 \
    tensorflow_probability==0.16.0 \
    onnx_tf==1.10.0

# Remaining training dependencies
RUN pip install --no-cache-dir \
    "numpy<2" \
    pronouncing==0.2.0 \
    pyarrow==15.0.2 \
    datasets==2.21.0 \
    deep-phonemizer==0.0.19 \
    pyyaml

# Download embedding models used by openwakeword for feature extraction
RUN mkdir -p /app/openwakeword/openwakeword/resources/models \
    && wget -q -O /app/openwakeword/openwakeword/resources/models/embedding_model.onnx \
       https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx \
    && wget -q -O /app/openwakeword/openwakeword/resources/models/embedding_model.tflite \
       https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite \
    && wget -q -O /app/openwakeword/openwakeword/resources/models/melspectrogram.onnx \
       https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx \
    && wget -q -O /app/openwakeword/openwakeword/resources/models/melspectrogram.tflite \
       https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite

# Copy training scripts and config
COPY download_data.py train.sh config.template.yml /app/
RUN chmod +x /app/train.sh

# Install web UI dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy web application
COPY app/ /app/app/

VOLUME ["/data", "/output"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
