# Who spoke when, on a RunPod GPU. Built by .github/workflows/image.yml into
# ghcr.io/tomasgonz/floor-worker. No keys and no model inside: the model is fetched at start with HF_TOKEN.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
# torch built for CUDA 12.6, and the pyannote the transcription server runs
RUN pip install --no-cache-dir "torch==2.14.1+cu126" torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir "pyannote.audio==4.0.7" "runpod>=1.7"
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
