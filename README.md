# floor-worker

Who spoke when, on a GPU at RunPod: [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
as a RunPod serverless worker, for long recordings such as a UN meeting's floor.

The image, `ghcr.io/tomasgonz/floor-worker:latest`, is built by GitHub Actions on every change. It holds no keys and no
model: the model is fetched when a worker starts, with `HF_TOKEN` set on the endpoint (a Hugging Face token whose owner
accepted the model's terms).

**Input**

- `{"url": "...", "min_speakers": 10}` — a link ffmpeg can read (a file, or an HLS stream); optionally `num_speakers`,
  `max_speakers`, `start` and `seconds`. The worker downloads the audio itself.
- `{"check": true}` — loads the model on the GPU and says how long that took.

**Output**

`{"turns": [[start, end, label], ...], "embeddings": {label: [...]}, "duration", "gpu", "seconds": {"reading", "speakers"}}`.
The turns are exclusive: one speaker at any instant.
