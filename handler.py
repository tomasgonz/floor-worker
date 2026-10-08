"""Who spoke when, on a GPU at RunPod: pyannote's speaker-diarization-community-1, for recordings too long to work out on
a CPU in good time (a UN meeting's floor, three hours of everyone in their own voice).

Asked {"url": a media link, "num_speakers" | "min_speakers" | "max_speakers": optional, "start", "seconds": optional}, it
reads the audio itself with ffmpeg and answers {"turns": [[start, end, label], ...], "embeddings": {label: [...]},
"duration", "gpu", "seconds": {"reading", "speakers"}}. The turns are exclusive - one speaker at any instant - as laying
them over a transcript's words wants. Asked {"check": true}, it loads the model on the GPU and says how long that took. The model is gated: HF_TOKEN
(set on the endpoint, never in the image) is a Hugging Face token whose owner accepted its terms."""

import os
import subprocess
import time

import numpy as np
import runpod
import torch

MODEL = "pyannote/speaker-diarization-community-1"
SAMPLE_RATE = 16000
_pipeline = None


def pipeline():
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline

        _pipeline = Pipeline.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN") or None)
        if _pipeline is None:
            raise RuntimeError("the speaker model could not be loaded: is HF_TOKEN set on the endpoint, and were the model's terms accepted?")
        _pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return _pipeline


def audio(url: str, start: float = 0.0, seconds: float = 0.0) -> np.ndarray:
    """16 kHz mono samples, decoded by ffmpeg as Boswell decodes them."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", *(["-ss", str(start)] if start else []), "-i", url,
           *(["-t", str(seconds)] if seconds else []), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    done = subprocess.run(cmd, capture_output=True, timeout=3600)
    if done.returncode:
        raise RuntimeError("the audio could not be read: " + done.stderr.decode(errors="replace")[-300:])
    return np.frombuffer(done.stdout, dtype=np.float32)


def count(asked: dict, name: str) -> int | None:
    return int(asked[name]) if asked.get(name) else None


def handler(job):
    asked = job.get("input") or {}
    started = time.time()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if asked.get("check"):
        pipeline()
        return {"ok": True, "gpu": gpu, "model": MODEL, "torch": torch.__version__, "loaded": round(time.time() - started, 1)}
    if not asked.get("url"):
        raise ValueError("no url: a link to the audio to listen to")
    runpod.serverless.progress_update(job, "reading the audio")
    sound = audio(asked["url"], float(asked.get("start") or 0), float(asked.get("seconds") or 0))
    if len(sound) < SAMPLE_RATE:
        raise ValueError("less than a second of audio")
    read = time.time()
    runpod.serverless.progress_update(job, "working out who spoke")
    output = pipeline()({"waveform": torch.from_numpy(sound.copy())[None], "sample_rate": SAMPLE_RATE},
                        num_speakers=count(asked, "num_speakers"), min_speakers=count(asked, "min_speakers"), max_speakers=count(asked, "max_speakers"))
    annotation = getattr(output, "exclusive_speaker_diarization", None) or output.speaker_diarization
    turns = [[round(float(s.start), 3), round(float(s.end), 3), label] for s, _, label in annotation.itertracks(yield_label=True)]
    embeddings = {}
    vectors = getattr(output, "speaker_embeddings", None)
    if vectors is not None:
        for label, vector in zip(output.speaker_diarization.labels(), vectors):
            if np.all(np.isfinite(vector)):
                embeddings[label] = [float(x) for x in vector]
    return {"turns": turns, "embeddings": embeddings, "duration": round(len(sound) / SAMPLE_RATE, 1), "gpu": gpu,
            "seconds": {"reading": round(read - started, 1), "speakers": round(time.time() - read, 1)}}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
