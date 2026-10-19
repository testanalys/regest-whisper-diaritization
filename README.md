# whisper-diarization

Forked from [thomasmol/cog-whisper-diarization](https://github.com/thomasmol/cog-whisper-diarization) and adapted for production deployment on RunPod Serverless.

## What this is

A speaker-diarized transcription pipeline combining **faster-whisper large-v3-turbo** and **pyannote `speaker-diarization-community-1`** (pyannote.audio 4) for speaker detection. Given an audio file or URL it returns a timestamped transcript with per-word and per-segment speaker labels.

Each word takes its speaker from community-1's *exclusive* diarization (one speaker at a time), and a Whisper segment that spans a change of speaker is split there. Each speaker also gets one 256-d voiceprint, stamped `model_version: pyannote-speaker-diarization-community-1`. Voiceprints from the earlier 3.1 model are in a different embedding space and must not be compared with these.

Achieves an approximate cost of: **~$0.27 per hour of audio** on RunPod RTX 4090 Serverless.

---

## Changes from upstream

The original repo was designed for deployment on Replicate via the Cog framework. This fork adapts it for direct RunPod Serverless deployment, which required resolving several compatibility issues and making meaningful performance improvements.

### Compatibility fixes

- **Cog → RunPod handler migration** — added `handler.py` to wrap the predictor in RunPod's serverless handler protocol, replacing Cog's built-in HTTP server which workers would start but never receive jobs through
- **HuggingFace authentication** — replaced deprecated `use_auth_token=` argument with `huggingface_hub.login()` at setup time
- **Dependency pinning** — `requirements.txt` pins torch 2.8.0 (cog base image: CUDA 12.9 with cuDNN 9), pyannote.audio 4.0.7, faster-whisper 1.2.1 and ctranslate2 4.6.3. Audio is passed to pyannote preloaded, so torchcodec's FFmpeg decoding is never needed

---

### Performance improvements

- **Batched inference** — replaced `WhisperModel` with `BatchedInferencePipeline`, processing multiple VAD-segmented audio chunks simultaneously on the GPU rather than sequentially
- **Compute type** — switched to `int8_float16`, using INT8 quantization on linear layers for faster throughput with minimal accuracy loss
- **TF32** — enabled TF32 matrix operations at setup for improved throughput on Ampere/Ada GPUs
- **Beam size** — reduced from 5 to 1 (greedy decoding) for faster decoding on conversational audio

---

## How to deploy

**1. Fork and configure**
- Fork this repo
- Accept pyannote model conditions on HuggingFace for `pyannote/speaker-diarization-community-1` with the account whose token is in `HUGGING_FACE_HUB_TOKEN`, or the worker fails at startup
- RunPod workers need an NVIDIA driver that supports CUDA 12.9 (restrict the endpoint's allowed CUDA versions)
- Add GitHub Actions secrets: `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`

**2. Build**
- Push any commit to `main` — GitHub Actions builds the image via Cog and pushes to Docker Hub automatically (no local Docker required)

**3. Deploy on RunPod**
- Create a new Serverless endpoint pointing to your Docker Hub image
- Set environment variable: `HUGGING_FACE_HUB_TOKEN`
- Set start command: `python -u handler.py`
- Enable Flash Boot for faster cold starts

---

## Calling the endpoint

Send a POST request to your RunPod endpoint with the following input schema:

- `file_url: str` — a direct audio file URL (recommended, avoids payload size limits)
- `file_string: str` — a Base64 encoded audio file
- `file: Path` — a local audio file path
- `num_speakers: int` — number of speakers (1–50), leave empty to autodetect
- `translate: bool` — translate speech into English
- `language: str` — language code e.g. `en`, leave empty to autodetect
- `prompt: str` — vocabulary hotwords: names, acronyms, loanwords; use punctuation for best accuracy

Example payload:
```json
{
  "input": {
    "file_url": "https://your-audio-url.com/episode.mp3",
    "num_speakers": 2,
    "language": "en",
    "prompt": "Anthropic, Claude, LLM"
  }
}
```
