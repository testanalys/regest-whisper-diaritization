# whisper-diarization

Forked from [thomasmol/cog-whisper-diarization](https://github.com/thomasmol/cog-whisper-diarization) and adapted for production deployment on RunPod Serverless.

## What this is

A speaker-diarized transcription pipeline combining **faster-whisper large-v3-turbo** and **pyannote 3.1** for speaker detection. Given an audio file or URL it returns a timestamped transcript with per-word and per-segment speaker labels.

Achieves an approximate cost of: **~$0.27 per hour of audio** on RunPod RTX 4090 Serverless.

---

## Changes from upstream

The original repo was designed for deployment on Replicate via the Cog framework. This fork adapts it for direct RunPod Serverless deployment, which required resolving several compatibility issues and making meaningful performance improvements.

### Compatibility fixes

- **Cog → RunPod handler migration** — added `handler.py` to wrap the predictor in RunPod's serverless handler protocol, replacing Cog's built-in HTTP server which workers would start but never receive jobs through
- **HuggingFace authentication** — replaced deprecated `use_auth_token=` argument with `huggingface_hub.login()` at setup time
- **Dependency pinning** — pinned `pyannote-audio==3.3.1` and `huggingface_hub==0.23.4` to resolve a version conflict between pyannote's internals and newer huggingface_hub releases

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
- Accept pyannote model conditions on HuggingFace for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`
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
