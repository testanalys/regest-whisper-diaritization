import base64
import subprocess
import os
import requests
import time
import wave
import torch
import re
import numpy as np

from cog import BasePredictor, BaseModel, Input, Path
from faster_whisper import WhisperModel, BatchedInferencePipeline
from pyannote.audio import Pipeline
from huggingface_hub import login

DIARIZATION_CHECKPOINT = "pyannote/speaker-diarization-community-1"
# Stamped on every centroid. Voiceprints from different models live in
# different embedding spaces and must never be compared; downstream
# partitions on this string.
VOICEPRINT_MODEL = "pyannote-speaker-diarization-community-1"


class Output(BaseModel):
    segments: list
    language: str = None
    num_speakers: int = None
    # Per-speaker voiceprint centroids for the cross-file identity layer:
    #   {SPEAKER_xx: {embedding: [float], dim, model_version, duration_s, segment_count}}
    # One embedding per speaker per file (community-1 speaker_embeddings).
    # Empty dict when diarization produced no usable embeddings.
    speakers: dict = None

class Predictor(BasePredictor):
    def setup(self):
        device = (os.getenv("WHISPER_DEVICE") or "cuda").strip().lower()
        if device not in ("cuda", "cpu"):
            raise ValueError(f"WHISPER_DEVICE must be cuda or cpu, got {device!r}")

        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            compute_type = "int8_float16"
        else:
            compute_type = "int8"

        model_name = (os.getenv("WHISPER_MODEL") or "large-v3-turbo").strip()
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        self.model = BatchedInferencePipeline(model=model)
        self._device = device
        self._skip_diarization = (os.getenv("SKIP_DIARIZATION") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            login(token=hf_token)

        self.diarization_model = None
        if not self._skip_diarization:
            # Gated on Hugging Face: the token's account must have accepted
            # the community-1 conditions, or this raises at startup.
            self.diarization_model = Pipeline.from_pretrained(
                DIARIZATION_CHECKPOINT, token=hf_token or None
            )
            if self.diarization_model is None:
                raise RuntimeError(f"could not load {DIARIZATION_CHECKPOINT}")
            self.diarization_model.to(torch.device(device))

    def predict(
        self,
        file_string: str = Input(description="Base64 encoded audio", default=None),
        file_url: str = Input(description="Direct audio URL", default=None),
        file: Path = Input(description="Audio file", default=None),
        num_speakers: int = Input(description="Number of speakers (1-50)", ge=1, le=50, default=None),
        translate: bool = Input(description="Translate to English", default=False),
        language: str = Input(description="Language code (e.g. 'en')", default=None),
        prompt: str = Input(description="Vocabulary/hotwords", default=None),
    ) -> Output:
        temp_wav_filename = f"temp-{time.time_ns()}.wav"
        temp_audio_filename = None

        try:
            # 1. AUDIO DOWNLOAD/PREP
            if file:
                input_path = str(file)
            elif file_url:
                response = requests.get(file_url)
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, "wb") as f:
                    f.write(response.content)
                input_path = temp_audio_filename
            elif file_string:
                audio_data = base64.b64decode(file_string.split(",")[1] if "," in file_string else file_string)
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, "wb") as f:
                    f.write(audio_data)
                input_path = temp_audio_filename
            else:
                raise ValueError("No audio input provided.")

            # FFmpeg conversion to 16k mono wav (required for Whisper/Pyannote)
            subprocess.run([
                "ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1",
                "-c:a", "pcm_s16le", temp_wav_filename, "-y"
            ], check=True, capture_output=True)

            # 2. INFERENCE
            segments, detected_speakers, detected_lang, speaker_voiceprints = self.speech_to_text(
                temp_wav_filename, num_speakers, prompt, language, translate
            )

            return Output(
                segments=segments,
                language=detected_lang,
                num_speakers=detected_speakers,
                speakers=speaker_voiceprints,
            )

        finally:
            for f in [temp_wav_filename, temp_audio_filename]:
                if f and os.path.exists(f):
                    os.remove(f)

    def speech_to_text(self, audio_path, num_speakers, prompt, language, translate):
        # TRANSCRIPTION
        # beam_size=1 is the secret sauce for the Turbo model's speed
        options = dict(
            language=language,
            beam_size=1,
            batch_size=16,
            vad_filter=True,
            word_timestamps=True,
            initial_prompt=prompt,
            task="translate" if translate else "transcribe",
        )

        t_start = time.time()
        whisper_segments, info = self.model.transcribe(audio_path, **options)
        whisper_segments = list(whisper_segments)
        t_transcribe = time.time() - t_start

        t_diarize = 0.0
        diarization = None
        if self.diarization_model is not None:
            # DIARIZATION (+ per-speaker voiceprint embeddings for cross-file identity).
            # Audio goes in preloaded, so pyannote never needs torchcodec to decode.
            diarization = self.diarization_model(
                {"waveform": _load_wav(audio_path), "sample_rate": 16000},
                num_speakers=num_speakers,
            )
            t_diarize = time.time() - (t_start + t_transcribe)

        # 3. WORD-LEVEL SPEAKER ASSIGNMENT
        # The exclusive diarization has one speaker at any instant, so each word
        # takes the speaker it overlaps most. A Whisper segment that spans a
        # change of speaker is split at the change: an interjection ("Mr. Fontes,
        # your response") stays with the voice that said it.
        diar_turns = []
        if diarization is not None:
            for turn, _, speaker in diarization.exclusive_speaker_diarization.itertracks(yield_label=True):
                diar_turns.append((turn.start, turn.end, speaker))
            diar_turns.sort()

        unique_speakers = len({t[2] for t in diar_turns}) if diar_turns else 1
        final_segments = split_by_speaker(whisper_segments, diar_turns)

        # 4. SMART GROUPING
        if not final_segments:
            return [], 0, info.language, {}

        grouped = []
        current = final_segments[0]
        end_punctuation = r"[.!?]$"

        for i in range(1, len(final_segments)):
            next_seg = final_segments[i]
            gap = next_seg["start"] - current["end"]
            duration = current["end"] - current["start"]

            # Merge if same speaker, small gap, and not a full sentence yet
            if (next_seg["speaker"] == current["speaker"] and
                gap <= 1.0 and
                duration < 25.0 and
                not re.search(end_punctuation, current["text"])):

                current["end"] = next_seg["end"]
                current["text"] += " " + next_seg["text"]
                current["words"].extend(next_seg["words"])
            else:
                grouped.append(current)
                current = next_seg

        grouped.append(current)

        # Cleanup final text strings
        for g in grouped:
            g["text"] = re.sub(r"\s+", " ", g["text"]).strip()
            g["duration"] = g["end"] - g["start"]

        # 5. VOICEPRINT CENTROIDS (one embedding per speaker per file)
        speaker_voiceprints = (
            build_speaker_voiceprints(diarization, final_segments)
            if diarization is not None
            else {}
        )

        print(f"Transcribe: {t_transcribe:.2f}s | Diarize: {t_diarize:.2f}s | Total: {time.time()-t_start:.2f}s")
        return grouped, unique_speakers, info.language, speaker_voiceprints


def _load_wav(path):
    """The 16 kHz mono PCM file ffmpeg wrote, as a (1, time) float tensor."""
    with wave.open(path, "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0)


def speaker_at(turns, start, end, cursor=0):
    """The speaker of [start, end] in sorted, non-overlapping turns.

    Most overlap wins. A zero-length word, or one that falls in a gap between
    turns, takes the turn containing or nearest to its midpoint. Returns
    (speaker, cursor) so a caller walking forward in time keeps the search
    linear.
    """
    while cursor < len(turns) and turns[cursor][1] < start:
        cursor += 1
    best, best_overlap = None, 0.0
    i = cursor
    while i < len(turns) and turns[i][0] < end:
        overlap = min(end, turns[i][1]) - max(start, turns[i][0])
        if overlap > best_overlap:
            best, best_overlap = turns[i][2], overlap
        i += 1
    if best is None:
        mid = (start + end) / 2
        near = turns[max(cursor - 1, 0):cursor + 1]
        if near:
            best = min(near, key=lambda t: max(t[0] - mid, mid - t[1], 0.0))[2]
    return best, cursor


def split_by_speaker(whisper_segments, turns):
    """Whisper segments re-cut so every output segment has one speaker.

    Each word takes its own speaker from the diarization turns; consecutive
    words of one speaker form a segment. Without diarization everything is
    SPEAKER_00.
    """
    out = []
    cursor = 0
    for s in whisper_segments:
        words = list(s.words or [])
        if not words:
            speaker = speaker_at(turns, float(s.start), float(s.end), cursor)[0] if turns else None
            out.append({
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
                "speaker": speaker or "SPEAKER_00",
                "avg_logprob": float(s.avg_logprob),
                "words": [],
            })
            continue

        run = []
        for w in words:
            speaker = "SPEAKER_00"
            if turns:
                speaker, cursor = speaker_at(turns, float(w.start), float(w.end), cursor)
            if run and run[-1]["speaker"] != speaker:
                out.append(_segment_from_words(run, s))
                run = []
            run.append({
                "start": float(w.start),
                "end": float(w.end),
                "word": w.word,
                "probability": float(w.probability),
                "speaker": speaker,
            })
        out.append(_segment_from_words(run, s))
    return out


def _segment_from_words(words, whisper_segment):
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": "".join(w["word"] for w in words).strip(),
        "speaker": words[0]["speaker"],
        "avg_logprob": float(whisper_segment.avg_logprob),
        "words": words,
    }


def build_speaker_voiceprints(diarization, final_segments):
    """Map community-1's per-speaker embeddings to a serializable dict.

    ``speaker_embeddings`` is a (n_speakers, dim) array whose rows align with
    ``speaker_diarization.labels()``. Speakers with too little speech can have
    NaN embeddings; those are skipped so we never store a poisoned centroid.
    ``duration_s``/``segment_count`` come from the word-aligned segments and
    act as a quality signal downstream.
    """
    voiceprints = {}
    embeddings = diarization.speaker_embeddings
    if embeddings is None:
        return voiceprints

    labels = list(diarization.speaker_diarization.labels())
    for idx, label in enumerate(labels):
        if idx >= len(embeddings):
            continue
        emb = np.asarray(embeddings[idx], dtype=float)
        if emb.size == 0 or not np.all(np.isfinite(emb)):
            # Too little speech to embed reliably -> degrade to text-only.
            continue
        voiceprints[label] = {
            "embedding": [float(x) for x in emb.tolist()],
            "dim": int(emb.shape[-1]),
            "model_version": VOICEPRINT_MODEL,
            "duration_s": 0.0,
            "segment_count": 0,
        }

    for seg in final_segments:
        vp = voiceprints.get(seg.get("speaker"))
        if vp is not None:
            vp["duration_s"] += float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
            vp["segment_count"] += 1

    return voiceprints
