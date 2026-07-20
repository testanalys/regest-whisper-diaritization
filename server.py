"""HTTP API for whisper_flow on Azure Container Apps (replaces RunPod handler)."""
from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

API_KEY = (os.getenv("TRANSCRIPTION_API_KEY") or "").strip()

_predictor = None
_predictor_lock = threading.Lock()
# One GPU job per replica — avoids VRAM races when ACA scales concurrent HTTP requests.
_gpu_semaphore = threading.Semaphore(1)


def _get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor
    with _predictor_lock:
        if _predictor is None:
            from predict import Predictor

            p = Predictor()
            p.setup()
            _predictor = p
    return _predictor


app = FastAPI(title="whisper-flow", version="1.0.0")

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


class TranscribeInput(BaseModel):
    file_url: str | None = None
    file_string: str | None = None
    file: str | None = None
    num_speakers: int | None = Field(default=None, ge=1, le=50)
    translate: bool = False
    language: str | None = None
    prompt: str | None = None


class RunRequest(BaseModel):
    input: TranscribeInput


def _check_auth(authorization: str | None) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid bearer token")


def _run_predict(inp: TranscribeInput) -> dict[str, Any]:
    with _gpu_semaphore:
        predictor = _get_predictor()
        output = predictor.predict(
            file_string=inp.file_string,
            file_url=inp.file_url,
            file=inp.file,
            num_speakers=inp.num_speakers,
            translate=inp.translate,
            language=inp.language,
            prompt=inp.prompt,
        )
        return {
            "segments": output.segments,
            "language": output.language,
            "num_speakers": output.num_speakers,
            "speakers": output.speakers or {},
        }


def _worker(job_id: str, inp: TranscribeInput) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "IN_PROGRESS"
    try:
        result = _run_predict(inp)
        with _jobs_lock:
            _jobs[job_id]["status"] = "COMPLETED"
            _jobs[job_id]["output"] = result
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "FAILED"
            _jobs[job_id]["error"] = str(exc)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/transcribe")
def transcribe_sync(inp: TranscribeInput, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Synchronous transcription (use for smoke tests)."""
    _check_auth(authorization)
    if not inp.file_url and not inp.file_string and not inp.file:
        raise HTTPException(status_code=400, detail="Provide file_url, file_string, or file")
    return _run_predict(inp)


@app.post("/v2/{endpoint_id}/run")
def run_async(endpoint_id: str, body: RunRequest, authorization: str | None = Header(default=None)) -> dict[str, str]:
    """RunPod-compatible async submit (pipeline integration)."""
    _check_auth(authorization)
    inp = body.input
    if not inp.file_url and not inp.file_string and not inp.file:
        raise HTTPException(status_code=400, detail="Provide file_url, file_string, or file")
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "IN_QUEUE", "output": None, "error": None, "created": time.time()}
    threading.Thread(target=_worker, args=(job_id, inp), daemon=True).start()
    return {"id": job_id}


@app.get("/v2/{endpoint_id}/status/{job_id}")
def job_status(endpoint_id: str, job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """RunPod-compatible status poll."""
    _check_auth(authorization)
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    payload: dict[str, Any] = {"status": job["status"]}
    if job["status"] == "COMPLETED":
        payload["output"] = job["output"]
    elif job["status"] == "FAILED":
        payload["error"] = job["error"]
    return payload
