"""FastAPI backend for the verbalization tracker."""

import uuid
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.transcribe import transcribe_audio
from services.nlp import analyze_text
from services.storage import save_result

app = FastAPI(title="Verbalization Tracker API", version="0.1.0")

# CORS — allow the frontend served from any local origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Max upload size: 25 MB (reasonable for short voice recordings)
MAX_FILE_SIZE = 25 * 1024 * 1024


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Receive an audio file, transcribe, extract entities/relations, return JSON."""

    # Validate content type
    if file.content_type and not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file.")

    # Read and validate size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 25 MB.")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Save to disk so Whisper can read it
    session_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix if file.filename else ".webm"
    upload_path = UPLOAD_DIR / f"{session_id}{ext}"

    try:
        upload_path.write_bytes(contents)

        # 1. Transcribe
        transcription = transcribe_audio(str(upload_path))

        # 2. NLP analysis
        nlp_result = analyze_text(transcription["text"])

        # 3. Build response
        result = {
            "transcription": transcription,
            "entities": nlp_result["entities"],
            "relations": nlp_result["relations"],
        }

        # 4. Persist
        save_result(session_id, result)

        return {"session_id": session_id, **result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        # Clean up uploaded file after processing
        if upload_path.exists():
            os.remove(upload_path)
