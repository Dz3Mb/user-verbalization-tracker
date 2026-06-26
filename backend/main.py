"""FastAPI backend for the verbalization tracker."""

import uuid
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from services.transcribe import transcribe_audio
from services.nlp import analyze_text
from services.linking import enrich_entities
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
async def analyze(file: UploadFile = File(...), language: str = Form("auto")):
    """Receive an audio file, transcribe, extract entities/relations, return JSON.

    `language` is an ISO code ("en", "fr", ...) or "auto" to let Whisper detect
    the spoken language. The detected language drives NER and entity linking.
    """

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

        # 1. Transcribe (auto-detect or forced language)
        transcription = transcribe_audio(str(upload_path), language)

        # Language actually used downstream (detected by Whisper, or forced).
        used_lang = transcription.get("language") or (
            None if language in (None, "", "auto") else language
        )

        # 2. NLP analysis (entities + simple relations), language-aware
        nlp_result = analyze_text(transcription["text"], used_lang)

        # 3. Align entities with knowledge graphs (Wikidata / DBpedia)
        linked = enrich_entities(transcription["text"], nlp_result["entities"], used_lang)

        # 4. Build response
        result = {
            "transcription": transcription,
            "entities": linked["entities"],
            "relations": nlp_result["relations"],
            "linking": linked["meta"],
        }

        # 5. Persist
        save_result(session_id, result)

        return {"session_id": session_id, **result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        # Clean up uploaded file after processing
        if upload_path.exists():
            os.remove(upload_path)
