"""Transcription service using openai-whisper running locally."""

import shutil
import whisper

# Load model once at module level to avoid reloading per request.
# "base" is a good balance of speed and accuracy for demos.
# Options: tiny, base, small, medium, large
_model = whisper.load_model("base")


def transcribe_audio(file_path: str) -> dict:
    """Transcribe an audio file and return text with segment timestamps."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found on PATH. Install FFmpeg and restart the server."
        )
    # For a fixed language, pass language="fr" or language="en" below.
    result = _model.transcribe(file_path, task="transcribe")

    segments = [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        }
        for seg in result.get("segments", [])
    ]

    return {
        "language": result.get("language"),
        "text": result["text"].strip(),
        "segments": segments,
    }
