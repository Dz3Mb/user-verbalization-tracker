"""Transcription service using openai-whisper running locally."""

import os
import glob
import shutil

import whisper


def _ensure_ffmpeg_on_path() -> None:
    """Make sure ffmpeg is discoverable by Whisper.

    Whisper shells out to ffmpeg to decode audio. On Windows, a fresh winget
    install adds ffmpeg to PATH only for *new* terminals, so a server started
    in an existing terminal fails with "[WinError 2] file not found".

    This helper locates ffmpeg (env override, PATH, or the winget install dir)
    and prepends its folder to PATH for the current process. No-op if ffmpeg
    is already available.
    """
    # 1. Explicit override wins.
    ffmpeg_dir = os.getenv("FFMPEG_DIR")
    if ffmpeg_dir and os.path.isfile(os.path.join(ffmpeg_dir, "ffmpeg.exe")):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        return

    # 2. Already on PATH? Nothing to do.
    if shutil.which("ffmpeg"):
        return

    # 3. Search common Windows install locations (winget, manual installs).
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    candidates = []
    if local_appdata:
        candidates.append(
            os.path.join(
                local_appdata,
                "Microsoft", "WinGet", "Packages",
                "Gyan.FFmpeg*", "**", "bin", "ffmpeg.exe",
            )
        )
    candidates.append(r"C:\ffmpeg\bin\ffmpeg.exe")
    candidates.append(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe")

    for pattern in candidates:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            os.environ["PATH"] = (
                os.path.dirname(matches[0]) + os.pathsep + os.environ.get("PATH", "")
            )
            return


_ensure_ffmpeg_on_path()

# Load model once at module level to avoid reloading per request.
# Model size is configurable via the WHISPER_MODEL env var.
#   tiny / base   -> fast, lower accuracy (demos)
#   small         -> good accuracy/speed tradeoff on CPU (default here)
#   medium / large-> best accuracy, slower and more RAM (GPU recommended)
_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
_model = whisper.load_model(_MODEL_NAME)

# Optional domain vocabulary hint to bias decoding (names, jargon, acronyms).
_INITIAL_PROMPT = os.getenv("WHISPER_INITIAL_PROMPT") or None


def transcribe_audio(file_path: str, language: str | None = None) -> dict:
    """Transcribe an audio file and return text with segment timestamps.

    `language` is an ISO code (e.g. "en", "fr"). When None or "auto", Whisper
    auto-detects the spoken language. The detected/used language is returned so
    the rest of the pipeline (NER, linking) can adapt to it.

    Decoding is tuned for accuracy on short clips:
    - beam search (beam_size/best_of) instead of greedy decoding;
    - temperature fallback so a failed/low-confidence pass is retried;
    - condition_on_previous_text=False to limit hallucination/looping on
      short recordings;
    - fp16=False because we run on CPU (also silences the FP16 warning).
    """
    # Normalize: treat "auto"/empty as auto-detect (let Whisper decide).
    lang = (language or "").strip().lower()
    options = {} if lang in ("", "auto") else {"language": lang}

    result = _model.transcribe(
        file_path,
        beam_size=5,
        best_of=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        condition_on_previous_text=False,
        initial_prompt=_INITIAL_PROMPT,
        fp16=False,
        **options,
    )

    segments = [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        }
        for seg in result.get("segments", [])
    ]

    return {
        "text": result["text"].strip(),
        "segments": segments,
        "language": result.get("language", lang or None),
        "model": _MODEL_NAME,
    }
