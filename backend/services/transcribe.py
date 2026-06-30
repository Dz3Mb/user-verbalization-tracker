"""Transcription service using faster-whisper (CTranslate2 backend).

faster-whisper is a drop-in alternative to openai-whisper that uses the
CTranslate2 inference engine. Same model weights, ~4x faster on CPU, lower
memory footprint, which makes the large-v3 model actually usable on CPU.

Public API:
    transcribe_audio(file_path, language=None) -> {text, segments, language, model}
"""

from __future__ import annotations

import os
import glob
import shutil
import logging

logger = logging.getLogger("transcribe")


def _ensure_ffmpeg_on_path() -> None:
    """Make sure ffmpeg is discoverable by faster-whisper / its dependencies.

    On Windows, a fresh winget install adds ffmpeg to PATH only for *new*
    terminals, so a server started in an existing terminal fails with
    "[WinError 2] file not found". This helper locates ffmpeg (env override,
    PATH, or the winget install dir) and prepends its folder to PATH for the
    current process. No-op if ffmpeg is already available.
    """
    ffmpeg_dir = os.getenv("FFMPEG_DIR")
    if ffmpeg_dir and os.path.isfile(os.path.join(ffmpeg_dir, "ffmpeg.exe")):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        return
    if shutil.which("ffmpeg"):
        return

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

# Use the OS trust store so model downloads (huggingface_hub) work behind
# corporate proxies that intercept HTTPS with a custom root CA. Must be
# injected before importing faster-whisper.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from faster_whisper import WhisperModel  # noqa: E402

# -- Configuration -----------------------------------------------------------
#
# WHISPER_MODEL: model size. "large-v3" is the highest-quality model OpenAI
# released; "large-v3-turbo" (alias "turbo") is ~6x faster and almost as good.
# On CPU, "large-v3-turbo" is the sweet spot for accuracy. Use "small" or
# "medium" to dial back if RAM/time matters.
_MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")

# Compute type: int8 is fastest on CPU with minor quality loss; float16 on GPU.
_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
_NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "1"))
_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "0"))  # 0 = auto

# Optional vocabulary hint that biases decoding toward specific names/jargon.
_INITIAL_PROMPT = os.getenv("WHISPER_INITIAL_PROMPT") or None

# Voice Activity Detection — uses Silero VAD via faster-whisper. Skips
# silences and dramatically reduces hallucinations on quiet or short clips.
_VAD = os.getenv("WHISPER_VAD", "true").strip().lower() in ("1", "true", "yes", "on")

# Word-level timestamps (useful downstream for highlighting / alignment).
_WORD_TIMESTAMPS = os.getenv("WHISPER_WORD_TIMESTAMPS", "true").strip().lower() in (
    "1", "true", "yes", "on",
)

logger.info(
    "Loading faster-whisper model=%r device=%r compute_type=%r",
    _MODEL_NAME, _DEVICE, _COMPUTE_TYPE,
)
_model = WhisperModel(
    _MODEL_NAME,
    device=_DEVICE,
    compute_type=_COMPUTE_TYPE,
    num_workers=_NUM_WORKERS,
    cpu_threads=_CPU_THREADS,
)


def transcribe_audio(file_path: str, language: str | None = None) -> dict:
    """Transcribe an audio file with faster-whisper.

    `language` is an ISO code (e.g. "en", "fr"). When None or "auto", Whisper
    auto-detects the spoken language.

    Returns:
        text:     full transcript (str)
        segments: list of {start, end, text, words?}
        language: detected/forced ISO code
        model:    model name actually used
    """
    lang = (language or "").strip().lower()
    forced_lang = None if lang in ("", "auto") else lang

    # Beam search + temperature fallback + word timestamps. faster-whisper
    # supports the same decoding options as openai-whisper.
    segments_iter, info = _model.transcribe(
        file_path,
        language=forced_lang,
        beam_size=5,
        best_of=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        condition_on_previous_text=False,
        initial_prompt=_INITIAL_PROMPT,
        word_timestamps=_WORD_TIMESTAMPS,
        vad_filter=_VAD,
        vad_parameters={"min_silence_duration_ms": 500} if _VAD else None,
    )

    segments = []
    full_text_parts = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        seg_out = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
        }
        if _WORD_TIMESTAMPS and getattr(seg, "words", None):
            seg_out["words"] = [
                {
                    "word": w.word.strip(),
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                    "probability": round(float(w.probability), 4),
                }
                for w in seg.words
            ]
        segments.append(seg_out)
        full_text_parts.append(text)

    return {
        "text": " ".join(full_text_parts).strip(),
        "segments": segments,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 4),
        "model": _MODEL_NAME,
        "vad": _VAD,
    }
