"""CLI tool: run the full pipeline on a local audio file (podcast, mp3, wav...).

Usage (from the `backend` folder, with the venv active):

    python tools/transcribe_file.py path/to/audio.mp3
    python tools/transcribe_file.py path/to/audio.mp3 --save
    python tools/transcribe_file.py path/to/audio.mp3 --language fr --save

Language defaults to "auto" (Whisper detects it). Use --language <code> to force
a language (e.g. en, fr, es, de, it).

Supports any format ffmpeg can decode (mp3, wav, m4a, webm, ogg, flac...).
Prints the structured JSON result and, with --save, stores it under results/.
"""

import sys
import json
import uuid
from pathlib import Path

# Make `services` importable when running this file directly from tools/.
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from services.transcribe import transcribe_audio  # noqa: E402
from services.nlp import analyze_text  # noqa: E402
from services.linking import enrich_entities  # noqa: E402
from services.storage import save_result  # noqa: E402


def process(audio_path: str, language: str = "auto") -> dict:
    transcription = transcribe_audio(audio_path, language)
    used_lang = transcription.get("language") or (
        None if language in (None, "", "auto") else language
    )
    nlp_result = analyze_text(transcription["text"], used_lang)
    linked = enrich_entities(transcription["text"], nlp_result["entities"], used_lang)
    return {
        "source_file": audio_path,
        "transcription": transcription,
        "entities": linked["entities"],
        "relations": nlp_result["relations"],
        "linking": linked["meta"],
    }


def main(argv: list[str]) -> int:
    # Optional: --language <code> (or --language=<code>); default "auto".
    language = "auto"
    cleaned = []
    skip = False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a.startswith("--language="):
            language = a.split("=", 1)[1]
        elif a == "--language":
            language = argv[i + 1] if i + 1 < len(argv) else "auto"
            skip = True
        elif a.startswith("--"):
            continue
        else:
            cleaned.append(a)

    save = "--save" in argv

    if not cleaned:
        print(__doc__)
        return 1

    audio_path = cleaned[0]
    if not Path(audio_path).exists():
        print(f"Error: file not found: {audio_path}")
        return 1

    print(f"Processing: {audio_path}  (language={language})")
    result = process(audio_path, language)

    if save:
        session_id = str(uuid.uuid4())
        path = save_result(session_id, result)
        print(f"Saved result to: {path}")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
