"""CLI tool: run the full pipeline on a local audio file (podcast, mp3, wav...).

Usage (from the `backend` folder, with the venv active):

    python tools/transcribe_file.py path/to/audio.mp3
    python tools/transcribe_file.py path/to/audio.mp3 --save
    python tools/transcribe_file.py path/to/audio.mp3 --language fr --save
    python tools/transcribe_file.py path/to/audio.mp3 --quiet  # JSON only on stdout

Language defaults to "auto" (Whisper detects it). Use --language <code> to force
a language (e.g. en, fr).

Supports any format ffmpeg can decode (mp3, wav, m4a, webm, ogg, flac...).
Prints the structured JSON result and, with --save, stores it under results/.
"""

import sys
import json
import uuid
import time
from pathlib import Path

# Make `services` importable when running this file directly from tools/.
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from services.transcribe import transcribe_audio  # noqa: E402
from services.nlp import analyze_text  # noqa: E402
from services.linking import enrich_entities  # noqa: E402
from services.storage import save_result  # noqa: E402


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg, file=sys.stderr)


def process(audio_path: str, language: str = "auto", quiet: bool = False) -> dict:
    timings = {}

    t0 = time.time()
    transcription = transcribe_audio(audio_path, language)
    timings["transcribe_s"] = round(time.time() - t0, 2)
    used_lang = transcription.get("language") or (
        None if language in (None, "", "auto") else language
    )
    _log(
        quiet,
        f"  transcribed in {timings['transcribe_s']}s "
        f"(model={transcription.get('model')}, "
        f"lang={transcription.get('language')} "
        f"prob={transcription.get('language_probability')})",
    )

    t0 = time.time()
    nlp_result = analyze_text(transcription["text"], used_lang)
    timings["nlp_s"] = round(time.time() - t0, 2)
    _log(
        quiet,
        f"  NLP in {timings['nlp_s']}s "
        f"({len(nlp_result['entities'])} entities, "
        f"{len(nlp_result['relations'])} relations)",
    )

    t0 = time.time()
    linked = enrich_entities(transcription["text"], nlp_result["entities"], used_lang)
    timings["linking_s"] = round(time.time() - t0, 2)
    n_linked = sum(1 for e in linked["entities"] if e.get("wikidata"))
    _log(
        quiet,
        f"  linking in {timings['linking_s']}s "
        f"({n_linked}/{len(nlp_result['entities'])} entities aligned with Wikidata)",
    )

    return {
        "source_file": audio_path,
        "transcription": transcription,
        "entities": linked["entities"],
        "relations": nlp_result["relations"],
        "linking": linked["meta"],
        "timings": timings,
    }


def main(argv: list[str]) -> int:
    # Parse flags: --language <code>, --save, --quiet
    language = "auto"
    save = False
    quiet = False
    cleaned: list[str] = []
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
        elif a == "--save":
            save = True
        elif a == "--quiet":
            quiet = True
        elif a.startswith("--"):
            continue
        else:
            cleaned.append(a)

    if not cleaned:
        print(__doc__)
        return 1

    audio_path = cleaned[0]
    if not Path(audio_path).exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        return 1

    _log(quiet, f"Processing: {audio_path}  (language={language})")
    result = process(audio_path, language, quiet)

    if save:
        session_id = str(uuid.uuid4())
        path = save_result(session_id, result)
        _log(quiet, f"  saved to: {path}")

    # JSON to stdout so `--quiet` keeps the output pipe-friendly.
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
