# Documentation: User Verbalization Tracker

## Overview
The app records audio in the browser, uploads it to the FastAPI backend, transcribes it with Whisper (local), extracts entities and relations with spaCy, and saves a JSON file per session.

## Entity Recognition (spaCy)
The backend uses spaCy's small English model `en_core_web_sm`.

- The model provides a named-entity recognizer (NER) that labels spans such as PERSON, ORG, GPE, DATE, etc.
- The code loads the model once and runs `doc = nlp(text)` on the transcription text.
- Entities are collected from `doc.ents` with their text, label, and character offsets.

Where it lives:
- [backend/services/nlp.py](backend/services/nlp.py)
- Function: `extract_entities()`

## Relation Extraction (rule-based)
Relations are rule-based and built from spaCy dependency parsing. The logic is intentionally simple:

- For each verb token, find:
  - subjects: dependency labels `nsubj` or `nsubjpass`
  - objects: dependency labels `dobj`, `attr`, or `pobj`
- Also captures prepositional objects attached to the verb.
- Each relation is stored as a subject -> predicate -> object triple plus the original sentence.

This is not an ML model; it is pattern-based and best for short, simple sentences.

Where it lives:
- [backend/services/nlp.py](backend/services/nlp.py)
- Function: `extract_relations()`

## Transcription (Whisper)
The backend uses `openai-whisper` locally and loads the `base` model by default.

- The audio file saved to `backend/uploads/` is passed to Whisper.
- Whisper returns a full transcription plus segment timestamps.
- Language is auto-detected unless you force it.

Where it lives:
- [backend/services/transcribe.py](backend/services/transcribe.py)
- Function: `transcribe_audio()`

To force a language, set `language="fr"` or `language="en"` in the `transcribe()` call.

## JSON Output Structure
The backend returns JSON to the frontend and also saves it to disk.

Top-level fields:
- `session_id`: unique ID for the recording session.
- `timestamp`: UTC timestamp when the result was saved.
- `transcription`:
  - `language`: detected language code (e.g., "en", "fr").
  - `text`: full transcription string.
  - `segments`: list of {start, end, text} for each segment.
- `entities`: list of named entities with text, label, and char offsets.
- `relations`: list of subject-predicate-object triples with the source sentence.

Example keys:
- `transcription.text`
- `entities[0].label`
- `relations[0].predicate`

The JSON is stored in `backend/results/<session_id>.json`.

## What Remains To Be Done (Documentation)
This MVP is functional but still needs these items documented or implemented:

1) Production hardening
- Restrict CORS to trusted origins.
- Add size/time limits for uploads and processing.
- Add structured logging and request IDs.

2) UX and workflow
- Add a language selector in the UI (auto/en/fr).
- Display detected language in the UI.
- Show upload/transcription progress.

3) NLP improvements
- Add better relation patterns ("works at", "lives in", "located in") or a real relation model.
- Add confidence scores and entity normalization.
- Add multilingual spaCy models when targeting non-English text.

4) Storage and session management
- Add a history list in the UI.
- Add export options (CSV, JSONL).
- Optional: database for multi-user studies.

5) Testing and benchmarks
- Unit tests for NLP rules and API responses.
- Small audio fixtures for regression testing.
- Add a performance note (model load time, transcription latency).
