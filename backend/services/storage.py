"""Storage service: persist analysis results as JSON files."""

import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_result(session_id: str, data: dict) -> Path:
    """Save a result dict as a JSON file named by session ID."""
    file_path = RESULTS_DIR / f"{session_id}.json"

    payload = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data,
    }

    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_path
