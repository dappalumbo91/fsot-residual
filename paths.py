"""Where a pass is allowed to put files.

Scores stay in results/scores. They are small and they are the record.
Live media goes in results/current/<pass> and is wiped at the start of that
pass. A playback wav and its archive are written only at the listening
grade. Every other grade is counted in memory and recorded in the score.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SCORES = RESULTS / "scores"
CURRENT = RESULTS / "current"
REFERENCE = RESULTS / "reference"
PROBE = RESULTS / "video_probe"

# The grade under ear review. Opus 96 kbps is the size bar for this grade.
LISTEN_Q = 2048


def begin_pass(name: str) -> Path:
    out = CURRENT / name
    if out.exists():
        for path in sorted(out.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_score(name: str, payload: dict) -> Path:
    SCORES.mkdir(parents=True, exist_ok=True)
    path = SCORES / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
