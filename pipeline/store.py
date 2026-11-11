"""
JSONL-backed score store. One CaseScore per line, append-only.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .analyze import CaseScore


class ScoreStore:
    def __init__(self, path) -> None:
        self.path = Path(path)

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # tolerate a torn final line after a crash
        return rows

    def load_existing_hashes(self) -> set[str]:
        """Hashes of successfully scored cases. Failed rows are not included, so they are retried."""
        return {r["content_hash"] for r in self._rows() if not r.get("error")}

    def append(self, scores: Iterable[CaseScore]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with self.path.open("a") as f:
            for s in scores:
                f.write(json.dumps(asdict(s)) + "\n")
                n += 1
        return n

    def load_scores(self) -> list[CaseScore]:
        return [CaseScore(**r) for r in self._rows()]
