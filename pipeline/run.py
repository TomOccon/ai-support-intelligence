"""
CLI: python -m pipeline.run --cases cases.csv --comments comments.csv --out out/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .analyze import ThrottledAnalyzer
from .llm import AnthropicClient, StubLLMClient
from .prepare import prepare_cases
from .rca import build_rca, render_markdown
from .store import ScoreStore


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.run")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--comments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm", choices=["stub", "anthropic"], default="stub")
    ap.add_argument("--incremental", action="store_true",
                    help="skip cases already in scores.jsonl and append new scores")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    store = ScoreStore(out / "scores.jsonl")
    if not args.incremental and store.path.exists():
        store.path.unlink()

    cases = pd.read_csv(args.cases)
    comments = pd.read_csv(args.comments)
    prepared = prepare_cases(cases, comments)

    try:
        client = StubLLMClient() if args.llm == "stub" else AnthropicClient()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    existing = store.load_existing_hashes()
    new_scores = ThrottledAnalyzer(client, existing).score_cases(prepared)
    store.append(new_scores)

    report = build_rca(store.load_scores())
    (out / "rca_report.md").write_text(render_markdown(report))
    (out / "rca_report.json").write_text(json.dumps(report, indent=2))

    print(f"Scored {len(new_scores)} new cases, skipped {len(prepared) - len(new_scores)}.")
    print(f"Wrote {store.path} and {out / 'rca_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
