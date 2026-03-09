#!/usr/bin/env python3
"""Check abstract coverage in a filtered publication list."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/filtered_german_collab_dataset.json")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/abstract_coverage_report.json")

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    rows = load_rows(input_path)
    total = len(rows)
    with_abs = 0
    missing = []
    for r in rows:
        abs_txt = (r.get("abstract") or "").strip()
        if abs_txt:
            with_abs += 1
        else:
            missing.append({"id": r.get("id"), "title": r.get("title")})

    coverage = {
        "input_file": str(input_path),
        "total_entries": total,
        "entries_with_abstract": with_abs,
        "entries_missing_abstract": total - with_abs,
        "coverage_ratio": (with_abs / total) if total else 0,
        "missing_entries": missing,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Abstract coverage report saved: {output_path.resolve()}")
    print(f"Coverage: {with_abs}/{total} ({coverage['coverage_ratio']:.2%})")
    print(f"Missing abstracts: {coverage['entries_missing_abstract']}")
    if missing:
        preview = missing[:10]
        print("Sample missing entries (up to 10):")
        for m in preview:
            print(f"- {m.get('title', 'Unknown title')} ({m.get('id', 'no-id')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
