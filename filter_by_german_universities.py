#!/usr/bin/env python3
"""Filter tagged relevant papers by an interactive list of German universities."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from config_loader import load_json_file


@dataclass
class FilterConfig:
    results_dir: str
    output_relevant_tagged_json: str
    output_university_filtered_json: str


def load_filter_config(path: str = "config.json") -> FilterConfig:
    raw = load_json_file(path)
    return FilterConfig(
        results_dir=str(raw["results_dir"]),
        output_relevant_tagged_json=str(raw.get("output_relevant_tagged_json", "topic_relevant_tagged_papers.json")),
        output_university_filtered_json=str(raw.get("output_university_filtered_json", "topic_relevant_filtered_by_universities.json")),
    )


def read_university_list() -> list[str]:
    print("Enter German university names one by one. Press Enter on an empty line to finish.")
    names: list[str] = []
    while True:
        name = input(f"University #{len(names)+1}: ").strip()
        if not name:
            break
        names.append(name)
    return names


def matches_any_university(partners: list[str], targets: list[str]) -> bool:
    partners_l = [p.lower() for p in partners]
    for t in targets:
        tl = t.lower()
        for p in partners_l:
            if tl in p or p in tl:
                return True
    return False


def main() -> int:
    try:
        cfg = load_filter_config("config.json")
        in_path = Path(cfg.results_dir) / cfg.output_relevant_tagged_json
        if not in_path.exists():
            print(f"Error: input file not found: {in_path}")
            return 1

        universities = read_university_list()
        if not universities:
            print("No universities entered. Nothing to filter.")
            return 1

        rows = json.loads(in_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            print("Error: tagged input JSON is not a list.")
            return 1

        kept = []
        for row in rows:
            partners = [str(x) for x in (row.get("german_collaborator_institutes") or [])]
            if matches_any_university(partners, universities):
                kept.append(row)

        out_path = Path(cfg.results_dir) / cfg.output_university_filtered_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"Input tagged papers: {len(rows)}")
        print(f"Filtered papers: {len(kept)}")
        print(f"Saved: {out_path.resolve()}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
