#!/usr/bin/env python3
"""Interactive helper to pick host institutes and write IDs into config.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

OPENALEX_INSTITUTIONS_ENDPOINT = "https://api.openalex.org/institutions"


class ProgressBar:
    def __init__(self, width: int = 28):
        self.width = width

    def update(self, label: str, current: int, total: int) -> None:
        total = max(total, 1)
        ratio = min(max(current / total, 0), 1)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        print(f"[{bar}] {int(ratio * 100):3d}% | {label}")




def add_openalex_auth(params: dict[str, Any], api_key: str) -> dict[str, Any]:
    enriched = dict(params)
    if api_key:
        enriched["api_key"] = api_key
    return enriched
def safe_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    full_url = f"{url}?{urlencode(params)}"
    with urlopen(full_url, timeout=45) as r:
        return json.loads(r.read().decode("utf-8"))


def search_institutions(query: str, openalex_api_key: str, per_page: int = 10) -> list[dict[str, Any]]:
    data = safe_get(
        OPENALEX_INSTITUTIONS_ENDPOINT,
        add_openalex_auth({
            "search": query,
            "per-page": per_page,
            "select": "id,display_name,country_code,type,works_count",
        }, openalex_api_key),
    )
    return data.get("results", [])


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("config.json not found. Copy config.example.json first.")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    pb = ProgressBar()
    cfg_path = Path("config.json")
    try:
        pb.update("Loading config", 0, 1)
        cfg = load_config(cfg_path)
        pb.update("Loading config", 1, 1)

        openalex_api_key = str(cfg.get("openalex_api_key", "")).strip()
        chosen_ids: list[str] = []
        while True:
            query = input("Search institution name (or press Enter to finish): ").strip()
            if not query:
                break

            pb.update("Searching institutions", 0, 1)
            results = search_institutions(query, openalex_api_key)
            pb.update("Searching institutions", 1, 1)

            if not results:
                print("No institutions found. Try another query.")
                continue

            print("\nCandidates:")
            for idx, inst in enumerate(results, start=1):
                print(
                    f"{idx:2d}. {inst.get('display_name')} "
                    f"[{inst.get('country_code', 'NA')}] "
                    f"type={inst.get('type', 'NA')} works={inst.get('works_count', 'NA')} id={inst.get('id')}"
                )

            pick = input("Select index (comma separated, e.g., 1,3) or Enter to skip: ").strip()
            if not pick:
                continue

            for token in pick.split(","):
                token = token.strip()
                if not token.isdigit():
                    continue
                i = int(token)
                if 1 <= i <= len(results):
                    iid = results[i - 1].get("id")
                    if iid and iid not in chosen_ids:
                        chosen_ids.append(iid)

            print(f"Current chosen host institute IDs: {chosen_ids}\n")

        if chosen_ids:
            cfg["institution_ids"] = chosen_ids
            cfg.setdefault("openalex_api_key", openalex_api_key)
            pb.update("Saving updated config", 0, 1)
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            pb.update("Saving updated config", 1, 1)
            print(f"Updated config.json with {len(chosen_ids)} host institutes.")
        else:
            print("No changes made to config.json.")

        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
