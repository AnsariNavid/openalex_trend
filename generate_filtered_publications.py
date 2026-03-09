#!/usr/bin/env python3
"""Generate filtered publication lists for topic-focused collaboration analysis."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"


@dataclass
class TopicConfig:
    institution_ids: list[str]
    from_date: str
    to_date: str
    max_pages: int
    per_page: int
    topics: list[str]
    results_dir: str
    output_filtered_json: str
    output_relevant_jsonl: str
    cheap_model: str
    openalex_api_key: str
    prompt_config_path: str


@dataclass
class PromptConfig:
    relevance_system_prompt: str
    relevance_user_prompt_template: str


class ProgressBar:
    def __init__(self, width: int = 28):
        self.width = width

    def update(self, label: str, current: int, total: int) -> None:
        total = max(total, 1)
        ratio = min(max(current / total, 0), 1)
        bar = "#" * int(self.width * ratio) + "-" * (self.width - int(self.width * ratio))
        print(f"[{bar}] {int(ratio*100):3d}% | {label}")


def safe_get(url: str, params: dict[str, Any], max_retries: int = 3) -> dict[str, Any]:
    full = f"{url}?{urlencode(params)}"
    for i in range(1, max_retries + 1):
        try:
            with urlopen(full, timeout=45) as r:
                return json.loads(r.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            if i == max_retries:
                raise
            time.sleep(i)
    return {}


def safe_post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    req = Request(url=url, data=json.dumps(payload).encode("utf-8"), method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    with urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8"))


def add_openalex_auth(params: dict[str, Any], api_key: str) -> dict[str, Any]:
    out = dict(params)
    if api_key:
        out["api_key"] = api_key
    return out


def invert_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    max_pos = max((max(v) for v in inverted.values() if v), default=-1)
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for token, positions in inverted.items():
        for pos in positions:
            if 0 <= pos < len(words):
                words[pos] = token
    return " ".join(w for w in words if w).strip()


def load_topic_config(path: str = "topic_config.json") -> TopicConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for d in ("from_date", "to_date"):
        datetime.strptime(raw[d], "%Y-%m-%d")
    return TopicConfig(
        institution_ids=list(raw["institution_ids"]),
        from_date=str(raw["from_date"]),
        to_date=str(raw["to_date"]),
        max_pages=int(raw["max_pages"]),
        per_page=min(200, int(raw["per_page"])),
        topics=[str(x) for x in raw["topics"]],
        results_dir=str(raw["results_dir"]),
        output_filtered_json=str(raw["output_filtered_json"]),
        output_relevant_jsonl=str(raw["output_relevant_jsonl"]),
        cheap_model=str(raw["cheap_model"]),
        openalex_api_key=str(raw.get("openalex_api_key", "")).strip(),
        prompt_config_path=str(raw["prompt_config_path"]),
    )


def load_prompt_config(path: str) -> PromptConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return PromptConfig(
        relevance_system_prompt=str(raw["relevance_system_prompt"]),
        relevance_user_prompt_template=str(raw["relevance_user_prompt_template"]),
    )


def fetch_works(cfg: TopicConfig, pb: ProgressBar) -> list[dict[str, Any]]:
    cursor = "*"
    works: list[dict[str, Any]] = []
    pb.update("Fetching host publications", 0, cfg.max_pages)
    for page in range(1, cfg.max_pages + 1):
        params = {
            "filter": ",".join([
                f"institutions.id:{'|'.join(cfg.institution_ids)}",
                f"from_publication_date:{cfg.from_date}",
                f"to_publication_date:{cfg.to_date}",
                "primary_location.source.type:journal|conference",
            ]),
            "per-page": cfg.per_page,
            "cursor": cursor,
            "select": "id,display_name,publication_year,publication_date,authorships,concepts,abstract_inverted_index,primary_location",
        }
        data = safe_get(OPENALEX_WORKS_ENDPOINT, add_openalex_auth(params, cfg.openalex_api_key))
        batch = data.get("results", [])
        if not batch:
            pb.update("Fetching host publications", page, cfg.max_pages)
            break
        works.extend(batch)
        cursor = data.get("meta", {}).get("next_cursor")
        pb.update("Fetching host publications", page, cfg.max_pages)
        if not cursor:
            break
    return works


def extract_metadata(work: dict[str, Any], host_ids: set[str]) -> dict[str, Any]:
    german = set()
    authors = []
    for a in work.get("authorships", []):
        nm = (a.get("author") or {}).get("display_name")
        if nm:
            authors.append(nm)
        for inst in a.get("institutions", []):
            iid = inst.get("id", "")
            if iid and iid not in host_ids and inst.get("country_code") == "DE":
                german.add(inst.get("display_name", "Unknown institution"))
    return {
        "id": work.get("id"),
        "title": work.get("display_name"),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "source_type": (((work.get("primary_location") or {}).get("source") or {}).get("type")),
        "authors": authors[:25],
        "german_collaborator_institutes": sorted(german),
        "concepts": [c.get("display_name") for c in work.get("concepts", []) if c.get("display_name")][:20],
        "abstract": invert_abstract(work.get("abstract_inverted_index")),
    }


def has_german_collab(work: dict[str, Any], host_ids: set[str]) -> bool:
    for a in work.get("authorships", []):
        for inst in a.get("institutions", []):
            iid = inst.get("id", "")
            if iid and iid not in host_ids and inst.get("country_code") == "DE":
                return True
    return False


def llm_relevance(meta: dict[str, Any], cfg: TopicConfig, prompt_cfg: PromptConfig, api_key: str) -> dict[str, Any]:
    user_prompt = prompt_cfg.relevance_user_prompt_template.format(
        topics_json=json.dumps(cfg.topics, ensure_ascii=False),
        title=meta.get("title", ""),
        abstract=meta.get("abstract", "")[:5000],
    )
    payload = {
        "model": cfg.cheap_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt_cfg.relevance_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = safe_post_json(OPENAI_CHAT_ENDPOINT, payload, headers)
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {"relevant": False, "matched_topics": [], "rationale": "Malformed model output."}


def keyword_relevance(meta: dict[str, Any], topics: list[str]) -> dict[str, Any]:
    text = f"{meta.get('title','')} {meta.get('abstract','')}".lower()
    matches = [t for t in topics if t.lower() in text]
    return {"relevant": bool(matches), "matched_topics": matches, "rationale": "Keyword fallback matching."}


def main() -> int:
    pb = ProgressBar()
    try:
        pb.update("Loading topic config", 0, 1)
        cfg = load_topic_config("topic_config.json")
        pb.update("Loading topic config", 1, 1)

        pb.update("Loading prompt config", 0, 1)
        prompt_cfg = load_prompt_config(cfg.prompt_config_path)
        pb.update("Loading prompt config", 1, 1)

        host_ids = set(cfg.institution_ids)
        works = fetch_works(cfg, pb)

        pb.update("Filtering journal/conference + German collaborations", 0, len(works))
        filtered: list[dict[str, Any]] = []
        for i, w in enumerate(works, start=1):
            if has_german_collab(w, host_ids):
                filtered.append(extract_metadata(w, host_ids))
            pb.update("Filtering journal/conference + German collaborations", i, len(works))

        api_key = os.getenv("OPENAI_API_KEY", "")
        pb.update("Classifying filtered papers by topics", 0, len(filtered))
        relevant: list[dict[str, Any]] = []
        for i, row in enumerate(filtered, start=1):
            decision = llm_relevance(row, cfg, prompt_cfg, api_key) if api_key else keyword_relevance(row, cfg.topics)
            if bool(decision.get("relevant")):
                row["matched_topics"] = decision.get("matched_topics", [])
                row["rationale"] = decision.get("rationale", "")
                relevant.append(row)
            pb.update("Classifying filtered papers by topics", i, len(filtered))

        pb.update("Saving outputs", 0, 1)
        out_dir = Path(cfg.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / cfg.output_filtered_json).write_text(json.dumps(filtered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        with (out_dir / cfg.output_relevant_jsonl).open("w", encoding="utf-8") as f:
            for row in relevant:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        pb.update("Saving outputs", 1, 1)

        print(f"Filtered dataset saved: {(out_dir / cfg.output_filtered_json).resolve()}")
        print(f"Relevant list saved: {(out_dir / cfg.output_relevant_jsonl).resolve()}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
