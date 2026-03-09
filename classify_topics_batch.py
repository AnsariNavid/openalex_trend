#!/usr/bin/env python3
"""Batch classifier: send up to 100 papers per model request and select relevant ones."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config_loader import load_json_file

OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"


@dataclass
class TopicConfig:
    topics: list[str]
    cooling_taxonomy: dict[str, list[str]]
    results_dir: str
    output_filtered_json: str
    output_relevant_jsonl: str
    output_relevant_tagged_json: str
    batch_classification_model: str
    batch_size: int


@dataclass
class PromptConfig:
    batch_relevance_system_prompt: str
    batch_relevance_user_prompt_template: str


class ProgressBar:
    def __init__(self, width: int = 28):
        self.width = width

    def update(self, label: str, current: int, total: int) -> None:
        total = max(total, 1)
        ratio = min(max(current / total, 0), 1)
        bar = "#" * int(self.width * ratio) + "-" * (self.width - int(self.width * ratio))
        print(f"[{bar}] {int(ratio*100):3d}% | {label}")


def safe_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], max_retries: int = 3) -> dict[str, Any]:
    req = Request(url=url, data=json.dumps(payload).encode("utf-8"), method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    for i in range(1, max_retries + 1):
        try:
            with urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            if i == max_retries:
                raise
            time.sleep(i)
    return {}


def load_topic_config(path: str = "config.json") -> TopicConfig:
    raw = load_json_file(path)
    return TopicConfig(
        topics=[str(x) for x in raw["topics"]],
        cooling_taxonomy={
            str(level): [str(m) for m in methods]
            for level, methods in (raw.get("cooling_taxonomy") or {}).items()
        },
        results_dir=str(raw["results_dir"]),
        output_filtered_json=str(raw["output_filtered_json"]),
        output_relevant_jsonl=str(raw["output_relevant_jsonl"]),
        output_relevant_tagged_json=str(raw.get("output_relevant_tagged_json", "topic_relevant_tagged_papers.json")),
        batch_classification_model=str(raw.get("batch_classification_model", raw.get("analysis_model", "gpt-4.1"))),
        batch_size=max(1, min(100, int(raw.get("batch_classification_batch_size", 100)))),
    )


def load_prompt_config_from_raw(raw: dict[str, Any]) -> PromptConfig:
    return PromptConfig(
        batch_relevance_system_prompt=str(raw["batch_relevance_system_prompt"]),
        batch_relevance_user_prompt_template=str(raw["batch_relevance_user_prompt_template"]),
    )


def keyword_taxonomy_tag(title: str, abstract: str, cfg: TopicConfig) -> tuple[str, str]:
    text = f"{title} {abstract}".lower()
    for level, methods in cfg.cooling_taxonomy.items():
        for method in methods:
            if method.lower() in text:
                return level, method
    return "unknown", ""


def keyword_is_relevant(title: str, abstract: str, topics: list[str]) -> tuple[bool, list[str]]:
    text = f"{title} {abstract}".lower()
    broad_terms = [
        "thermal management",
        "thermal",
        "cooling",
        "heat dissipation",
        "heat transfer",
        "temperature control",
        "therm",
    ]
    matches = [t for t in topics if t.lower() in text]
    matches.extend([t for t in broad_terms if t in text and t not in matches])
    return bool(matches), matches


def chunked(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def llm_batch_decision(batch: list[dict[str, Any]], cfg: TopicConfig, prompt_cfg: PromptConfig, api_key: str) -> list[dict[str, Any]]:
    payload_rows = []
    for row in batch:
        payload_rows.append({
            "id": row.get("id", ""),
            "title": row.get("title", ""),
            "abstract": (row.get("abstract") or "")[:3000],
        })

    user_prompt = prompt_cfg.batch_relevance_user_prompt_template.format(
        topics_json=json.dumps(cfg.topics, ensure_ascii=False),
        cooling_taxonomy_json=json.dumps(cfg.cooling_taxonomy, ensure_ascii=False),
        papers_json=json.dumps(payload_rows, ensure_ascii=False),
    )

    payload = {
        "model": cfg.batch_classification_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt_cfg.batch_relevance_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = safe_post_json(OPENAI_CHAT_ENDPOINT, payload, headers)
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    try:
        parsed = json.loads(content)
        items = parsed.get("decisions", []) if isinstance(parsed, dict) else []
        return items if isinstance(items, list) else []
    except json.JSONDecodeError:
        return []


def main() -> int:
    pb = ProgressBar()
    try:
        pb.update("Loading topic config", 0, 1)
        cfg = load_topic_config("config.json")
        pb.update("Loading topic config", 1, 1)

        pb.update("Loading prompt config", 0, 1)
        raw_cfg = load_json_file("config.json")
        prompt_cfg = load_prompt_config_from_raw(raw_cfg)
        pb.update("Loading prompt config", 1, 1)

        filtered_path = Path(cfg.results_dir) / cfg.output_filtered_json
        if not filtered_path.exists():
            print(f"Error: filtered dataset not found: {filtered_path}")
            return 1

        filtered = json.loads(filtered_path.read_text(encoding="utf-8"))
        batches = chunked(filtered, cfg.batch_size)
        api_key = os.getenv("OPENAI_API_KEY", "")

        relevant: list[dict[str, Any]] = []
        by_id = {str(row.get("id", "")): row for row in filtered}

        pb.update("Batch classifying papers", 0, len(batches))
        for i, batch in enumerate(batches, start=1):
            if api_key:
                decisions = llm_batch_decision(batch, cfg, prompt_cfg, api_key)
            else:
                decisions = []

            if decisions:
                for d in decisions:
                    rid = str(d.get("id", ""))
                    if not rid or rid not in by_id:
                        continue
                    if not bool(d.get("relevant")):
                        continue
                    base = dict(by_id[rid])
                    base["matched_topics"] = d.get("matched_topics", [])
                    base["cooling_level"] = str(d.get("cooling_level", "unknown"))
                    base["cooling_methodology"] = str(d.get("cooling_methodology", ""))
                    base["rationale"] = d.get("rationale", "")
                    if base["cooling_level"] == "unknown":
                        lvl, meth = keyword_taxonomy_tag(base.get("title", ""), base.get("abstract", ""), cfg)
                        base["cooling_level"] = lvl
                        base["cooling_methodology"] = meth
                    relevant.append(base)
            else:
                for row in batch:
                    ok, matches = keyword_is_relevant(row.get("title", ""), row.get("abstract", ""), cfg.topics)
                    if not ok:
                        continue
                    lvl, meth = keyword_taxonomy_tag(row.get("title", ""), row.get("abstract", ""), cfg)
                    out = dict(row)
                    out["matched_topics"] = matches
                    out["cooling_level"] = lvl
                    out["cooling_methodology"] = meth
                    out["rationale"] = "Keyword fallback matching."
                    relevant.append(out)

            pb.update("Batch classifying papers", i, len(batches))

        out_dir = Path(cfg.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = out_dir / cfg.output_relevant_jsonl
        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in relevant:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        out_json = out_dir / cfg.output_relevant_tagged_json
        out_json.write_text(json.dumps(relevant, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"Relevant list saved: {out_jsonl.resolve()}")
        print(f"Relevant tagged JSON saved: {out_json.resolve()}")
        print(f"Relevant entries: {len(relevant)}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
