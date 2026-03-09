#!/usr/bin/env python3
"""Step 3: classify topic relevance from an already filtered dataset."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from config_loader import load_json_file
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"


@dataclass
class TopicConfig:
    topics: list[str]
    cooling_taxonomy: dict[str, list[str]]
    results_dir: str
    output_filtered_json: str
    output_relevant_jsonl: str
    output_relevant_tagged_json: str
    cheap_model: str


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


def safe_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], max_retries: int = 3) -> dict[str, Any]:
    req = Request(url=url, data=json.dumps(payload).encode("utf-8"), method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    for i in range(1, max_retries + 1):
        try:
            with urlopen(req, timeout=90) as r:
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
        cheap_model=str(raw["cheap_model"]),
    )


def load_prompt_config_from_raw(raw: dict[str, Any]) -> PromptConfig:
    return PromptConfig(
        relevance_system_prompt=str(raw["relevance_system_prompt"]),
        relevance_user_prompt_template=str(raw["relevance_user_prompt_template"]),
    )


def llm_relevance(row: dict[str, Any], cfg: TopicConfig, prompt_cfg: PromptConfig, api_key: str) -> dict[str, Any]:
    user_prompt = prompt_cfg.relevance_user_prompt_template.format(
        topics_json=json.dumps(cfg.topics, ensure_ascii=False),
        cooling_taxonomy_json=json.dumps(cfg.cooling_taxonomy, ensure_ascii=False),
        title=row.get("title", ""),
        abstract=(row.get("abstract") or "")[:5000],
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
    return {
        "relevant": False,
        "matched_topics": [],
        "cooling_level": "unknown",
        "cooling_methodology": "",
        "rationale": "Malformed model output.",
    }


def keyword_taxonomy_tag(row: dict[str, Any], cfg: TopicConfig) -> tuple[str, str]:
    text = f"{row.get('title','')} {row.get('abstract','')}".lower()
    for level, methods in cfg.cooling_taxonomy.items():
        for method in methods:
            if method.lower() in text:
                return level, method
    return "unknown", ""


def keyword_relevance(row: dict[str, Any], topics: list[str]) -> dict[str, Any]:
    text = f"{row.get('title','')} {row.get('abstract','')}".lower()
    matches = [t for t in topics if t.lower() in text]
    return {"relevant": bool(matches), "matched_topics": matches, "rationale": "Keyword fallback matching."}


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
        api_key = os.getenv("OPENAI_API_KEY", "")

        pb.update("Classifying filtered papers by topics", 0, len(filtered))
        relevant = []
        for i, row in enumerate(filtered, start=1):
            decision = llm_relevance(row, cfg, prompt_cfg, api_key) if api_key else keyword_relevance(row, cfg.topics)
            if bool(decision.get("relevant")):
                out = dict(row)
                out["matched_topics"] = decision.get("matched_topics", [])
                out["cooling_level"] = str(decision.get("cooling_level", "unknown"))
                out["cooling_methodology"] = str(decision.get("cooling_methodology", ""))
                out["rationale"] = decision.get("rationale", "")

                if (not out["cooling_level"] or out["cooling_level"] == "unknown") and cfg.cooling_taxonomy:
                    fallback_level, fallback_method = keyword_taxonomy_tag(row, cfg)
                    out["cooling_level"] = fallback_level
                    out["cooling_methodology"] = fallback_method

                relevant.append(out)
            pb.update("Classifying filtered papers by topics", i, len(filtered))

        out_dir = Path(cfg.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / cfg.output_relevant_jsonl
        with out_path.open("w", encoding="utf-8") as f:
            for r in relevant:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        tagged_out_path = out_dir / cfg.output_relevant_tagged_json
        tagged_out_path.write_text(json.dumps(relevant, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"Relevant list saved: {out_path.resolve()}")
        print(f"Relevant tagged JSON saved: {tagged_out_path.resolve()}")
        print(f"Relevant entries: {len(relevant)}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
