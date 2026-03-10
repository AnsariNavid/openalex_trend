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
    topic_taxonomy: dict[str, list[str]]
    keyword_hints: list[str]
    category_field: str
    method_field: str
    checkpoint_every: int
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
    taxonomy_raw = raw.get("topic_taxonomy") or raw.get("cooling_taxonomy") or {}
    return TopicConfig(
        topics=[str(x) for x in raw["topics"]],
        topic_taxonomy={str(level): [str(m) for m in methods] for level, methods in taxonomy_raw.items()},
        keyword_hints=[str(x).strip() for x in raw.get("relevance_keyword_hints", []) if str(x).strip()],
        category_field=str(raw.get("category_label_field", "topic_category")),
        method_field=str(raw.get("method_label_field", "topic_method")),
        checkpoint_every=max(1, int(raw.get("classification_checkpoint_every", 100))),
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
        topic_taxonomy_json=json.dumps(cfg.topic_taxonomy, ensure_ascii=False),
        cooling_taxonomy_json=json.dumps(cfg.topic_taxonomy, ensure_ascii=False),
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
        cfg.category_field: "unknown",
        cfg.method_field: "",
        "rationale": "Malformed model output.",
    }


def keyword_taxonomy_tag(row: dict[str, Any], cfg: TopicConfig) -> tuple[str, str]:
    text = f"{row.get('title','')} {row.get('abstract','')}".lower()
    for level, methods in cfg.topic_taxonomy.items():
        for method in methods:
            if method.lower() in text:
                return level, method
    return "unknown", ""


def keyword_relevance(row: dict[str, Any], cfg: TopicConfig) -> dict[str, Any]:
    text = f"{row.get('title','')} {row.get('abstract','')}".lower()
    matches = [t for t in cfg.topics if t.lower() in text]
    hints = cfg.keyword_hints or cfg.topics
    matches.extend([t for t in hints if t.lower() in text and t not in matches])
    return {"relevant": bool(matches), "matched_topics": matches, "rationale": "Keyword fallback matching."}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_outputs(out_jsonl: Path, out_tagged: Path, rows: list[dict[str, Any]]) -> None:
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_tagged.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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

        out_dir = Path(cfg.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        filtered_path = out_dir / cfg.output_filtered_json
        if not filtered_path.exists():
            print(f"Error: filtered dataset not found: {filtered_path}")
            return 1

        out_jsonl = out_dir / cfg.output_relevant_jsonl
        out_tagged = out_dir / cfg.output_relevant_tagged_json
        progress_path = out_dir / ".classify_topics_from_filtered.progress.json"

        filtered = json.loads(filtered_path.read_text(encoding="utf-8"))
        relevant = load_jsonl(out_jsonl)

        start_idx = 0
        if progress_path.exists():
            st = json.loads(progress_path.read_text(encoding="utf-8"))
            start_idx = int(st.get("next_index", 0))
            start_idx = max(0, min(start_idx, len(filtered)))
            print(f"Resuming classification from index {start_idx}/{len(filtered)}")
            print(f"Previously selected papers: {len(relevant)}")

        api_key = os.getenv("OPENAI_API_KEY", "")
        pb.update("Classifying filtered papers by topics", start_idx, len(filtered))

        processed_since_checkpoint = 0
        for i in range(start_idx, len(filtered)):
            row = filtered[i]
            decision = llm_relevance(row, cfg, prompt_cfg, api_key) if api_key else keyword_relevance(row, cfg)
            if bool(decision.get("relevant")):
                out = dict(row)
                out["matched_topics"] = decision.get("matched_topics", [])
                out[cfg.category_field] = str(decision.get(cfg.category_field, decision.get("cooling_level", "unknown")))
                out[cfg.method_field] = str(decision.get(cfg.method_field, decision.get("cooling_methodology", "")))
                out["rationale"] = decision.get("rationale", "")

                if (not out[cfg.category_field] or out[cfg.category_field] == "unknown") and cfg.topic_taxonomy:
                    fallback_level, fallback_method = keyword_taxonomy_tag(row, cfg)
                    out[cfg.category_field] = fallback_level
                    out[cfg.method_field] = fallback_method

                relevant.append(out)

            processed_since_checkpoint += 1
            pb.update("Classifying filtered papers by topics", i + 1, len(filtered))

            if processed_since_checkpoint >= cfg.checkpoint_every:
                analyzed_count = i + 1
                save_outputs(out_jsonl, out_tagged, relevant)
                progress_path.write_text(
                    json.dumps({"next_index": analyzed_count, "selected_count": len(relevant)}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"Checkpoint: analyzed {analyzed_count}/{len(filtered)} papers | "
                    f"relevant selected so far: {len(relevant)}"
                )
                processed_since_checkpoint = 0

        save_outputs(out_jsonl, out_tagged, relevant)
        if progress_path.exists():
            progress_path.unlink()

        print(f"Relevant list saved: {out_jsonl.resolve()}")
        print(f"Relevant tagged JSON saved: {out_tagged.resolve()}")
        print(f"Relevant entries (final): {len(relevant)}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
