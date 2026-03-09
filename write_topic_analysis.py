#!/usr/bin/env python3
"""Write analysis from an existing filtered/relevant publication list."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"


@dataclass
class TopicConfig:
    topics: list[str]
    results_dir: str
    output_relevant_jsonl: str
    output_analysis_md: str
    analysis_model: str
    prompt_config_path: str


@dataclass
class PromptConfig:
    analysis_system_prompt: str
    analysis_user_prompt_template: str


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


def load_topic_config(path: str = "topic_config.json") -> TopicConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return TopicConfig(
        topics=[str(x) for x in raw["topics"]],
        results_dir=str(raw["results_dir"]),
        output_relevant_jsonl=str(raw["output_relevant_jsonl"]),
        output_analysis_md=str(raw["output_analysis_md"]),
        analysis_model=str(raw["analysis_model"]),
        prompt_config_path=str(raw["prompt_config_path"]),
    )


def load_prompt_config(path: str) -> PromptConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return PromptConfig(
        analysis_system_prompt=str(raw["analysis_system_prompt"]),
        analysis_user_prompt_template=str(raw["analysis_user_prompt_template"]),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_stats(rows: list[dict[str, Any]], topics: list[str]) -> dict[str, Any]:
    by_topic = Counter()
    by_year_topic: dict[str, Counter[int]] = {t: Counter() for t in topics}
    for r in rows:
        y = r.get("publication_year")
        for t in r.get("matched_topics", []):
            by_topic[t] += 1
            if isinstance(y, int):
                by_year_topic.setdefault(t, Counter())[y] += 1
    return {
        "topic_counts": dict(by_topic),
        "topic_year_counts": {t: dict(sorted(c.items())) for t, c in by_year_topic.items()},
    }


def generate_analysis(cfg: TopicConfig, prompt_cfg: PromptConfig, rows: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        lines = ["# Topic-Focused Collaboration Analysis", "", f"- Relevant papers: {len(rows)}", "", "## Topic Coverage"]
        for t, c in sorted(stats["topic_counts"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {t}: {c}")
        lines.append("\n## Actionable Next Steps\n- Prioritize topics with sustained yearly activity and strongest German partner participation.")
        return "\n".join(lines) + "\n"

    sample = [
        {
            "title": r.get("title"),
            "year": r.get("publication_year"),
            "matched_topics": r.get("matched_topics", []),
            "german_collaborator_institutes": r.get("german_collaborator_institutes", [])[:5],
            "rationale": r.get("rationale", ""),
        }
        for r in rows[:300]
    ]

    user_prompt = prompt_cfg.analysis_user_prompt_template.format(
        topics_json=json.dumps(cfg.topics, ensure_ascii=False),
        relevant_count=len(rows),
        stats_json=json.dumps(stats, ensure_ascii=False),
        paper_sample_json=json.dumps(sample, ensure_ascii=False),
    )

    payload = {
        "model": cfg.analysis_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": prompt_cfg.analysis_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = safe_post_json(OPENAI_CHAT_ENDPOINT, payload, headers)
        return resp.get("choices", [{}])[0].get("message", {}).get("content", "# Topic-Focused Collaboration Analysis\n\nNo analysis generated.\n")
    except Exception:
        return "# Topic-Focused Collaboration Analysis\n\nAnalysis generation failed due to network/API limits.\n"


def main() -> int:
    cfg = load_topic_config("topic_config.json")
    prompt_cfg = load_prompt_config(cfg.prompt_config_path)
    relevant_path = Path(cfg.results_dir) / cfg.output_relevant_jsonl
    if not relevant_path.exists():
        print(f"Error: relevant list not found: {relevant_path}")
        return 1

    rows = load_jsonl(relevant_path)
    stats = build_stats(rows, cfg.topics)
    analysis = generate_analysis(cfg, prompt_cfg, rows, stats)

    out_path = Path(cfg.results_dir) / cfg.output_analysis_md
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(analysis, encoding="utf-8")
    (Path(cfg.results_dir) / "topic_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Analysis saved: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
