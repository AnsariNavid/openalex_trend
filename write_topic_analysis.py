#!/usr/bin/env python3
"""Write concise collaboration analysis from tagged relevant publications."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config_loader import load_json_file

OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
LEVEL_ORDER = ["chip-level", "board-level", "system-level", "unknown"]
CATEGORY_FACTS = {
    "chip-level": [
        "Microfluidics",
        "Thermal Interface Materials (TIMs)",
        "hotspots mitigation in 2.5D/3D ICs",
    ],
    "board-level": [
        "Integrated heat pipes",
        "vapor chambers",
        "PCB thermal via optimization",
    ],
    "system-level": [
        "Cold-plate cooling",
        "Two-phase immersion cooling",
        "Rack-scale thermal management",
    ],
}


@dataclass
class TopicConfig:
    topics: list[str]
    results_dir: str
    output_relevant_jsonl: str
    output_relevant_tagged_json: str
    output_analysis_md: str
    analysis_model: str


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


def load_topic_config(path: str = "config.json") -> TopicConfig:
    raw = load_json_file(path)
    return TopicConfig(
        topics=[str(x) for x in raw["topics"]],
        results_dir=str(raw["results_dir"]),
        output_relevant_jsonl=str(raw["output_relevant_jsonl"]),
        output_relevant_tagged_json=str(raw.get("output_relevant_tagged_json", "topic_relevant_tagged_papers.json")),
        output_analysis_md=str(raw["output_analysis_md"]),
        analysis_model=str(raw["analysis_model"]),
    )


def load_prompt_config_from_raw(raw: dict[str, Any]) -> PromptConfig:
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


def load_relevant_rows(cfg: TopicConfig) -> tuple[list[dict[str, Any]], str]:
    tagged_path = Path(cfg.results_dir) / cfg.output_relevant_tagged_json
    if tagged_path.exists():
        data = json.loads(tagged_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data, str(tagged_path)

    jsonl_path = Path(cfg.results_dir) / cfg.output_relevant_jsonl
    if jsonl_path.exists():
        return load_jsonl(jsonl_path), str(jsonl_path)

    return [], str(tagged_path)


def institute_type(name: str) -> str:
    n = name.lower()
    company_terms = ["gmbh", "ag", "inc", "ltd", "corp", "company", "siemens", "bosch", "infineon", "intel"]
    if any(t in n for t in company_terms):
        return "Company"
    if "university" in n or "universit" in n or "technische hochschule" in n:
        return "University"
    return "Institute/Other"


def build_stats(rows: list[dict[str, Any]], topics: list[str]) -> dict[str, Any]:
    by_topic = Counter()
    by_level = Counter()
    methods = Counter()
    trends_by_year_level: dict[int, Counter[str]] = defaultdict(Counter)
    uni_company_focus: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for r in rows:
        year = r.get("publication_year")
        level = str(r.get("cooling_level", "unknown") or "unknown")
        method = str(r.get("cooling_methodology", "") or "")

        by_level[level] += 1
        if method:
            methods[method] += 1
        if isinstance(year, int):
            trends_by_year_level[year][level] += 1

        for t in r.get("matched_topics", []):
            by_topic[t] += 1

        partners = r.get("german_collaborator_institutes", []) or []
        for p in partners:
            p_str = str(p)
            p_type = institute_type(p_str)
            uni_company_focus[(p_str, p_type)][level] += 1

    top_pairs = []
    for (name, ptype), cnt in uni_company_focus.items():
        total = sum(cnt.values())
        dominant = cnt.most_common(1)[0][0] if cnt else "unknown"
        top_pairs.append({
            "partner": name,
            "partner_type": ptype,
            "total_papers": total,
            "dominant_level": dominant,
            "level_counts": dict(cnt),
        })

    top_pairs.sort(key=lambda x: x["total_papers"], reverse=True)

    return {
        "topic_counts": dict(by_topic),
        "level_counts": {k: by_level.get(k, 0) for k in LEVEL_ORDER if by_level.get(k, 0) > 0},
        "method_counts": dict(methods),
        "year_level_counts": {
            str(y): {lvl: c for lvl, c in cnt.items()} for y, cnt in sorted(trends_by_year_level.items())
        },
        "partner_focus": top_pairs[:12],
        "category_facts": CATEGORY_FACTS,
        "relevant_count": len(rows),
    }


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return out


def fallback_analysis(rows: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    lines = ["# Topic-Focused Collaboration Analysis", ""]
    lines.append(f"- Relevant papers: **{len(rows)}**")
    lines.append("")

    lines.append("## Collaboration by Cooling Category")
    level_rows = [[lvl, str(stats["level_counts"].get(lvl, 0))] for lvl in ["chip-level", "board-level", "system-level"]]
    lines.extend(_table(["Cooling category", "Paper count"], level_rows))
    lines.append("")

    lines.append("## Research Focus by Partner (University/Company)")
    partner_rows = []
    for p in stats["partner_focus"][:8]:
        partner_rows.append([
            p["partner"],
            p["partner_type"],
            p["dominant_level"],
            str(p["total_papers"]),
        ])
    if partner_rows:
        lines.extend(_table(["Partner", "Type", "Dominant cooling level", "Papers"], partner_rows))
    else:
        lines.append("No partner focus data available.")
    lines.append("")

    lines.append("## Key Methodology Signals")
    top_methods = sorted(stats["method_counts"].items(), key=lambda x: x[1], reverse=True)[:8]
    if top_methods:
        lines.extend(_table(["Methodology", "Count"], [[m, str(c)] for m, c in top_methods]))
    else:
        lines.append("No methodology tags available.")
    lines.append("")

    lines.append("## Important Breakthrough Signals")
    for m, c in top_methods[:3]:
        lines.append(f"- {m}: strong momentum with {c} relevant papers.")
    if not top_methods:
        lines.append("- Insufficient tagged methodology data to infer breakthrough signals.")
    lines.append("")

    lines.append("## Future Direction Outlook")
    lines.append("- Strengthen chip-to-system integration projects to translate component-level advances into deployable thermal systems.")
    lines.append("- Expand partnerships with the highest-output collaborators in each cooling level to accelerate commercialization-ready outcomes.")
    lines.append("- Prioritize methods with repeated yearly presence as near-term scalable research bets.")
    lines.append("")

    lines.append("## Category Definition Used")
    for lvl in ["chip-level", "board-level", "system-level"]:
        lines.append(f"- **{lvl}**: " + ", ".join(CATEGORY_FACTS[lvl]))

    return "\n".join(lines) + "\n"


def generate_analysis(cfg: TopicConfig, prompt_cfg: PromptConfig, rows: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback_analysis(rows, stats)

    sample = [
        {
            "title": r.get("title"),
            "year": r.get("publication_year"),
            "cooling_level": r.get("cooling_level", "unknown"),
            "cooling_methodology": r.get("cooling_methodology", ""),
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
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return content or fallback_analysis(rows, stats)
    except Exception:
        return fallback_analysis(rows, stats)


def main() -> int:
    cfg = load_topic_config("config.json")
    raw_cfg = load_json_file("config.json")
    prompt_cfg = load_prompt_config_from_raw(raw_cfg)

    rows, source_path = load_relevant_rows(cfg)
    if not rows:
        print(f"Error: relevant list not found or empty. Expected tagged JSON at: {source_path}")
        return 1

    stats = build_stats(rows, cfg.topics)
    analysis = generate_analysis(cfg, prompt_cfg, rows, stats)

    out_path = Path(cfg.results_dir) / cfg.output_analysis_md
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(analysis, encoding="utf-8")
    (Path(cfg.results_dir) / "topic_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Analysis source: {source_path}")
    print(f"Analysis saved: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
