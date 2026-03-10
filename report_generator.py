#!/usr/bin/env python3
"""Unified report generator.

Generates a concise, table-first collaboration report from existing pipeline outputs.
Input: results/topic_relevant_tagged_papers.json (configurable)
"""

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


@dataclass
class ReportConfig:
    topics: list[str]
    topic_taxonomy: dict[str, list[str]]
    category_field: str
    method_field: str
    results_dir: str
    output_relevant_jsonl: str
    output_relevant_tagged_json: str
    output_report: str
    analysis_model: str
    report_system_prompt: str
    report_user_prompt_template: str


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


def load_config(path: str = "config.json") -> ReportConfig:
    raw = load_json_file(path)
    taxonomy = raw.get("topic_taxonomy") or raw.get("cooling_taxonomy") or {}
    system_prompt = str(raw.get("report_system_prompt", raw.get("analysis_system_prompt", "")))
    user_prompt = str(raw.get("report_user_prompt_template", raw.get("analysis_user_prompt_template", "")))
    return ReportConfig(
        topics=[str(x) for x in raw.get("topics", [])],
        topic_taxonomy={str(k): [str(v) for v in vals] for k, vals in taxonomy.items()},
        category_field=str(raw.get("category_label_field", "topic_category")),
        method_field=str(raw.get("method_label_field", "topic_method")),
        results_dir=str(raw["results_dir"]),
        output_relevant_jsonl=str(raw["output_relevant_jsonl"]),
        output_relevant_tagged_json=str(raw.get("output_relevant_tagged_json", "topic_relevant_tagged_papers.json")),
        output_report=str(raw.get("output_report", "collaboration_report.md")),
        analysis_model=str(raw.get("analysis_model", "gpt-4.1")),
        report_system_prompt=system_prompt,
        report_user_prompt_template=user_prompt,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rows(cfg: ReportConfig) -> tuple[list[dict[str, Any]], str]:
    out_dir = Path(cfg.results_dir)
    tagged = out_dir / cfg.output_relevant_tagged_json
    if not tagged.exists():
        return [], str(tagged)

    data = json.loads(tagged.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, str(tagged)
    return [], str(tagged)


def org_type(name: str) -> str:
    n = name.lower()
    if any(t in n for t in ["gmbh", "ag", "inc", "ltd", "llc", "corp", "co."]):
        return "Company"
    if any(t in n for t in ["university", "universit", "college"]):
        return "University"
    return "Institute/Other"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(lines)


def build_stats(rows: list[dict[str, Any]], cfg: ReportConfig) -> dict[str, Any]:
    categories = list(cfg.topic_taxonomy.keys())
    if not categories:
        categories = sorted({str(r.get(cfg.category_field, "unknown")) for r in rows})
    if "unknown" not in categories:
        categories.append("unknown")

    cat_count = Counter()
    meth_count = Counter()
    year_cat: dict[int, Counter[str]] = defaultdict(Counter)
    partner_focus: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for r in rows:
        cat = str(r.get(cfg.category_field, "unknown") or "unknown")
        meth = str(r.get(cfg.method_field, "") or "")
        year = r.get("publication_year")
        cat_count[cat] += 1
        if meth:
            meth_count[meth] += 1
        if isinstance(year, int):
            year_cat[year][cat] += 1
        for p in (r.get("german_collaborator_institutes") or []):
            p_name = str(p)
            partner_focus[(p_name, org_type(p_name))][cat] += 1

    partner_rows = []
    for (name, ptype), cnt in partner_focus.items():
        partner_rows.append({
            "partner": name,
            "type": ptype,
            "papers": sum(cnt.values()),
            "dominant_category": cnt.most_common(1)[0][0] if cnt else "unknown",
            "category_counts": dict(cnt),
        })
    partner_rows.sort(key=lambda x: x["papers"], reverse=True)

    return {
        "relevant_count": len(rows),
        "category_counts": {k: cat_count.get(k, 0) for k in categories if cat_count.get(k, 0) > 0},
        "method_counts": dict(meth_count),
        "year_category_counts": {str(y): dict(c) for y, c in sorted(year_cat.items())},
        "partner_focus": partner_rows[:12],
    }


def fallback_report(rows: list[dict[str, Any]], stats: dict[str, Any], cfg: ReportConfig) -> str:
    cat_rows = [[k, str(v)] for k, v in stats["category_counts"].items()]
    partner_rows = [[p["partner"], p["type"], p["dominant_category"], str(p["papers"])] for p in stats["partner_focus"][:10]]
    top_methods = sorted(stats["method_counts"].items(), key=lambda x: x[1], reverse=True)[:8]

    trend_lines = []
    for y, counts in sorted(stats["year_category_counts"].items()):
        dom = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:2]
        trend_lines.append(f"- {y}: " + ", ".join([f"{k} ({v})" for k, v in dom]))

    out = [
        "# Collaboration Report",
        "",
        f"- Relevant papers analyzed: **{stats['relevant_count']}**",
        "",
        "## Category Relevance Overview",
        table(["Category", "Papers"], cat_rows or [["n/a", "0"]]),
        "",
        "## Research Focus: University/Company Collaborations",
        table(["Partner", "Type", "Dominant category", "Papers"], partner_rows or [["n/a", "n/a", "n/a", "0"]]),
        "",
        "## Most Important Breakthrough Signals",
        table(["Method/Approach", "Count"], [[m, str(c)] for m, c in top_methods] or [["n/a", "0"]]),
        "",
        "## Interesting Trends",
        *(trend_lines or ["- Not enough yearly data to identify trends."]),
        "",
        "## Future Directions",
        "- Focus investment on the highest-growth categories and methods with repeated yearly momentum.",
        "- Expand strongest university-company pairs into multi-year programs and joint demonstrators.",
        "- Prioritize cross-category translation paths from lab methods to scalable manufacturing impact.",
        "",
    ]
    return "\n".join(out)


def llm_report(rows: list[dict[str, Any]], stats: dict[str, Any], cfg: ReportConfig) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback_report(rows, stats, cfg)

    sample = [
        {
            "title": r.get("title"),
            "year": r.get("publication_year"),
            "category": r.get(cfg.category_field, "unknown"),
            "method": r.get(cfg.method_field, ""),
            "partners": (r.get("german_collaborator_institutes") or [])[:5],
        }
        for r in rows[:300]
    ]

    user_prompt = cfg.report_user_prompt_template.format(
        topics_json=json.dumps(cfg.topics, ensure_ascii=False),
        relevant_count=len(rows),
        stats_json=json.dumps(stats, ensure_ascii=False),
        paper_sample_json=json.dumps(sample, ensure_ascii=False),
    )

    payload = {
        "model": cfg.analysis_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": cfg.report_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = safe_post_json(OPENAI_CHAT_ENDPOINT, payload, headers)
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return content or fallback_report(rows, stats, cfg)
    except Exception:
        return fallback_report(rows, stats, cfg)


def main() -> int:
    cfg = load_config("config.json")
    rows, src = load_rows(cfg)
    if not rows:
        print(f"Error: no relevant data found. Expected: {src}")
        return 1

    stats = build_stats(rows, cfg)
    report = llm_report(rows, stats, cfg)

    out_path = Path(cfg.results_dir) / cfg.output_report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    (Path(cfg.results_dir) / "topic_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Report source: {src}")
    print(f"Report saved: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
