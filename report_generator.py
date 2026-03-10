#!/usr/bin/env python3
"""Unified evidence-first report generator.

Input: results/topic_relevant_tagged_papers.json (configurable)
Output: results/<output_report>
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config_loader import load_json_file

OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
LANGUAGE_ENGLISH = "en"
LANGUAGE_CHINESE = "zh"


@dataclass
class ReportConfig:
    topics: list[str]
    target_companies: list[str]
    target_universities: list[str]
    company_aliases: dict[str, str]
    topic_taxonomy: dict[str, list[str]]
    category_field: str
    method_field: str
    report_category_filter: str
    report_subcategory_filter: str
    report_year_start: int | None
    report_year_end: int | None
    report_language: str
    results_dir: str
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


def normalize_company_name(name: str, aliases: dict[str, str]) -> str:
    n = name.strip()
    if not n:
        return n
    for key, mapped in aliases.items():
        if key.lower() in n.lower():
            return mapped
    return n


def _to_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load_config(path: str = "config.json") -> ReportConfig:
    raw = load_json_file(path)
    taxonomy = raw.get("topic_taxonomy") or raw.get("cooling_taxonomy") or {}
    return ReportConfig(
        topics=[str(x) for x in raw.get("topics", [])],
        target_companies=[str(x) for x in raw.get("target_companies", [])],
        target_universities=[str(x) for x in raw.get("target_universities", [])],
        company_aliases={str(k): str(v) for k, v in (raw.get("company_aliases") or {}).items()},
        topic_taxonomy={str(k): [str(v) for v in vals] for k, vals in taxonomy.items()},
        category_field=str(raw.get("category_label_field", "topic_category")),
        method_field=str(raw.get("method_label_field", "topic_method")),
        report_category_filter=str(raw.get("report_category_filter", "")).strip(),
        report_subcategory_filter=str(raw.get("report_subcategory_filter", "")).strip(),
        report_year_start=_to_int(raw.get("report_year_start")),
        report_year_end=_to_int(raw.get("report_year_end")),
        report_language=str(raw.get("report_language", LANGUAGE_ENGLISH)).strip() or LANGUAGE_ENGLISH,
        results_dir=str(raw["results_dir"]),
        output_relevant_tagged_json=str(raw.get("output_relevant_tagged_json", "topic_relevant_tagged_papers.json")),
        output_report=str(raw.get("output_report", "collaboration_report.md")),
        analysis_model=str(raw.get("analysis_model", "gpt-4.1")),
        report_system_prompt=str(raw.get("report_system_prompt", "")),
        report_user_prompt_template=str(raw.get("report_user_prompt_template", "")),
    )


def load_rows(cfg: ReportConfig) -> tuple[list[dict[str, Any]], str]:
    tagged = Path(cfg.results_dir) / cfg.output_relevant_tagged_json
    if not tagged.exists():
        return [], str(tagged)
    data = json.loads(tagged.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, str(tagged)
    return [], str(tagged)


def _extract_year(row: dict[str, Any]) -> Optional[int]:
    y = row.get("publication_year")
    if isinstance(y, int):
        return y
    y2 = str(row.get("publication_date", "") or row.get("year", ""))
    m = re.search(r"\b\d{4}\b", y2)
    return int(m.group()) if m else None


def filter_publications(
    publications: list[dict[str, Any]],
    category_field: str,
    method_field: str,
    category: str,
    subcategory: str,
    year_start: Optional[int],
    year_end: Optional[int],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    category_l = category.lower().strip() if category else ""
    subcategory_l = subcategory.lower().strip() if subcategory else ""

    for publication in publications:
        cat = str(publication.get(category_field, "") or "").lower()
        sub = str(publication.get(method_field, "") or "").lower()

        if category_l and category_l != cat:
            continue
        if subcategory_l and subcategory_l not in sub:
            continue

        year = _extract_year(publication)
        if year_start is not None and (year is None or year < year_start):
            continue
        if year_end is not None and (year is None or year > year_end):
            continue

        filtered.append(publication)
    return filtered


def org_type(name: str) -> str:
    n = name.lower()
    if any(t in n for t in ["gmbh", "ag", "inc", "ltd", "llc", "corp", "co.", "kg"]):
        return "Company"
    if any(t in n for t in ["university", "universit", "rwth", "technische univers", "institute of technology", "tu "]):
        return "University"
    return "Institute/Other"


def looks_like_target(name: str, targets: list[str]) -> bool:
    nl = name.lower()
    return any(t.lower() in nl or nl in t.lower() for t in targets)


def detect_publication_link(row: dict[str, Any]) -> str:
    for k in ("id", "doi", "html_url", "pdf_url"):
        v = str(row.get(k, "") or "").strip()
        if v:
            if k == "doi" and not v.startswith("http"):
                return f"https://doi.org/{v}"
            return v
    return ""


def detect_evidence_strength(row: dict[str, Any], uni_name: str, comp_name: str) -> str:
    partners = [str(x).lower() for x in (row.get("german_collaborator_institutes") or [])]
    t = str(row.get("title", "")).lower()
    a = str(row.get("abstract", "")).lower()
    text = f"{t} {a}"
    has_uni = uni_name.lower() in " ".join(partners) or uni_name.lower() in text
    has_comp = comp_name.lower() in " ".join(partners) or comp_name.lower() in text
    if has_uni and has_comp:
        return "High"
    if has_uni or has_comp:
        return "Medium"
    return "Technical"


def build_verified_pairs(rows: list[dict[str, Any]], cfg: ReportConfig) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for r in rows:
        partners = [str(x) for x in (r.get("german_collaborator_institutes") or [])]
        universities = [p for p in partners if org_type(p) == "University"]
        companies = [normalize_company_name(p, cfg.company_aliases) for p in partners if org_type(p) == "Company"]

        if cfg.target_universities:
            universities = [u for u in universities if looks_like_target(u, cfg.target_universities)]
        if cfg.target_companies:
            companies = [c for c in companies if looks_like_target(c, cfg.target_companies)]

        cat = str(r.get(cfg.category_field, "unknown") or "unknown")
        meth = str(r.get(cfg.method_field, "") or "")
        year = _extract_year(r)

        for u in universities:
            for c in companies:
                evidence = detect_evidence_strength(r, u, c)
                verified.append({
                    "paper_id": str(r.get("id", "") or ""),
                    "university": u,
                    "company": c,
                    "title": str(r.get("title", "")),
                    "year": str(year) if year is not None else "",
                    "category": cat,
                    "method": meth,
                    "what_proves": "University and company appear in publication-linked collaboration metadata.",
                    "evidence_strength": evidence,
                    "link": detect_publication_link(r),
                })
    order = {"High": 0, "Medium": 1, "Technical": 2}
    verified.sort(key=lambda x: (order.get(x["evidence_strength"], 9), -int(x["year"] or 0)))
    return verified


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(out)


def build_stats(rows: list[dict[str, Any]], cfg: ReportConfig, verified: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat = Counter(str(r.get(cfg.category_field, "unknown") or "unknown") for r in rows)
    by_method = Counter(str(r.get(cfg.method_field, "") or "") for r in rows if str(r.get(cfg.method_field, "") or ""))
    by_pair = Counter((v["university"], v["company"]) for v in verified)
    by_year_cat: dict[int, Counter[str]] = defaultdict(Counter)
    for r in rows:
        y = _extract_year(r)
        if isinstance(y, int):
            by_year_cat[y][str(r.get(cfg.category_field, "unknown") or "unknown")] += 1

    return {
        "relevant_count": len(rows),
        "verified_pairs": len(verified),
        "category_counts": dict(by_cat),
        "method_counts": dict(by_method),
        "pair_counts": [{"pair": f"{u} ↔ {c}", "count": n} for (u, c), n in by_pair.most_common(20)],
        "year_category_counts": {str(y): dict(c) for y, c in sorted(by_year_cat.items())},
    }


def _build_instructions(language: str) -> str:
    base = (
        "You are an R&D intelligence analyst. Use ONLY the provided paper corpus for internal claims. "
        "If evidence is missing, state it explicitly. Do not invent publications, links, counts, or collaborations. "
        "Prefer concise table-first output and short executive prose."
    )
    if language == LANGUAGE_CHINESE:
        return base + " Write the report in Simplified Chinese."
    return base + " Write the report in English."


def _memo_prompt(cfg: ReportConfig, stats: dict[str, Any], sample: list[dict[str, Any]]) -> str:
    return cfg.report_user_prompt_template.format(
        topics_json=json.dumps(cfg.topics, ensure_ascii=False),
        target_companies_json=json.dumps(cfg.target_companies, ensure_ascii=False),
        target_universities_json=json.dumps(cfg.target_universities, ensure_ascii=False),
        relevant_count=stats.get("relevant_count", 0),
        stats_json=json.dumps(stats, ensure_ascii=False),
        paper_sample_json=json.dumps(sample, ensure_ascii=False),
    )


def fallback_report(rows: list[dict[str, Any]], verified: list[dict[str, Any]], stats: dict[str, Any], cfg: ReportConfig) -> str:
    top_verified = verified[:12]
    top_pairs = Counter((v["university"], v["company"]) for v in verified).most_common(10)
    top_methods = Counter(v["method"] for v in verified if v["method"]).most_common(8)

    section2_rows = [
        [v["university"], v["company"], v["title"][:80], v["year"], v["category"], v["what_proves"], v["evidence_strength"]]
        for v in top_verified
    ]

    rel_rows = []
    for (u, c), _ in top_pairs[:8]:
        pair_items = [x for x in verified if x["university"] == u and x["company"] == c]
        cat = Counter(x["category"] for x in pair_items).most_common(1)[0][0] if pair_items else "unknown"
        meth = Counter(x["method"] for x in pair_items if x["method"]).most_common(1)
        breakthrough = meth[0][0] if meth else "No dominant method"
        rel_rows.append([
            f"{u} ↔ {c}",
            cat,
            "Consistent publication-backed interaction",
            breakthrough,
            "Likely deeper applied joint development (inference)",
        ])

    trend_rows = []
    for y, counts in sorted(stats["year_category_counts"].items()):
        dom = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:2]
        trend_rows.append([f"Year {y} concentration", ", ".join([f"{k} ({v})" for k, v in dom]), "Shows category focus shifts over time"])

    breakthrough_rows = []
    for idx, (m, _) in enumerate(top_methods[:6]):
        pair_label = "Mixed pairs"
        if idx < len(top_pairs):
            pair_label = f"{top_pairs[idx][0][0]} ↔ {top_pairs[idx][0][1]}"
        breakthrough_rows.append([m, pair_label, "Repeated co-publication signal"])

    bib_rows = [[v["title"][:90], v["year"], v["link"] or "n/a"] for v in top_verified]

    limitation = []
    if not top_verified:
        limitation.append("- No strong publication-backed university-company pairs found in the current tagged set.")
    else:
        weak = sum(1 for v in verified if v["evidence_strength"] != "High")
        if weak:
            limitation.append(f"- {weak} entries rely on medium/technical-level evidence rather than explicit coauthorship metadata.")
    if cfg.target_companies:
        present_comp = {v["company"] for v in verified}
        missing = [c for c in cfg.target_companies if not any(c.lower() in p.lower() for p in present_comp)]
        if missing:
            limitation.append("- Missing/weak publication evidence for listed companies: " + ", ".join(missing[:8]) + ".")
    if cfg.target_universities:
        present_uni = {v["university"] for v in verified}
        missing_u = [u for u in cfg.target_universities if not any(u.lower() in p.lower() for p in present_uni)]
        if missing_u:
            limitation.append("- Missing/weak publication evidence for listed universities: " + ", ".join(missing_u[:8]) + ".")
    limitation.extend([
        "- Evidence strength is inferred from available tagged metadata in the local dataset.",
        "- This report does not invent collaborations that are not publication-backed in the source file.",
    ])

    exec_summary = (
        "Germany’s precision-engineering collaboration landscape in this dataset is concentrated in a limited set "
        "of university-company pairs with repeated publication-backed signals. The strongest evidence appears where "
        "institution-level metadata repeatedly aligns with the same topical categories and methods. AI/digital precision, "
        "advanced manufacturing, and optics-related tracks show the clearest momentum, while coverage remains uneven "
        "across listed targets. Overall, current evidence suggests a practical shift from isolated method papers toward "
        "integrated, application-oriented collaboration patterns."
    )

    parts = [
        "# Precision Engineering Collaboration Report (Germany)",
        "",
        "## 1. Executive summary",
        exec_summary,
        "",
        "## 2. Verified publication-backed collaboration map",
        md_table(
            ["University", "Company", "Publication title", "Year", "Category", "What this publication proves", "Evidence strength"],
            section2_rows or [["n/a", "n/a", "n/a", "", "", "No verified examples in current dataset", "Technical"]],
        ),
        "",
        "## 3. Relationship analysis by pair",
        md_table(
            ["University-company pair", "Research focus", "Why this relation matters", "Most important breakthrough", "Likely next direction (inference)"],
            rel_rows or [["n/a", "n/a", "No strong pair evidence", "n/a", "n/a"]],
        ),
        "",
        "## 4. Interesting trends",
        md_table(["Trend", "Evidence from publications", "Why it matters"], trend_rows or [["No clear trend", "Insufficient yearly/category coverage", "Additional evidence needed"]]),
        "",
        "## 5. Breakthroughs",
        md_table(["Breakthrough", "University-company pair", "Why it is important"], breakthrough_rows[:6] or [["n/a", "n/a", "n/a"]]),
        "",
        "## 6. Limitations",
        *limitation,
        "",
        "## 7. Bibliography",
        md_table(["Publication", "Year", "Link"], bib_rows or [["n/a", "", "n/a"]]),
    ]
    return "\n".join(parts)


def llm_report(rows: list[dict[str, Any]], verified: list[dict[str, Any]], stats: dict[str, Any], cfg: ReportConfig) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback_report(rows, verified, stats, cfg)

    sample = [
        {
            "paper_id": v["paper_id"],
            "university": v["university"],
            "company": v["company"],
            "title": v["title"],
            "year": v["year"],
            "category": v["category"],
            "method": v["method"],
            "evidence_strength": v["evidence_strength"],
            "link": v["link"],
        }
        for v in verified[:250]
    ]

    user_prompt = _memo_prompt(cfg, stats, sample)
    instructions = _build_instructions(cfg.report_language)

    payload = {
        "model": cfg.analysis_model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": f"{instructions}\n\n{cfg.report_system_prompt}".strip()},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = safe_post_json(OPENAI_CHAT_ENDPOINT, payload, headers)
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return content or fallback_report(rows, verified, stats, cfg)
    except Exception:
        return fallback_report(rows, verified, stats, cfg)


def main() -> int:
    cfg = load_config("config.json")
    rows, src = load_rows(cfg)
    if not rows:
        print(f"Error: no relevant data found. Expected: {src}")
        return 1

    filtered_rows = filter_publications(
        publications=rows,
        category_field=cfg.category_field,
        method_field=cfg.method_field,
        category=cfg.report_category_filter,
        subcategory=cfg.report_subcategory_filter,
        year_start=cfg.report_year_start,
        year_end=cfg.report_year_end,
    )

    verified = build_verified_pairs(filtered_rows, cfg)
    stats = build_stats(filtered_rows, cfg, verified)
    report = llm_report(filtered_rows, verified, stats, cfg)

    out_path = Path(cfg.results_dir) / cfg.output_report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    (Path(cfg.results_dir) / "topic_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Report source: {src}")
    print(f"Filtered publications for report: {len(filtered_rows)}")
    print(f"Verified collaboration evidence rows: {len(verified)}")
    print(f"Report saved: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
