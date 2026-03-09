#!/usr/bin/env python3
"""Topic-focused collaboration filter for OpenAlex publications."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
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
    output_analysis_md: str
    cheap_model: str
    analysis_model: str
    openalex_api_key: str
    prompt_config_path: str


@dataclass
class PromptConfig:
    relevance_system_prompt: str
    relevance_user_prompt_template: str
    analysis_system_prompt: str
    analysis_user_prompt_template: str


class ProgressBar:
    def __init__(self, width: int = 28):
        self.width = width

    def update(self, label: str, current: int, total: int) -> None:
        total = max(total, 1)
        ratio = min(max(current / total, 0), 1)
        fill = int(self.width * ratio)
        bar = "#" * fill + "-" * (self.width - fill)
        print(f"[{bar}] {int(ratio * 100):3d}% | {label}")


def safe_get(url: str, params: dict[str, Any], max_retries: int = 3) -> dict[str, Any]:
    full = f"{url}?{urlencode(params)}"
    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(full, timeout=45) as r:
                return json.loads(r.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            if attempt == max_retries:
                raise
            time.sleep(1.1 * attempt)
    return {}


def safe_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], max_retries: int = 3) -> dict[str, Any]:
    req = Request(url=url, data=json.dumps(payload).encode("utf-8"), method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            if attempt == max_retries:
                raise
            time.sleep(1.1 * attempt)
    return {}


def add_openalex_auth(params: dict[str, Any], api_key: str) -> dict[str, Any]:
    out = dict(params)
    if api_key:
        out["api_key"] = api_key
    return out


def invert_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    max_pos = -1
    for positions in inverted.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for token, positions in inverted.items():
        for pos in positions:
            if 0 <= pos < len(words):
                words[pos] = token
    return " ".join(w for w in words if w).strip()


def load_topic_config(path: str = "config.json") -> TopicConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("config.json not found.")
    raw = json.loads(p.read_text(encoding="utf-8"))

    required = [
        "institution_ids",
        "from_date",
        "to_date",
        "max_pages",
        "per_page",
        "topics",
        "results_dir",
        "output_filtered_json",
        "output_relevant_jsonl",
        "output_analysis_md",
        "cheap_model",
        "analysis_model",
        "openalex_api_key",
        "prompt_config_path",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Missing keys: {', '.join(missing)}")

    for d in ("from_date", "to_date"):
        datetime.strptime(raw[d], "%Y-%m-%d")

    if not raw["institution_ids"]:
        raise ValueError("institution_ids must not be empty")
    if not raw["topics"]:
        raise ValueError("topics must not be empty")

    return TopicConfig(
        institution_ids=list(raw["institution_ids"]),
        from_date=str(raw["from_date"]),
        to_date=str(raw["to_date"]),
        max_pages=int(raw["max_pages"]),
        per_page=min(200, int(raw["per_page"])),
        topics=[str(t).strip() for t in raw["topics"] if str(t).strip()],
        results_dir=str(raw["results_dir"]),
        output_filtered_json=str(raw["output_filtered_json"]),
        output_relevant_jsonl=str(raw["output_relevant_jsonl"]),
        output_analysis_md=str(raw["output_analysis_md"]),
        cheap_model=str(raw["cheap_model"]),
        analysis_model=str(raw["analysis_model"]),
        openalex_api_key=str(raw.get("openalex_api_key", "")).strip(),
        prompt_config_path=str(raw["prompt_config_path"]),
    )


def load_prompt_config(path: str) -> PromptConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt config not found: {path}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    required = [
        "relevance_system_prompt",
        "relevance_user_prompt_template",
        "analysis_system_prompt",
        "analysis_user_prompt_template",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Missing prompt keys: {', '.join(missing)}")
    return PromptConfig(
        relevance_system_prompt=str(raw["relevance_system_prompt"]),
        relevance_user_prompt_template=str(raw["relevance_user_prompt_template"]),
        analysis_system_prompt=str(raw["analysis_system_prompt"]),
        analysis_user_prompt_template=str(raw["analysis_user_prompt_template"]),
    )


def fetch_candidate_works(cfg: TopicConfig, pb: ProgressBar) -> list[dict[str, Any]]:
    filters = [
        f"institutions.id:{'|'.join(cfg.institution_ids)}",
        f"from_publication_date:{cfg.from_date}",
        f"to_publication_date:{cfg.to_date}",
        "primary_location.source.type:journal|conference",
    ]
    cursor = "*"
    out: list[dict[str, Any]] = []
    pb.update("Fetching host publications", 0, cfg.max_pages)

    for page in range(1, cfg.max_pages + 1):
        params = {
            "filter": ",".join(filters),
            "per-page": cfg.per_page,
            "cursor": cursor,
            "select": (
                "id,display_name,publication_year,publication_date,"
                "authorships,concepts,abstract_inverted_index,primary_location"
            ),
        }
        data = safe_get(OPENALEX_WORKS_ENDPOINT, add_openalex_auth(params, cfg.openalex_api_key))
        batch = data.get("results", [])
        if not batch:
            pb.update("Fetching host publications", page, cfg.max_pages)
            break
        out.extend(batch)
        cursor = data.get("meta", {}).get("next_cursor")
        pb.update("Fetching host publications", page, cfg.max_pages)
        if not cursor:
            break
    return out


def keep_german_collaboration_subset(works: list[dict[str, Any]], host_ids: set[str], pb: ProgressBar) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    pb.update("Filtering works with German collaborators", 0, len(works))
    for idx, work in enumerate(works, start=1):
        has_de_partner = False
        for authorship in work.get("authorships", []):
            for inst in authorship.get("institutions", []):
                iid = inst.get("id", "")
                if iid and iid not in host_ids and inst.get("country_code") == "DE":
                    has_de_partner = True
                    break
            if has_de_partner:
                break
        if has_de_partner:
            kept.append(work)
        pb.update("Filtering works with German collaborators", idx, len(works))
    return kept


def llm_topic_decision(
    title: str,
    abstract: str,
    topics: list[str],
    model: str,
    api_key: str,
    prompt_cfg: PromptConfig,
) -> dict[str, Any]:
    user_prompt = prompt_cfg.relevance_user_prompt_template.format(
        topics_json=json.dumps(topics, ensure_ascii=False),
        title=title,
        abstract=abstract[:5000],
    )
    payload = {
        "model": model,
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


def keyword_fallback_decision(title: str, abstract: str, topics: list[str]) -> dict[str, Any]:
    text = f"{title} {abstract}".lower()
    matches = [t for t in topics if t.lower() in text]
    return {"relevant": bool(matches), "matched_topics": matches, "rationale": "Keyword fallback matching."}


def extract_metadata(work: dict[str, Any], host_ids: set[str]) -> dict[str, Any]:
    german_partners: set[str] = set()
    author_names: list[str] = []
    for a in work.get("authorships", []):
        author = a.get("author", {}).get("display_name")
        if author:
            author_names.append(author)
        for inst in a.get("institutions", []):
            iid = inst.get("id", "")
            if iid and iid not in host_ids and inst.get("country_code") == "DE":
                german_partners.add(inst.get("display_name", "Unknown institution"))

    return {
        "id": work.get("id"),
        "title": work.get("display_name"),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "source_type": (((work.get("primary_location") or {}).get("source") or {}).get("type")),
        "authors": author_names[:25],
        "german_collaborator_institutes": sorted(german_partners),
        "concepts": [c.get("display_name") for c in work.get("concepts", []) if c.get("display_name")][:20],
    }


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_topic_stats(relevant_rows: list[dict[str, Any]], topics: list[str]) -> dict[str, Any]:
    by_topic: Counter[str] = Counter()
    by_year_topic: dict[str, Counter[int]] = {t: Counter() for t in topics}

    for row in relevant_rows:
        year = row.get("publication_year")
        for t in row.get("matched_topics", []):
            by_topic[t] += 1
            if isinstance(year, int):
                by_year_topic.setdefault(t, Counter())[year] += 1

    return {
        "topic_counts": dict(by_topic),
        "topic_year_counts": {t: dict(sorted(c.items())) for t, c in by_year_topic.items()},
    }


def generate_topic_analysis(
    cfg: TopicConfig,
    prompt_cfg: PromptConfig,
    relevant_rows: list[dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        lines = ["# Topic-Focused Collaboration Analysis", ""]
        lines.append(f"- Relevant papers: {len(relevant_rows)}")
        lines.append("- Topic coverage:")
        for topic, count in sorted(stats["topic_counts"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {topic}: {count}")
        lines.append("")
        lines.append("## Interpretation")
        lines.append("The strongest collaboration direction is indicated by topics with the highest relevant-paper counts, while yearly patterns indicate where momentum is building or slowing.")
        return "\n".join(lines) + "\n"

    slim_rows = [
        {
            "title": r.get("title"),
            "year": r.get("publication_year"),
            "matched_topics": r.get("matched_topics", []),
            "german_collaborator_institutes": r.get("german_collaborator_institutes", [])[:5],
            "rationale": r.get("rationale", ""),
        }
        for r in relevant_rows[:300]
    ]

    user_prompt = prompt_cfg.analysis_user_prompt_template.format(
        topics_json=json.dumps(cfg.topics, ensure_ascii=False),
        relevant_count=len(relevant_rows),
        stats_json=json.dumps(stats, ensure_ascii=False),
        paper_sample_json=json.dumps(slim_rows, ensure_ascii=False),
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


def serialize_filtered_work(work: dict[str, Any], host_ids: set[str]) -> dict[str, Any]:
    entry = extract_metadata(work, host_ids)
    entry["abstract"] = invert_abstract(work.get("abstract_inverted_index"))
    return entry


def main() -> int:
    pb = ProgressBar()
    try:
        pb.update("Loading topic config", 0, 1)
        cfg = load_topic_config("config.json")
        pb.update("Loading topic config", 1, 1)
        print(f"Time period searched: {cfg.from_date} to {cfg.to_date}")

        pb.update("Loading prompt config", 0, 1)
        prompt_cfg = load_prompt_config(cfg.prompt_config_path)
        pb.update("Loading prompt config", 1, 1)

        host_ids = set(cfg.institution_ids)
        works = fetch_candidate_works(cfg, pb)
        de_subset = keep_german_collaboration_subset(works, host_ids, pb)

        api_key = os.getenv("OPENAI_API_KEY", "")
        relevant_rows: list[dict[str, Any]] = []
        pb.update("Classifying papers by topic relevance", 0, len(de_subset))
        for idx, work in enumerate(de_subset, start=1):
            title = work.get("display_name", "")
            abstract = invert_abstract(work.get("abstract_inverted_index"))

            if api_key:
                decision = llm_topic_decision(title, abstract, cfg.topics, cfg.cheap_model, api_key, prompt_cfg)
            else:
                decision = keyword_fallback_decision(title, abstract, cfg.topics)

            if bool(decision.get("relevant")):
                row = extract_metadata(work, host_ids)
                row["matched_topics"] = decision.get("matched_topics", [])
                row["rationale"] = decision.get("rationale", "")
                row["abstract"] = abstract
                relevant_rows.append(row)

            pb.update("Classifying papers by topic relevance", idx, len(de_subset))

        stats = build_topic_stats(relevant_rows, cfg.topics)

        pb.update("Generating topic analysis", 0, 1)
        analysis_md = generate_topic_analysis(cfg, prompt_cfg, relevant_rows, stats)
        pb.update("Generating topic analysis", 1, 1)

        pb.update("Saving outputs", 0, 1)
        out_dir = Path(cfg.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filtered_json_path = out_dir / cfg.output_filtered_json
        jsonl_path = out_dir / cfg.output_relevant_jsonl
        md_path = out_dir / cfg.output_analysis_md

        filtered_rows = [serialize_filtered_work(w, host_ids) for w in de_subset]
        filtered_json_path.write_text(json.dumps(filtered_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        save_jsonl(jsonl_path, relevant_rows)
        md_path.write_text(analysis_md, encoding="utf-8")
        (out_dir / "topic_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        pb.update("Saving outputs", 1, 1)

        print(f"Filtered dataset saved to: {filtered_json_path.resolve()}")
        print(f"Relevant papers saved to: {jsonl_path.resolve()}")
        print(f"Analysis saved to: {md_path.resolve()}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
