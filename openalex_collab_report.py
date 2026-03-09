#!/usr/bin/env python3
"""Generate a collaboration report from OpenAlex for selected host institutions.

Run directly from PyCharm (Run button).
Configure parameters in config.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
OPENALEX_AUTHORS_ENDPOINT = "https://api.openalex.org/authors"
OPENALEX_INSTITUTIONS_ENDPOINT = "https://api.openalex.org/institutions"
OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"


@dataclass
class AppConfig:
    institution_ids: list[str]
    from_date: str
    to_date: str
    max_pages: int
    per_page: int
    top_collaborators: int
    top_individuals: int
    output_report: str
    results_dir: str
    openai_model: str
    analysis_model: str
    openalex_api_key: str


class ConfigError(Exception):
    pass


class ProgressBar:
    def __init__(self, width: int = 28):
        self.width = width

    def update(self, label: str, current: int, total: int) -> None:
        total = max(total, 1)
        ratio = min(max(current / total, 0), 1)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        print(f"[{bar}] {int(ratio * 100):3d}% | {label}")


def load_config(path: str = "config.json") -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigError("config.json not found.")
    raw = json.loads(p.read_text(encoding="utf-8"))

    required = [
        "institution_ids",
        "from_date",
        "to_date",
        "max_pages",
        "per_page",
        "top_collaborators",
        "top_individuals",
        "output_report",
        "results_dir",
        "openai_model",
        "openalex_api_key",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ConfigError(f"Missing keys in config.json: {', '.join(missing)}")

    for date_key in ("from_date", "to_date"):
        datetime.strptime(raw[date_key], "%Y-%m-%d")

    if not raw["institution_ids"]:
        raise ConfigError("institution_ids cannot be empty")

    return AppConfig(
        institution_ids=list(raw["institution_ids"]),
        from_date=str(raw["from_date"]),
        to_date=str(raw["to_date"]),
        max_pages=int(raw["max_pages"]),
        per_page=min(200, int(raw["per_page"])),
        top_collaborators=int(raw["top_collaborators"]),
        top_individuals=int(raw["top_individuals"]),
        output_report=str(raw["output_report"]),
        results_dir=str(raw["results_dir"]),
        openai_model=str(raw["openai_model"]),
        analysis_model=str(raw.get("analysis_model", "gpt-4.1")),
        openalex_api_key=str(raw.get("openalex_api_key", "")).strip(),
    )


def safe_get(url: str, params: dict[str, Any], max_retries: int = 3) -> dict[str, Any]:
    full_url = f"{url}?{urlencode(params)}"
    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(full_url, timeout=45) as r:
                return json.loads(r.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            if attempt == max_retries:
                raise
            time.sleep(1.2 * attempt)
    return {}




def add_openalex_auth(params: dict[str, Any], api_key: str) -> dict[str, Any]:
    enriched = dict(params)
    if api_key:
        enriched["api_key"] = api_key
    return enriched
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
            time.sleep(1.2 * attempt)
    return {}


def invert_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    max_pos = -1
    for positions in inverted.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return ""
    tokens = [""] * (max_pos + 1)
    for token, positions in inverted.items():
        for pos in positions:
            if 0 <= pos < len(tokens):
                tokens[pos] = token
    return " ".join(t for t in tokens if t).strip()


def normalize_for_dedup(sentence: str) -> str:
    s = sentence.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s


def dedupe_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        clean = part.strip()
        if not clean:
            continue
        key = normalize_for_dedup(clean)
        if key and key not in seen:
            seen.add(key)
            kept.append(clean)
    return " ".join(kept).strip()


def parse_json_object_from_text(raw_text: str) -> dict[str, Any] | None:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def build_local_theme_summaries(abstracts_by_theme: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for theme, abstracts in abstracts_by_theme.items():
        if not abstracts:
            out[theme] = "No abstract text available for this theme in the selected period."
            continue
        merged = " ".join(abstracts[:3])
        sentences = re.split(r"(?<=[.!?])\s+", merged)
        picked = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        summary = " ".join(picked) if picked else merged[:320]
        out[theme] = dedupe_sentences(summary)
    return out


def fetch_institution_name(inst_id: str, openalex_api_key: str) -> str:
    short_id = inst_id.replace("https://openalex.org/", "")
    data = safe_get(f"{OPENALEX_INSTITUTIONS_ENDPOINT}/{short_id}", add_openalex_auth({"select": "display_name"}, openalex_api_key))
    return data.get("display_name", short_id)


def fetch_host_institutions(inst_ids: list[str], openalex_api_key: str, pb: ProgressBar) -> dict[str, str]:
    out: dict[str, str] = {}
    total = len(inst_ids)
    pb.update("Resolving host institute names", 0, total)
    for idx, iid in enumerate(inst_ids, start=1):
        try:
            out[iid] = fetch_institution_name(iid, openalex_api_key)
        except Exception:
            out[iid] = iid.replace("https://openalex.org/", "")
        pb.update("Resolving host institute names", idx, total)
    return out


def fetch_works(config: AppConfig, pb: ProgressBar) -> list[dict[str, Any]]:
    filters = [
        f"institutions.id:{'|'.join(config.institution_ids)}",
        f"from_publication_date:{config.from_date}",
        f"to_publication_date:{config.to_date}",
        "primary_location.source.type:journal|conference",
    ]
    cursor = "*"
    works: list[dict[str, Any]] = []
    pb.update("Fetching publications", 0, config.max_pages)

    for page in range(1, config.max_pages + 1):
        params = {
            "filter": ",".join(filters),
            "per-page": config.per_page,
            "cursor": cursor,
            "select": (
                "id,display_name,publication_year,publication_date,"
                "authorships,concepts,primary_location,abstract_inverted_index"
            ),
        }
        data = safe_get(OPENALEX_WORKS_ENDPOINT, add_openalex_auth(params, config.openalex_api_key))
        batch = data.get("results", [])
        if not batch:
            pb.update("Fetching publications", page, config.max_pages)
            break
        works.extend(batch)

        cursor = data.get("meta", {}).get("next_cursor")
        pb.update("Fetching publications", page, config.max_pages)
        if not cursor:
            break

    return works


def build_collaboration_stats(works: list[dict[str, Any]], host_ids: set[str], top_n: int, pb: ProgressBar) -> list[tuple[str, int, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    total = len(works)
    pb.update("Computing institute collaborations", 0, total)

    for idx, work in enumerate(works, start=1):
        per_work: set[tuple[str, str]] = set()
        for authorship in work.get("authorships", []):
            for inst in authorship.get("institutions", []):
                iid = inst.get("id", "")
                if not iid or iid in host_ids:
                    continue
                if inst.get("country_code") == "DE":
                    per_work.add((iid, inst.get("display_name", "Unknown institution")))
        for x in per_work:
            counter[x] += 1
        pb.update("Computing institute collaborations", idx, total)

    return [(name, c, iid) for (iid, name), c in counter.most_common(top_n)]


def fetch_author(author_id: str, openalex_api_key: str) -> dict[str, Any] | None:
    if not author_id:
        return None
    short_id = author_id.replace("https://openalex.org/", "")
    try:
        return safe_get(
            f"{OPENALEX_AUTHORS_ENDPOINT}/{short_id}",
            add_openalex_auth({"select": "display_name,works_count,cited_by_count,last_known_institutions,x_concepts"}, openalex_api_key),
        )
    except Exception:
        return None


def build_top_individuals(works: list[dict[str, Any]], host_ids: set[str], top_n: int, openalex_api_key: str, pb: ProgressBar) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    total = len(works)
    pb.update("Identifying collaborator individuals", 0, total)

    for idx, work in enumerate(works, start=1):
        per_work: set[tuple[str, str]] = set()
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            aid = author.get("id", "")
            aname = author.get("display_name", "Unknown")

            is_host = any(i.get("id") in host_ids for i in authorship.get("institutions", []))
            is_de = any(i.get("country_code") == "DE" for i in authorship.get("institutions", []))
            if aid and is_de and not is_host:
                per_work.add((aid, aname))

        for p in per_work:
            counter[p] += 1
        pb.update("Identifying collaborator individuals", idx, total)

    top = counter.most_common(top_n)
    result: list[dict[str, Any]] = []
    pb.update("Enriching collaborator bios", 0, len(top))
    for idx, ((aid, name), papers) in enumerate(top, start=1):
        details = fetch_author(aid, openalex_api_key) or {}
        inst_names = ", ".join(
            x.get("display_name", "")
            for x in (details.get("last_known_institutions") or [])[:2]
            if x.get("display_name")
        )
        bg_terms = [x.get("display_name", "") for x in (details.get("x_concepts") or [])[:3] if x.get("display_name")]
        bg_text = ", ".join(bg_terms) if bg_terms else "interdisciplinary collaboration"
        bio = dedupe_sentences(
            f"{name} works mainly on {bg_text}; has {details.get('works_count', 'N/A')} indexed works "
            f"and {details.get('cited_by_count', 'N/A')} citations"
            + (f"; recent affiliation: {inst_names}." if inst_names else ".")
        )

        result.append(
            {
                "name": name,
                "coauthored_papers": papers,
                "works_count": details.get("works_count", "N/A"),
                "cited_by_count": details.get("cited_by_count", "N/A"),
                "research_background": bg_text,
                "bio": bio,
            }
        )
        pb.update("Enriching collaborator bios", idx, len(top))

    return result


def summarize_themes(works: list[dict[str, Any]], top_k: int = 12) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for w in works:
        for c in w.get("concepts", []):
            if c.get("display_name") and c.get("score", 0) >= 0.4:
                counter[c["display_name"]] += 1
    return counter.most_common(top_k)


def yearly_changes(works: list[dict[str, Any]], host_ids: set[str]) -> list[dict[str, Any]]:
    by_year_publications: Counter[int] = Counter()
    by_year_de_inst: dict[int, set[str]] = defaultdict(set)

    for w in works:
        year = w.get("publication_year")
        if not isinstance(year, int):
            continue
        by_year_publications[year] += 1
        for au in w.get("authorships", []):
            for inst in au.get("institutions", []):
                iid = inst.get("id")
                if iid and iid not in host_ids and inst.get("country_code") == "DE":
                    by_year_de_inst[year].add(iid)

    rows = []
    for year in sorted(by_year_publications):
        rows.append(
            {
                "year": year,
                "publications": by_year_publications[year],
                "unique_german_collaborator_institutes": len(by_year_de_inst.get(year, set())),
            }
        )
    return rows


def summarize_theme_abstracts(
    config: AppConfig,
    works: list[dict[str, Any]],
    themes: list[tuple[str, int]],
) -> dict[str, str]:
    top_theme_names = [t for t, _ in themes[:6]]
    abstracts_by_theme: dict[str, list[str]] = {t: [] for t in top_theme_names}

    for w in works:
        abstract = invert_abstract(w.get("abstract_inverted_index"))
        if not abstract:
            continue
        lowered = abstract.lower()
        for theme in top_theme_names:
            if theme.lower() in lowered and len(abstracts_by_theme[theme]) < 8:
                abstracts_by_theme[theme].append(abstract[:1500])

    local_fallback = build_local_theme_summaries(abstracts_by_theme)

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return local_fallback

    payload = {
        "model": config.analysis_model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You summarize collaboration themes in concise, smooth, non-repetitive language. "
                    "Never repeat the same fact in different wording."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return only JSON mapping each theme to a 2-3 sentence summary grounded in these abstracts.\n"
                    f"Themes: {top_theme_names}\n"
                    + json.dumps(abstracts_by_theme, ensure_ascii=False)
                ),
            },
        ],
    }

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        resp = safe_post_json(OPENAI_CHAT_ENDPOINT, payload, headers)
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = parse_json_object_from_text(content)
        if not parsed:
            return local_fallback

        out: dict[str, str] = {}
        for theme in top_theme_names:
            candidate = str(parsed.get(theme, local_fallback.get(theme, ""))).strip()
            out[theme] = dedupe_sentences(candidate) if candidate else local_fallback.get(theme, "")
        return out
    except Exception:
        return local_fallback


def build_top_collaborator_abstract_bundles(
    works: list[dict[str, Any]],
    top_institutes: list[tuple[str, int, str]],
) -> list[dict[str, Any]]:
    top_ids = {iid for _, _, iid in top_institutes}
    by_inst: dict[str, dict[str, Any]] = {
        iid: {"name": name, "id": iid, "collaboration_count": count, "abstracts": [], "concepts": Counter()}
        for name, count, iid in top_institutes
    }

    for work in works:
        abstract = invert_abstract(work.get("abstract_inverted_index"))
        work_inst_ids: set[str] = set()
        for authorship in work.get("authorships", []):
            for inst in authorship.get("institutions", []):
                iid = inst.get("id", "")
                if iid in top_ids:
                    work_inst_ids.add(iid)

        if not work_inst_ids:
            continue

        for iid in work_inst_ids:
            entry = by_inst[iid]
            if abstract and len(entry["abstracts"]) < 18:
                entry["abstracts"].append(abstract[:2000])
            for concept in work.get("concepts", []):
                cname = concept.get("display_name")
                cscore = concept.get("score", 0)
                if cname and cscore >= 0.4:
                    entry["concepts"][cname] += 1

    output: list[dict[str, Any]] = []
    for name, count, iid in top_institutes:
        entry = by_inst.get(iid, {"name": name, "id": iid, "collaboration_count": count, "abstracts": [], "concepts": Counter()})
        output.append(
            {
                "name": entry["name"],
                "id": entry["id"],
                "collaboration_count": entry["collaboration_count"],
                "top_concepts": [k for k, _ in entry["concepts"].most_common(8)],
                "abstracts": entry["abstracts"],
            }
        )
    return output


def build_local_collaborator_direction_summary(collab_bundles: list[dict[str, Any]]) -> str:
    lines = ["## Collaboration Directions by Top German Partners", ""]
    for idx, item in enumerate(collab_bundles, start=1):
        concept_line = ", ".join(item.get("top_concepts", [])[:5]) or "No strong concept signal"
        abstract_sample = ""
        if item.get("abstracts"):
            pieces = re.split(r"(?<=[.!?])\s+", item["abstracts"][0])
            abstract_sample = dedupe_sentences(" ".join(p for p in pieces[:2] if p.strip()))
        lines.append(f"### {idx}. {item['name']}")
        lines.append(
            f"Collaboration appears to center on: {concept_line}. "
            f"Representative collaboration focus from abstracts: {abstract_sample or 'No abstract detail available.'}"
        )
        lines.append("")
    return "\n".join(lines).strip()


def generate_top_collaborator_theme_analysis(config: AppConfig, collab_bundles: list[dict[str, Any]]) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return build_local_collaborator_direction_summary(collab_bundles)

    payload = {
        "model": config.analysis_model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a research strategy analyst. Focus on collaboration patterns inferred from abstracts. "
                    "Do not explain general fields. Do not repeat facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write a markdown section titled 'Collaboration Directions by Top German Partners'. "
                    "For each partner, write 2-4 sentences on the important collaboration directions derived from abstracts. "
                    "Explain what the host-partner collaborations are trying to solve or build, not generic field descriptions. "
                    "Avoid repeating the same information across partners.\n\n"
                    f"Data:\n{json.dumps(collab_bundles, ensure_ascii=False)}"
                ),
            },
        ],
    }

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        resp = safe_post_json(OPENAI_CHAT_ENDPOINT, payload, headers)
        raw = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        return dedupe_sentences(raw) if raw else build_local_collaborator_direction_summary(collab_bundles)
    except Exception:
        return build_local_collaborator_direction_summary(collab_bundles)


def generate_gpt_analysis(
    config: AppConfig,
    works: list[dict[str, Any]],
    host_names: dict[str, str],
    top_institutes: list[tuple[str, int, str]],
    themes: list[tuple[str, int]],
    yearly: list[dict[str, Any]],
) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "## Narrative Analysis\n\nOPENAI_API_KEY is not set, so advanced narrative generation is skipped."

    payload = {
        "model": config.analysis_model,
        "temperature": 0.25,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Write smooth, concise markdown text for research leaders. "
                    "Strictly avoid repeating the same fact."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write short sections with headings:\n"
                    "1) Narrative Analysis\n"
                    "2) Potential Future Research Collaborations\n"
                    "Mention host institutes, collaboration intensity, yearly changes, and practical next steps.\n"
                    f"Host institutes: {list(host_names.values())}\n"
                    f"Publications: {len(works)}\n"
                    f"Top German collaborators: {top_institutes}\n"
                    f"Top themes: {themes}\n"
                    f"Yearly changes: {yearly}"
                ),
            },
        ],
    }

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        resp = safe_post_json(OPENAI_CHAT_ENDPOINT, payload, headers)
        raw = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        text = dedupe_sentences(raw) if raw else ""
        return text or "## Narrative Analysis\n\nGPT returned empty analysis content."
    except Exception:
        return "## Narrative Analysis\n\nGPT narrative could not be generated due to API/network limits."


def format_report(
    config: AppConfig,
    host_names: dict[str, str],
    works: list[dict[str, Any]],
    top_institutes: list[tuple[str, int, str]],
    partner_direction_text: str,
    themes: list[tuple[str, int]],
    theme_summaries: dict[str, str],
    top_people: list[dict[str, Any]],
    yearly: list[dict[str, Any]],
    gpt_text: str,
) -> str:
    lines = [
        "# OpenAlex Collaboration Report",
        "",
        f"- **Period:** {config.from_date} to {config.to_date}",
        "- **Host institutes:** " + "; ".join(host_names.values()),
        f"- **Host institute IDs:** {', '.join(config.institution_ids)}",
        "- **Filtered publication types:** Journals and conferences",
        f"- **Total publications retrieved:** {len(works)}",
        "",
        "## Top 10 Collaborating Institutes in Germany",
        "",
        "| Rank | Institute | # Co-authored Publications | OpenAlex ID |",
        "|---:|---|---:|---|",
    ]
    for idx, (name, count, iid) in enumerate(top_institutes, start=1):
        lines.append(f"| {idx} | {name} | {count} | {iid} |")

    lines.extend(["", partner_direction_text.strip(), ""])

    lines.extend(["## Changes Over the Years", "", "| Year | # Publications | # Unique German Collaborator Institutes |", "|---:|---:|---:|"])
    for row in yearly:
        lines.append(f"| {row['year']} | {row['publications']} | {row['unique_german_collaborator_institutes']} |")

    lines.extend(["", "## Main Research Themes", ""])
    for theme, cnt in themes:
        lines.append(f"- **{theme}** ({cnt} publications)")

    lines.extend(["", "## Theme Summaries from Abstracts", ""])
    for theme, _ in themes[:6]:
        lines.append(f"### {theme}")
        lines.append(dedupe_sentences(theme_summaries.get(theme, "No summary available.")))
        lines.append("")

    lines.extend([
        "## Main Collaborator Individuals",
        "",
        "| Name | Co-authored papers | Works | Citations | Research background | Short bio |",
        "|---|---:|---:|---:|---|---|",
    ])
    for p in top_people:
        bio = dedupe_sentences(p["bio"]).replace("|", "\\|")
        rb = str(p["research_background"]).replace("|", "\\|")
        lines.append(
            f"| {p['name']} | {p['coauthored_papers']} | {p['works_count']} | {p['cited_by_count']} | {rb} | {bio} |"
        )

    lines.extend(["", gpt_text.strip(), ""])
    lines.append(textwrap.dedent("""
    ---
    _Notes:_
    - Data is pulled live from OpenAlex and may differ between runs.
    - Collaborator institutes are counted once per publication.
    - Direction analysis is inferred from abstract text in publications co-authored with each top collaborator.
    """).strip())
    return "\n".join(lines) + "\n"


def ensure_result_path(config: AppConfig) -> Path:
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / config.output_report


def main() -> int:
    pb = ProgressBar()
    try:
        pb.update("Loading config", 0, 1)
        config = load_config("config.json")
        pb.update("Loading config", 1, 1)
        print(f"Time period searched: {config.from_date} to {config.to_date}")

        host_ids = set(config.institution_ids)
        host_names = fetch_host_institutions(config.institution_ids, config.openalex_api_key, pb)
        works = fetch_works(config, pb)
        top_institutes = build_collaboration_stats(works, host_ids, config.top_collaborators, pb)
        themes = summarize_themes(works)
        top_people = build_top_individuals(works, host_ids, config.top_individuals, config.openalex_api_key, pb)
        yearly = yearly_changes(works, host_ids)

        pb.update("Building collaborator abstract bundles", 0, 1)
        collab_bundles = build_top_collaborator_abstract_bundles(works, top_institutes)
        pb.update("Building collaborator abstract bundles", 1, 1)

        pb.update("Analyzing collaboration directions", 0, 1)
        partner_direction_text = generate_top_collaborator_theme_analysis(config, collab_bundles)
        pb.update("Analyzing collaboration directions", 1, 1)

        pb.update("Summarizing themes from abstracts", 0, 1)
        theme_summaries = summarize_theme_abstracts(config, works, themes)
        pb.update("Summarizing themes from abstracts", 1, 1)

        pb.update("Generating narrative", 0, 1)
        gpt_text = generate_gpt_analysis(config, works, host_names, top_institutes, themes, yearly)
        pb.update("Generating narrative", 1, 1)

        pb.update("Saving report", 0, 1)
        out_path = ensure_result_path(config)
        report = format_report(
            config,
            host_names,
            works,
            top_institutes,
            partner_direction_text,
            themes,
            theme_summaries,
            top_people,
            yearly,
            gpt_text,
        )
        out_path.write_text(report, encoding="utf-8")
        pb.update("Saving report", 1, 1)

        print(f"Report generated: {out_path.resolve()}")
        return 0
    except ConfigError as e:
        print(f"Configuration error: {e}")
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
