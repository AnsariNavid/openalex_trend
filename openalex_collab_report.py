#!/usr/bin/env python3
"""Generate a short collaboration report from OpenAlex for selected institutions.

Run directly from PyCharm using the Run button.
Configure behavior in `config.json`.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
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
OPENALEX_AUTHORS_ENDPOINT = "https://api.openalex.org/authors"
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
    openai_model: str


class ConfigError(Exception):
    pass


def load_config(config_path: str = "config.json") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "institution_ids",
        "from_date",
        "to_date",
        "max_pages",
        "per_page",
        "top_collaborators",
        "top_individuals",
        "output_report",
        "openai_model",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ConfigError(f"Missing keys in config.json: {', '.join(missing)}")

    if not raw["institution_ids"]:
        raise ConfigError("institution_ids must not be empty")

    for key in ("from_date", "to_date"):
        datetime.strptime(raw[key], "%Y-%m-%d")

    return AppConfig(
        institution_ids=list(raw["institution_ids"]),
        from_date=str(raw["from_date"]),
        to_date=str(raw["to_date"]),
        max_pages=int(raw["max_pages"]),
        per_page=min(200, int(raw["per_page"])),
        top_collaborators=int(raw["top_collaborators"]),
        top_individuals=int(raw["top_individuals"]),
        output_report=str(raw["output_report"]),
        openai_model=str(raw["openai_model"]),
    )


def safe_get(url: str, params: dict[str, Any], max_retries: int = 3) -> dict[str, Any]:
    query = urlencode(params)
    full_url = f"{url}?{query}"
    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(full_url, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            if attempt == max_retries:
                raise
            time.sleep(1.2 * attempt)
    return {}


def safe_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], max_retries: int = 3) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url=url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)

    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            if attempt == max_retries:
                raise
            time.sleep(1.2 * attempt)
    return {}


def fetch_works(config: AppConfig) -> list[dict[str, Any]]:
    filters = [
        f"institutions.id:{'|'.join(config.institution_ids)}",
        f"from_publication_date:{config.from_date}",
        f"to_publication_date:{config.to_date}",
        "primary_location.source.type:journal|conference",
    ]
    cursor = "*"
    works: list[dict[str, Any]] = []

    for _ in range(config.max_pages):
        params = {
            "filter": ",".join(filters),
            "per-page": config.per_page,
            "cursor": cursor,
            "select": "id,display_name,publication_year,publication_date,authorships,concepts,primary_location",
        }
        data = safe_get(OPENALEX_WORKS_ENDPOINT, params)
        batch = data.get("results", [])
        if not batch:
            break
        works.extend(batch)

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    return works


def build_collaboration_stats(works: list[dict[str, Any]], institution_ids: set[str], top_n: int) -> list[tuple[str, int, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for work in works:
        per_work: set[tuple[str, str]] = set()
        for authorship in work.get("authorships", []):
            for inst in authorship.get("institutions", []):
                iid = inst.get("id", "")
                if not iid or iid in institution_ids:
                    continue
                if inst.get("country_code") == "DE":
                    per_work.add((iid, inst.get("display_name", "Unknown institution")))
        for key in per_work:
            counter[key] += 1

    top = counter.most_common(top_n)
    return [(name, count, iid) for (iid, name), count in top]


def fetch_author(author_id: str) -> dict[str, Any] | None:
    if not author_id:
        return None
    aid = author_id.replace("https://openalex.org/", "")
    try:
        return safe_get(
            f"{OPENALEX_AUTHORS_ENDPOINT}/{aid}",
            {"select": "display_name,works_count,cited_by_count,last_known_institutions,orcid"},
        )
    except Exception:
        return None


def build_top_individuals(works: list[dict[str, Any]], institution_ids: set[str], top_n: int) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()

    for work in works:
        per_work: set[tuple[str, str]] = set()
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            author_id = author.get("id", "")
            author_name = author.get("display_name", "Unknown")

            is_target_author = any(inst.get("id") in institution_ids for inst in authorship.get("institutions", []))
            is_german_collaborator = any(inst.get("country_code") == "DE" for inst in authorship.get("institutions", []))

            if author_id and is_german_collaborator and not is_target_author:
                per_work.add((author_id, author_name))

        for k in per_work:
            counter[k] += 1

    people = []
    for (author_id, author_name), paper_count in counter.most_common(top_n):
        details = fetch_author(author_id) or {}
        inst_names = ", ".join(
            i.get("display_name", "") for i in (details.get("last_known_institutions") or [])[:2] if i.get("display_name")
        )
        bio = (
            f"{author_name} has {details.get('works_count', 'N/A')} indexed works and "
            f"{details.get('cited_by_count', 'N/A')} citations"
            + (f"; recent affiliation: {inst_names}." if inst_names else ".")
        )

        people.append(
            {
                "name": author_name,
                "coauthored_papers": paper_count,
                "works_count": details.get("works_count", "N/A"),
                "cited_by_count": details.get("cited_by_count", "N/A"),
                "bio": bio,
            }
        )

    return people


def summarize_themes(works: list[dict[str, Any]], top_k: int = 12) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for w in works:
        for c in w.get("concepts", []):
            if c.get("display_name") and c.get("score", 0) >= 0.4:
                counter[c["display_name"]] += 1
    return counter.most_common(top_k)


def generate_gpt_analysis(config: AppConfig, works: list[dict[str, Any]], top_institutes: list[tuple[str, int, str]], themes: list[tuple[str, int]], top_people: list[dict[str, Any]]) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "## Narrative Analysis\n\nOPENAI_API_KEY is not set; GPT section skipped."

    payload = {
        "model": config.openai_model,
        "temperature": 0.3,
        "messages": [
            {
                "role": "system",
                "content": "You write concise and smooth research intelligence summaries in Markdown.",
            },
            {
                "role": "user",
                "content": (
                    "Write a short report with markdown headings: 'Narrative Analysis' and "
                    "'Potential Future Research Collaborations'. Keep it easy to follow and include key details.\n"
                    f"Period: {config.from_date} to {config.to_date}\n"
                    f"Publications: {len(works)}\n"
                    f"Top German institutes: {top_institutes}\n"
                    f"Themes: {themes}\n"
                    f"Top individuals: {[{'name': p['name'], 'coauthored_papers': p['coauthored_papers']} for p in top_people]}"
                ),
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    try:
        response = safe_post_json(OPENAI_CHAT_ENDPOINT, payload, headers)
        return response["choices"][0]["message"]["content"]
    except Exception:
        return "## Narrative Analysis\n\nGPT analysis failed due to API/network issue."


def format_report(config: AppConfig, works: list[dict[str, Any]], top_institutes: list[tuple[str, int, str]], themes: list[tuple[str, int]], top_people: list[dict[str, Any]], gpt_text: str) -> str:
    lines = [
        "# OpenAlex Collaboration Report",
        "",
        f"- **Period:** {config.from_date} to {config.to_date}",
        f"- **Input institution IDs:** {', '.join(config.institution_ids)}",
        "- **Filtered publication types:** Journals and conferences",
        f"- **Total publications retrieved:** {len(works)}",
        "",
        "## Top 10 Collaborating Institutes in Germany",
        "",
        "| Rank | Institute | # Co-authored Publications | OpenAlex ID |",
        "|---:|---|---:|---|",
    ]

    for i, (name, count, inst_id) in enumerate(top_institutes, start=1):
        lines.append(f"| {i} | {name} | {count} | {inst_id} |")

    lines.extend(["", "## Main Collaboration Themes", ""])
    for theme, count in themes:
        lines.append(f"- **{theme}** ({count} publications)")

    lines.extend([
        "",
        "## Main Collaborator Individuals",
        "",
        "| Name | Co-authored papers | Works | Citations | Short bio |",
        "|---|---:|---:|---:|---|",
    ])
    for person in top_people:
        bio = person["bio"].replace("|", "\\|")
        lines.append(
            f"| {person['name']} | {person['coauthored_papers']} | {person['works_count']} | {person['cited_by_count']} | {bio} |"
        )

    lines.extend(["", gpt_text.strip(), ""])
    lines.append(textwrap.dedent("""
    ---
    _Notes:_
    - OpenAlex data is live and may change between runs.
    - German collaborators are counted once per institute per publication.
    """).strip())

    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        config = load_config("config.json")
        target_set = set(config.institution_ids)
        works = fetch_works(config)
        top_institutes = build_collaboration_stats(works, target_set, config.top_collaborators)
        themes = summarize_themes(works)
        top_people = build_top_individuals(works, target_set, config.top_individuals)
        gpt = generate_gpt_analysis(config, works, top_institutes, themes, top_people)

        report = format_report(config, works, top_institutes, themes, top_people, gpt)
        Path(config.output_report).write_text(report, encoding="utf-8")
        print(f"Report generated: {Path(config.output_report).resolve()}")
        return 0
    except ConfigError as e:
        print(f"Configuration error: {e}")
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
