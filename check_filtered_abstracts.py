#!/usr/bin/env python3
"""Check abstract coverage and attempt abstract recovery from PDF links.

Recovery policy:
1) Try exact abstract extraction from PDF text (best effort).
2) If exact extraction fails, use GPT to generate an abstract-like summary from PDF text.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def download_pdf_bytes(pdf_url: str, timeout_s: int = 45) -> bytes | None:
    if not pdf_url:
        return None
    try:
        req = Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout_s) as r:
            content_type = str(r.headers.get("Content-Type", "")).lower()
            data = r.read()
            if "pdf" in content_type or data[:4] == b"%PDF":
                return data
            return None
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None


def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 4) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ""

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        chunks: list[str] = []
        for page in reader.pages[:max_pages]:
            chunks.append(page.extract_text() or "")
        return "\n".join(chunks).strip()
    except Exception:
        return ""


def find_exact_abstract_from_text(text: str) -> str:
    if not text:
        return ""
    clean = re.sub(r"\s+", " ", text).strip()
    patterns = [
        r"(?i)abstract\s*[:\-]?\s*(.{200,3000}?)(?:\bkeywords\b|\bintroduction\b|\b1\.\b)",
        r"(?i)summary\s*[:\-]?\s*(.{200,3000}?)(?:\bkeywords\b|\bintroduction\b|\b1\.\b)",
    ]
    for pat in patterns:
        m = re.search(pat, clean)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) >= 120:
                return candidate
    return ""


def gpt_generate_summary(title: str, paper_text: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    if not api_key or not paper_text:
        return ""
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": "You generate concise, factual abstract-style summaries from paper text.",
            },
            {
                "role": "user",
                "content": (
                    "Write an abstract-style summary (120-220 words) based only on the paper text below. "
                    "If text is insufficient, return exactly: INSUFFICIENT_TEXT\n\n"
                    f"Title: {title}\n\nPaper text:\n{paper_text[:12000]}"
                ),
            },
        ],
    }
    req = Request(
        OPENAI_CHAT_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8"))
        content = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        if content == "INSUFFICIENT_TEXT":
            return ""
        return content
    except Exception:
        return ""


def main() -> int:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/filtered_german_collab_dataset.json")
    report_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/abstract_coverage_report.json")
    enriched_path = Path(sys.argv[3]) if len(sys.argv) > 3 else input_path

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    rows = load_rows(input_path)
    total = len(rows)

    exact_recovered = 0
    gpt_generated = 0
    still_missing = 0

    api_key = os.getenv("OPENAI_API_KEY", "")

    for idx, r in enumerate(rows, start=1):
        abstract = (r.get("abstract") or "").strip()
        if abstract:
            continue

        pdf_url = str(r.get("pdf_url") or "").strip()
        recovered = ""
        method = ""

        if pdf_url:
            pdf_bytes = download_pdf_bytes(pdf_url)
            if pdf_bytes:
                pdf_text = extract_text_from_pdf(pdf_bytes)
                recovered = find_exact_abstract_from_text(pdf_text)
                if recovered:
                    method = "exact_from_pdf"
                elif pdf_text and api_key:
                    recovered = gpt_generate_summary(str(r.get("title") or ""), pdf_text, api_key)
                    if recovered:
                        method = "gpt_generated_from_pdf"

        if recovered:
            r["abstract"] = recovered
            r["abstract_recovery_method"] = method
            if method == "exact_from_pdf":
                exact_recovered += 1
            elif method == "gpt_generated_from_pdf":
                gpt_generated += 1
        else:
            r["abstract_recovery_method"] = "not_recovered"
            still_missing += 1

        if idx % 10 == 0:
            print(f"Processed missing-abstract recovery for {idx}/{total} entries...")

    with_abs = sum(1 for r in rows if (r.get("abstract") or "").strip())
    missing_entries = [{"id": r.get("id"), "title": r.get("title")} for r in rows if not (r.get("abstract") or "").strip()]

    report = {
        "input_file": str(input_path),
        "enriched_output_file": str(enriched_path),
        "total_entries": total,
        "entries_with_abstract": with_abs,
        "entries_missing_abstract": total - with_abs,
        "coverage_ratio": (with_abs / total) if total else 0,
        "exact_abstract_recovered_from_pdf": exact_recovered,
        "abstract_generated_by_gpt": gpt_generated,
        "still_missing_after_recovery": still_missing,
        "missing_entries": missing_entries,
    }

    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_rows(enriched_path, rows)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Abstract coverage report saved: {report_path.resolve()}")
    print(f"Enriched dataset saved: {enriched_path.resolve()}")
    print(f"Coverage: {with_abs}/{total} ({report['coverage_ratio']:.2%})")
    print(f"Recovered exact abstracts from PDF: {exact_recovered}")
    print(f"Generated abstracts with GPT: {gpt_generated}")
    print(f"Still missing abstracts: {still_missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
