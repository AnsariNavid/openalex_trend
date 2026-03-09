#!/usr/bin/env python3
"""Check abstract coverage and attempt abstract recovery from HTML/PDF links.

Recovery order:
1) Try exact abstract extraction from HTML text (best effort).
2) If not found, try exact abstract extraction from PDF text (best effort).
3) If neither works, use GPT to generate an abstract-style summary from paper text.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
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


def download_url_bytes(url: str, timeout_s: int = 45) -> tuple[bytes | None, str]:
    if not url:
        return None, ""
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout_s) as r:
            return r.read(), str(r.headers.get("Content-Type", "")).lower()
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None, ""


def download_pdf_bytes(pdf_url: str, timeout_s: int = 45) -> bytes | None:
    data, content_type = download_url_bytes(pdf_url, timeout_s)
    if not data:
        return None
    if "pdf" in content_type or data[:4] == b"%PDF":
        return data
    return None


def download_html_text(html_url: str, timeout_s: int = 45) -> str:
    data, content_type = download_url_bytes(html_url, timeout_s)
    if not data:
        return ""
    if "html" not in content_type and b"<html" not in data[:2000].lower():
        return ""
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def clean_html_to_text(html: str) -> str:
    if not html:
        return ""
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", html)
    return text.strip()


def find_exact_abstract_from_html(html: str) -> str:
    if not html:
        return ""

    meta_patterns = [
        r'(?is)<meta\s+name=["\']citation_abstract["\']\s+content=["\'](.*?)["\']',
        r'(?is)<meta\s+name=["\']dc\.description["\']\s+content=["\'](.*?)["\']',
        r'(?is)<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']',
        r'(?is)<meta\s+property=["\']og:description["\']\s+content=["\'](.*?)["\']',
    ]
    for pat in meta_patterns:
        m = re.search(pat, html)
        if m:
            candidate = re.sub(r"\s+", " ", m.group(1)).strip()
            if len(candidate) >= 120:
                return candidate

    text = clean_html_to_text(html)
    return find_exact_abstract_from_text(text)


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

    initial_with_abs = sum(1 for r in rows if (r.get("abstract") or "").strip())
    initial_missing = total - initial_with_abs

    print(f"Initial abstract availability: {initial_with_abs}/{total} ({(initial_with_abs/total if total else 0):.2%})")
    print(f"Initial missing abstracts: {initial_missing}")

    exact_html_recovered = 0
    exact_pdf_recovered = 0
    gpt_generated = 0

    api_key = os.getenv("OPENAI_API_KEY", "")

    for idx, r in enumerate(rows, start=1):
        if (r.get("abstract") or "").strip():
            continue

        html_url = str(r.get("html_url") or "").strip()
        pdf_url = str(r.get("pdf_url") or "").strip()

        recovered = ""
        method = ""
        combined_text = ""

        # 1) exact from HTML
        if html_url:
            html = download_html_text(html_url)
            if html:
                recovered = find_exact_abstract_from_html(html)
                combined_text += "\n" + clean_html_to_text(html)[:12000]
                if recovered:
                    method = "exact_from_html"

        # 2) exact from PDF
        if not recovered and pdf_url:
            pdf_bytes = download_pdf_bytes(pdf_url)
            if pdf_bytes:
                pdf_text = extract_text_from_pdf(pdf_bytes)
                combined_text += "\n" + pdf_text[:12000]
                recovered = find_exact_abstract_from_text(pdf_text)
                if recovered:
                    method = "exact_from_pdf"

        # 3) GPT generated from available paper text
        if not recovered and api_key and combined_text.strip():
            recovered = gpt_generate_summary(str(r.get("title") or ""), combined_text, api_key)
            if recovered:
                method = "gpt_generated_from_paper"

        if recovered:
            r["abstract"] = recovered
            r["abstract_recovery_method"] = method
            if method == "exact_from_html":
                exact_html_recovered += 1
            elif method == "exact_from_pdf":
                exact_pdf_recovered += 1
            elif method == "gpt_generated_from_paper":
                gpt_generated += 1
        else:
            r["abstract_recovery_method"] = "not_recovered"

        if idx % 10 == 0:
            print(f"Processed missing-abstract recovery for {idx}/{total} entries...")

    final_with_abs = sum(1 for r in rows if (r.get("abstract") or "").strip())
    final_missing = total - final_with_abs
    missing_entries = [{"id": r.get("id"), "title": r.get("title")} for r in rows if not (r.get("abstract") or "").strip()]

    report = {
        "input_file": str(input_path),
        "enriched_output_file": str(enriched_path),
        "total_entries": total,
        "initial_entries_with_abstract": initial_with_abs,
        "initial_entries_missing_abstract": initial_missing,
        "final_entries_with_abstract": final_with_abs,
        "final_entries_missing_abstract": final_missing,
        "final_coverage_ratio": (final_with_abs / total) if total else 0,
        "exact_abstract_recovered_from_html": exact_html_recovered,
        "exact_abstract_recovered_from_pdf": exact_pdf_recovered,
        "abstract_generated_by_gpt": gpt_generated,
        "still_missing_after_recovery": final_missing,
        "missing_entries": missing_entries,
    }

    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_rows(enriched_path, rows)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Abstract coverage report saved: {report_path.resolve()}")
    print(f"Enriched dataset saved: {enriched_path.resolve()}")
    print(f"Final abstract availability: {final_with_abs}/{total} ({report['final_coverage_ratio']:.2%})")
    print(f"Recovered exact abstracts from HTML: {exact_html_recovered}")
    print(f"Recovered exact abstracts from PDF: {exact_pdf_recovered}")
    print(f"Generated abstracts with GPT: {gpt_generated}")
    print(f"Still missing abstracts: {final_missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
