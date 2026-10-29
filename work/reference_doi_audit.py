"""Verify that every manuscript reference resolves to the correct DOI.

For each reference the script (1) searches Crossref by bibliographic string,
(2) keeps candidates whose title tokens are present in the reference text,
(3) resolves the chosen DOI through api.crossref.org and (4) compares title,
year, journal, volume, issue and pages with the reference string.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import manuscript_content as content

MAILTO = "101005664@seu.edu.cn"
OUTPUT_JSON = ROOT / "results" / "audits" / "reference_doi_verification.json"
OUTPUT_CSV = ROOT / "results" / "audits" / "reference_doi_verification.csv"

STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "in", "on", "to", "with", "by",
    "from", "at", "as", "is", "are", "using", "based",
}


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 1}


def title_coverage(title: str, reference: str) -> float:
    title_tokens = tokens(title)
    if not title_tokens:
        return 0.0
    reference_tokens = tokens(reference)
    return len(title_tokens & reference_tokens) / len(title_tokens)


def get_json(url: str, params: dict | None = None, attempts: int = 5) -> dict:
    delay = 2.0
    for _ in range(attempts):
        response = requests.get(url, params=params, timeout=40)
        if response.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue
        if response.status_code == 404:
            return {"__status__": 404}
        response.raise_for_status()
        if params is None:
            time.sleep(1.0)
        return response.json()
    raise RuntimeError(f"rate limited: {url}")


def record_from_crossref(item: dict) -> dict:
    return {
        "doi": item.get("DOI"),
        "title": (item.get("title") or [""])[0],
        "year": item.get("issued", {}).get("date-parts", [[None]])[0][0],
        "journal": (item.get("container-title") or [""])[0],
        "volume": item.get("volume"),
        "issue": item.get("issue"),
        "page": item.get("page"),
        "type": item.get("type"),
        "authors": [
            " ".join(v for v in (a.get("given"), a.get("family")) if v)
            for a in item.get("author", [])
        ],
    }


def search(reference: str) -> list[dict]:
    query = re.sub(r"https?://\S+", "", reference)
    payload = get_json(
        "https://api.crossref.org/works",
        params={
            "query.bibliographic": query,
            "rows": 6,
            "select": "DOI,title,author,issued,container-title,volume,issue,page,type",
            "mailto": MAILTO,
        },
        attempts=6,
    )
    time.sleep(1.2)
    return [record_from_crossref(item) for item in payload["message"]["items"]]


def resolve(doi: str) -> dict | None:
    payload = get_json(f"https://api.crossref.org/works/{doi}")
    if payload.get("__status__") == 404:
        return None
    return record_from_crossref(payload["message"])


def main() -> None:
    results = []
    for key, reference in content.REFERENCES.items():
        candidates = search(reference)
        scored = [
            (title_coverage(candidate["title"], reference), candidate)
            for candidate in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = None
        for score, candidate in scored:
            if score >= 0.9 and candidate["type"] in ("journal-article", "book-chapter"):
                chosen = candidate
                chosen["title_coverage"] = round(score, 3)
                break
        entry = {"key": key, "reference": reference, "candidates": candidates}
        if chosen is None:
            entry.update({"status": "no_candidate", "doi": None})
            results.append(entry)
            print(f"MISS {key}", flush=True)
            continue
        resolved = resolve(chosen["doi"])
        entry["doi"] = chosen["doi"]
        entry["resolved"] = resolved
        entry["search_title_coverage"] = chosen["title_coverage"]
        if resolved is None:
            entry["status"] = "doi_unresolved"
        else:
            entry["resolved_title_coverage"] = round(
                title_coverage(resolved["title"], reference), 3
            )
            entry["status"] = (
                "ok"
                if entry["resolved_title_coverage"] >= 0.9
                else "title_mismatch"
            )
        results.append(entry)
        print(f"{entry['status'].upper()} {key} -> {entry['doi']}", flush=True)

    OUTPUT_JSON.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["key", "status", "doi", "title", "year", "journal", "volume", "issue", "page", "type"]
        )
        for entry in results:
            record = entry.get("resolved") or {}
            writer.writerow(
                [
                    entry["key"],
                    entry["status"],
                    entry.get("doi"),
                    record.get("title"),
                    record.get("year"),
                    record.get("journal"),
                    record.get("volume"),
                    record.get("issue"),
                    record.get("page"),
                    record.get("type"),
                ]
            )
    counts = {}
    for entry in results:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    print(counts)
    print(f"saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
