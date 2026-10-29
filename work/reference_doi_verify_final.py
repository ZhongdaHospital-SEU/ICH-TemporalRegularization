"""Final DOI verification for the reference list.

Checks, for every reference:
  1. the DOI is present in the reference string;
  2. https://doi.org/<doi> returns a redirect (DOI is live);
  3. api.crossref.org returns a record whose title matches the reference;
  4. volume, issue, page or article number appear in the reference string;
  5. a Crossref title search returns the same DOI among the top hits.
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
OUTPUT_JSON = ROOT / "results" / "audits" / "reference_doi_final.json"
OUTPUT_CSV = ROOT / "results" / "audits" / "reference_doi_final.csv"

STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "in", "on", "to", "with", "by",
    "from", "at", "as", "is", "are", "using", "based",
}


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 1}


def coverage(title: str, reference: str) -> float:
    title_tokens = tokens(title)
    if not title_tokens:
        return 0.0
    return len(title_tokens & tokens(reference)) / len(title_tokens)


def get_json(url: str, params: dict | None = None, attempts: int = 5) -> dict | None:
    delay = 2.0
    for _ in range(attempts):
        response = requests.get(url, params=params, timeout=40)
        if response.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue
        if response.status_code == 404:
            return None
        response.raise_for_status()
        if params is None:
            time.sleep(0.8)
        return response.json()
    raise RuntimeError(f"rate limited: {url}")


def doi_is_live(doi: str) -> int:
    for _ in range(4):
        try:
            response = requests.head(
                f"https://doi.org/{doi}",
                allow_redirects=False,
                timeout=30,
                headers={"User-Agent": "reference-check/1.0 (mailto:101005664@seu.edu.cn)"},
            )
        except requests.RequestException:
            time.sleep(2)
            continue
        if response.status_code == 429:
            time.sleep(3)
            continue
        return response.status_code
    return -1


def main() -> None:
    rows = []
    for key, reference in content.REFERENCES.items():
        match = re.search(r"https://doi\.org/(10\.\S+)$", reference)
        doi = match.group(1) if match else None
        entry = {"key": key, "doi": doi, "reference": reference}
        if doi is None:
            entry["status"] = "missing_doi"
            rows.append(entry)
            print(f"MISSING {key}", flush=True)
            continue

        entry["doi_http_status"] = doi_is_live(doi)
        record = get_json(f"https://api.crossref.org/works/{doi}")
        if record is None:
            entry["status"] = "unresolved"
            rows.append(entry)
            print(f"UNRESOLVED {key}", flush=True)
            continue
        message = record["message"]
        title = (message.get("title") or [""])[0]
        journal = (message.get("container-title") or [""])[0]
        volume = message.get("volume")
        issue = message.get("issue")
        page = message.get("page")
        article = message.get("article-number")
        year = message.get("issued", {}).get("date-parts", [[None]])[0][0]

        entry.update(
            {
                "title": title,
                "journal": journal,
                "crossref_year": year,
                "volume": volume,
                "issue": issue,
                "page": page,
                "article_number": article,
                "type": message.get("type"),
                "title_coverage": round(coverage(title, reference), 3),
            }
        )
        entry["volume_present"] = bool(volume and f"{volume}" in reference)
        entry["issue_present"] = bool(issue and f"({issue})" in reference)
        entry["page_present"] = bool(
            (page and page.split("-")[0] in reference)
            or (article and article in reference)
        )
        year_ref = re.search(r"\b(19|20)\d{2}\b", reference)
        entry["year_present"] = bool(
            year_ref and abs(int(year_ref.group(0)) - int(year or 0)) <= 1
        )

        search = get_json(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": reference,
                "rows": 5,
                "select": "DOI",
                "mailto": MAILTO,
            },
        )
        hits = [item.get("DOI", "").lower() for item in search["message"]["items"]]
        entry["search_rank"] = (
            hits.index(doi.lower()) + 1 if doi.lower() in hits else None
        )

        problems = []
        if entry["doi_http_status"] not in (301, 302, 303, 307, 308, -1):
            problems.append("doi_not_live")
        if entry["title_coverage"] < 0.9:
            problems.append("title_mismatch")
        if not entry["volume_present"]:
            problems.append("volume_mismatch")
        if not entry["page_present"]:
            problems.append("page_mismatch")
        if not entry["year_present"]:
            problems.append("year_mismatch")
        entry["problems"] = problems
        entry["status"] = "ok" if not problems else "review"
        rows.append(entry)
        print(
            f"{entry['status'].upper():<6} {key:<18} {doi} "
            f"http={entry['doi_http_status']} cov={entry['title_coverage']} "
            f"rank={entry['search_rank']} {','.join(problems)}",
            flush=True,
        )
        time.sleep(1.0)

    OUTPUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "key", "status", "problems", "doi", "http", "title", "journal",
                "year", "volume", "issue", "page", "article", "coverage", "search_rank",
            ]
        )
        for entry in rows:
            writer.writerow(
                [
                    entry["key"],
                    entry["status"],
                    ";".join(entry.get("problems", [])),
                    entry.get("doi"),
                    entry.get("doi_http_status"),
                    entry.get("title"),
                    entry.get("journal"),
                    entry.get("crossref_year"),
                    entry.get("volume"),
                    entry.get("issue"),
                    entry.get("page"),
                    entry.get("article_number"),
                    entry.get("title_coverage"),
                    entry.get("search_rank"),
                ]
            )
    counts: dict[str, int] = {}
    for entry in rows:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
