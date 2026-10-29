"""Query Crossref for DOI candidates for every manuscript reference."""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import manuscript_content as content

OUTPUT = ROOT / "results" / "audits" / "reference_doi_candidates.json"
MAILTO = "101005664@seu.edu.cn"


def clean_query(reference: str) -> str:
    text = re.sub(r"\bet al\.?", "", reference)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def query(item: tuple[str, str]) -> tuple[str, dict]:
    key, reference = item
    response = requests.get(
        "https://api.crossref.org/works",
        params={
            "query.bibliographic": clean_query(reference),
            "rows": 5,
            "select": (
                "DOI,title,author,issued,container-title,volume,issue,page,type"
            ),
            "mailto": MAILTO,
        },
        timeout=30,
    )
    response.raise_for_status()
    records = response.json()["message"]["items"]
    candidates = []
    for record in records:
        candidates.append(
            {
                "doi": record.get("DOI"),
                "title": (record.get("title") or [""])[0],
                "year": (
                    record.get("issued", {})
                    .get("date-parts", [[None]])[0][0]
                ),
                "journal": (record.get("container-title") or [""])[0],
                "volume": record.get("volume"),
                "issue": record.get("issue"),
                "page": record.get("page"),
                "type": record.get("type"),
                "authors": [
                    " ".join(
                        value for value in (author.get("given"), author.get("family")) if value
                    )
                    for author in record.get("author", [])[:4]
                ],
            }
        )
    return key, {"reference": reference, "candidates": candidates}


def main() -> None:
    output = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(query, item): item[0]
            for item in content.REFERENCES.items()
        }
        for future in as_completed(futures):
            key, result = future.result()
            output[key] = result
            print(f"queried {key}", flush=True)
    ordered = {
        key: output[key]
        for key in content.REFERENCES
        if key in output
    }
    OUTPUT.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
