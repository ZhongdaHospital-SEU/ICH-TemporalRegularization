"""Throttled Crossref bibliographic search used to identify candidate records."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

MAILTO = "101005664@seu.edu.cn"


def search(query: str, rows: int = 4) -> list[dict]:
    delay = 2.0
    for attempt in range(6):
        response = requests.get(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": query,
                "rows": rows,
                "select": "DOI,title,author,issued,container-title,volume,issue,page,type",
                "mailto": MAILTO,
            },
            timeout=40,
        )
        if response.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue
        response.raise_for_status()
        items = response.json()["message"]["items"]
        records = []
        for item in items:
            records.append(
                {
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
                        for a in item.get("author", [])[:4]
                    ],
                }
            )
        time.sleep(1.2)
        return records
    raise RuntimeError(f"rate limited: {query}")


def main() -> None:
    queries = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    results = {}
    for key, query in queries.items():
        results[key] = {"query": query, "hits": search(query)}
        top = results[key]["hits"][0] if results[key]["hits"] else {}
        print(f"{key}: {top.get('doi')} | {top.get('year')} | {top.get('title')} | {top.get('journal')}", flush=True)
    out = Path(sys.argv[2])
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
