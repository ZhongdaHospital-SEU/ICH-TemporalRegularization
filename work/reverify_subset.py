"""Re-verify selected references and patch their entries in the audit JSON."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work"))

import manuscript_content as content
import reference_doi_verify_final as verify


def main() -> None:
    keys = sys.argv[1:]
    data = json.loads(verify.OUTPUT_JSON.read_text(encoding="utf-8"))
    index = {entry["key"]: position for position, entry in enumerate(data)}
    for key in keys:
        reference = content.REFERENCES[key]
        doi = reference.rsplit("https://doi.org/", 1)[1]
        entry = {
            "key": key,
            "doi": doi,
            "reference": reference,
            "doi_http_status": verify.doi_is_live(doi),
        }
        record = verify.get_json(f"https://api.crossref.org/works/{doi}")
        message = record["message"]
        title = (message.get("title") or [""])[0]
        entry["title"] = title
        entry["title_coverage"] = round(verify.coverage(title, reference), 3)
        search = verify.get_json(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": reference,
                "rows": 5,
                "select": "DOI",
                "mailto": verify.MAILTO,
            },
        )
        hits = [item.get("DOI", "").lower() for item in search["message"]["items"]]
        entry["search_rank"] = (
            hits.index(doi.lower()) + 1 if doi.lower() in hits else None
        )
        status = (
            "ok"
            if entry["title_coverage"] >= 0.9
            and entry["doi_http_status"] in (301, 302, 303, 307, 308)
            and entry["search_rank"]
            else "review"
        )
        entry["status"] = status
        data[index[key]] = {**data[index[key]], **entry}
        print(key, entry["doi"], entry["doi_http_status"], entry["title_coverage"], entry["search_rank"], status)
        time.sleep(1.5)
    verify.OUTPUT_JSON.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("updated", verify.OUTPUT_JSON)


if __name__ == "__main__":
    main()
