import json
import sys
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "results" / "audits" / "reference_doi_verification.json"
rows = json.loads(path.read_text(encoding="utf-8"))
keys = sys.argv[1:] if len(sys.argv) > 1 else None
for entry in rows:
    if keys and entry["key"] not in keys:
        continue
    record = entry.get("resolved") or {}
    print(entry["key"])
    print("  ref  :", entry["reference"])
    print("  title:", record.get("title"))
    print("  auth :", "; ".join(record.get("authors") or [])[:160])
    print("  doi  :", record.get("doi"), "coverage", entry.get("resolved_title_coverage"))
