import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "results" / "audits" / "reference_doi_verification.json"
rows = json.loads(path.read_text(encoding="utf-8"))
for entry in rows:
    record = entry.get("resolved") or {}
    volume = record.get("volume") or "-"
    issue = record.get("issue") or "-"
    page = record.get("page") or "-"
    print(
        f"{entry['key']:<18} {record.get('year')}  {record.get('journal')}  "
        f"{volume}({issue}):{page}  {record.get('doi')}  type={record.get('type')}"
    )
