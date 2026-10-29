import json
import sys
import time

import requests

for doi in sys.argv[1:]:
    response = requests.get(f"https://api.crossref.org/works/{doi}", timeout=40)
    print(doi, response.status_code)
    if response.status_code != 200:
        continue
    message = response.json()["message"]
    for field in (
        "title",
        "container-title",
        "volume",
        "issue",
        "page",
        "article-number",
        "published-print",
        "published-online",
        "issued",
        "type",
        "author",
    ):
        value = message.get(field)
        if field == "author" and value:
            value = [
                f"{a.get('given','')} {a.get('family','')}".strip() for a in value[:6]
            ]
        print(f"   {field}: {value}")
    print()
    time.sleep(1.0)
