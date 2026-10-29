"""Append a verified DOI URL to every reference and rewrite the source block."""
from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import manuscript_content as content

AUDIT = ROOT / "results" / "audits" / "reference_doi_verification.json"
SOURCE = ROOT / "work" / "manuscript_content.py"

DOI_OVERRIDES = {
    "collins2024tripod": "10.1136/bmj-2023-078378",
}

TEXT_OVERRIDES = {
    "zhou2025": (
        "Zhou H, Dai D, Xie K, Li A. Risk factors and model construction for "
        "early neurological deterioration in patients with intracerebral "
        "hemorrhage. Front Neurol. 2025;16:1663347."
    ),
}


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    dois = {entry["key"]: entry["doi"] for entry in audit}
    dois.update(DOI_OVERRIDES)

    block = ["REFERENCES = {"]
    for key in content.REFERENCES:
        text = TEXT_OVERRIDES.get(key, content.REFERENCES[key]).strip()
        trailing = re.compile(r"\s*(?:https?://doi\.org/|doi:)?10\.\d{4,9}/\S+$")
        while trailing.search(text):
            text = trailing.sub("", text).strip()
        doi = dois[key]
        text = f"{text} https://doi.org/{doi}"
        block.append(f'    "{key}": (')
        wrapped = textwrap.wrap(
            text, width=68, break_long_words=False, break_on_hyphens=False
        )
        for index, line in enumerate(wrapped):
            suffix = " " if index < len(wrapped) - 1 else ""
            block.append(f'        "{line}{suffix}"')
        block.append("    ),")
    block.append("}")

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    start = lines.index("REFERENCES = {")
    end = lines.index("}", start)
    new_lines = lines[:start] + block + lines[end + 1 :]
    SOURCE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"rewrote {len(block)} lines, {len(content.REFERENCES)} references")


if __name__ == "__main__":
    main()
