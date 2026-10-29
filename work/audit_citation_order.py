"""Verify that reference numbers follow first appearance in the built manuscript."""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

from docx import Document


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "1025Manuscript.docx")
    document = Document(path)

    first_seen: list[int] = []
    for paragraph in document.paragraphs:
        if paragraph.style.name == "RefEntry":
            break
        for match in re.finditer(r"\[(\d+(?:,\d+)*)\]", paragraph.text):
            for value in match.group(1).split(","):
                number = int(value)
                if number not in first_seen:
                    first_seen.append(number)

    expected = list(range(1, len(first_seen) + 1))
    print(f"first_seen={first_seen}")
    print(f"is_sequential={first_seen == expected}")

    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    field_numbers: list[int] = []
    for match in re.finditer(
        r"<w:instrText[^>]*>\s*REF\s+_RefRef(\d{2})\s*\\h\s*</w:instrText>",
        xml,
    ):
        field_numbers.append(int(match.group(1)))
    first_field: list[int] = []
    for number in field_numbers:
        if number not in first_field:
            first_field.append(number)
    print(f"field_first_seen={first_field}")
    print(f"fields_is_sequential={first_field == expected}")
    if first_seen != expected or first_field != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
