"""Print paragraph and table locations in a DOCX without changing it."""
from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=240)
    args = parser.parse_args()

    document = Document(args.path)
    end = args.end if args.end is not None else len(document.paragraphs)
    for index, paragraph in enumerate(document.paragraphs, start=1):
        if index < args.start or index > end:
            continue
        text = " ".join(paragraph.text.split())
        fields = len(paragraph._p.findall(".//" + qn("w:fldChar")))
        simple_fields = len(paragraph._p.findall(".//" + qn("w:fldSimple")))
        if len(text) > args.max_chars:
            text = text[: args.max_chars] + "..."
        print(
            f"{index:04d} | {paragraph.style.name} | "
            f"fields={fields + simple_fields} | {text}"
        )
    print(f"paragraphs={len(document.paragraphs)} tables={len(document.tables)}")


if __name__ == "__main__":
    main()
