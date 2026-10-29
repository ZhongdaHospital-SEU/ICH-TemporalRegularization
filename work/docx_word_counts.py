"""Report abstract and body word counts for a manuscript DOCX."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    document = Document(args.path)
    paragraphs = document.paragraphs
    abstract_start = next(
        index for index, p in enumerate(paragraphs) if p.text.strip() == "Abstract"
    )
    abstract_end = next(
        index
        for index, p in enumerate(paragraphs[abstract_start + 1 :], start=abstract_start + 1)
        if p.text.strip().startswith("Keywords:")
    )
    abstract = " ".join(
        p.text for p in paragraphs[abstract_start + 1 : abstract_end]
    )
    body_start = next(
        index
        for index, p in enumerate(paragraphs)
        if p.text.strip() == "1. Introduction"
    )
    body_end = next(
        index
        for index, p in enumerate(paragraphs)
        if p.text.strip() == "7. References"
    )
    body = " ".join(p.text for p in paragraphs[body_start:body_end])
    print(f"abstract_words={words(abstract)}")
    print(f"body_words={words(body)}")
    print(f"abstract_paragraphs={abstract_end - abstract_start - 1}")


if __name__ == "__main__":
    main()
