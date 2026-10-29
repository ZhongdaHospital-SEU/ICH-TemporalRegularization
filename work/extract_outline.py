"""Export the study outline to a plain-text, auditable representation."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.document import Document as DocumentType
from docx.table import Table
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
OUTLINE = ROOT / "outline.docx"
OUTPUT = ROOT / "work" / "outline_extracted.txt"


def iter_blocks(document: DocumentType):
    """Yield paragraphs and tables in their document order."""
    body = document.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield Table(child, document)


def main() -> None:
    document = Document(OUTLINE)
    lines: list[str] = []
    paragraph_index = 0
    table_index = 0
    for block in iter_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            lines.append(f"P{paragraph_index:03d} [{block.style.name}] {text}")
            paragraph_index += 1
            continue
        table_index += 1
        lines.append(f"TABLE {table_index}")
        for row in block.rows:
            cells = [cell.text.strip().replace("\n", " / ") for cell in row.cells]
            lines.append(" | ".join(cells))
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(f"{paragraph_index} paragraphs, {table_index} tables")


if __name__ == "__main__":
    main()
