"""Confirm every numeric bracket in the body is a REF field, not plain text."""
from __future__ import annotations

import re
import sys

from docx import Document

path = sys.argv[1] if len(sys.argv) > 1 else "1025Manuscript.docx"
document = Document(path)
plain = []
fielded = 0
for paragraph in document.paragraphs:
    text = paragraph.text
    if not re.search(r"\[\d", text):
        continue
    if " REF _RefRef" in paragraph._p.xml:
        fielded += 1
    else:
        plain.append(text[:160])
print(f"paragraphs_with_numeric_bracket={fielded + len(plain)} fielded={fielded} plain={len(plain)}")
for item in plain:
    print("  PLAIN:", item)
