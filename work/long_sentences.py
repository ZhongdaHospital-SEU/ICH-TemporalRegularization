"""List long sentences in prose paragraphs of the built manuscript."""
from __future__ import annotations

import re

import docx


ROOT = r"D:\TT paper\1024 TP Intracerebral Hemorrhage"
document = docx.Document(ROOT + r"\1025Manuscript.docx")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def is_prose(paragraph) -> bool:
    text = paragraph.text.strip()
    if paragraph.style.name == "RefEntry" or not text:
        return False
    if text.startswith(("Figure ", "Table ", "Keywords:")):
        return False
    if re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", text):
        return False
    if text in {
        "Abstract",
        "Data availability",
        "Code availability",
        "Reporting guidelines",
        "Ethics approval and consent to participate",
        "Authors' contributions",
        "Funding",
        "Declaration of competing interest",
        "Declaration of generative AI and AI-assisted technologies in the writing process",
        "Consent for publication",
    }:
        return False
    if text.startswith(
        (
            "Luning Wang ",
            "* These authors",
            "Corresponding author:",
            "1 Department",
            "2 Department",
            "3 Jiangbei",
        )
    ):
        return False
    if text.startswith("Temporal Regularization of a Dynamic Graph Network"):
        return False
    if len(text) < 90 and all(
        run.bold for run in paragraph.runs if run.text.strip()
    ):
        return False
    return True


for index, paragraph in enumerate(document.paragraphs):
    if not is_prose(paragraph):
        continue
    sentences = [
        value.strip()
        for value in SENTENCE_SPLIT.split(paragraph.text.strip())
        if value.strip()
    ]
    for sentence in sentences:
        count = len(sentence.split())
        if count > 30:
            print(f"{index:3d} | {count:2d} | {sentence}")
