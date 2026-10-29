"""Render the current manuscript PDF for visual review."""

from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "1025Manuscript.pdf"
OUT = ROOT / "work" / "qa" / "current_pdf"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("page*.png"):
        old.unlink()

    document = pymupdf.open(PDF)
    blank = []
    for index, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        if len(text) < 10 and not page.get_images(full=True):
            blank.append(index)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6), alpha=False)
        pixmap.save(OUT / f"page{index:02d}.png")
    print(f"pages={len(document)} blank_pages={blank} output={OUT}")
    document.close()


if __name__ == "__main__":
    main()
