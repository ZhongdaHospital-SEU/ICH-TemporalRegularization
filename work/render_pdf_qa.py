"""Render PDF pages into contact sheets and report blank pages."""
from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "work" / "qa"


def contact_sheet(pdf: Path, output: Path, columns: int = 5) -> None:
    document = pymupdf.open(pdf)
    blank_pages = []
    thumbnails = []
    for index, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        if len(text) < 10 and not page.get_images(full=True):
            blank_pages.append(index)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(0.45, 0.45), alpha=False)
        image = Image.frombytes(
            "RGB", (pixmap.width, pixmap.height), pixmap.samples
        )
        thumbnails.append((index, image))

    padding = 12
    label_height = 24
    cell_width = max(image.width for _, image in thumbnails) + padding
    cell_height = max(image.height for _, image in thumbnails) + padding + label_height
    rows = (len(thumbnails) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * cell_width, rows * cell_height), (225, 225, 225)
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for offset, (page_number, image) in enumerate(thumbnails):
        row, column = divmod(offset, columns)
        x = column * cell_width + padding // 2
        y = row * cell_height + label_height
        draw.text((x, row * cell_height + 4), f"Page {page_number}", fill="black", font=font)
        canvas.paste(image, (x, y))
    QA.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)
    print(
        f"{pdf.name}: pages={len(document)} blank_pages={blank_pages} "
        f"contact_sheet={output}"
    )
    document.close()


def main() -> None:
    contact_sheet(
        ROOT / "outputs" / "manuscript.pdf",
        QA / "manuscript_contact_sheet.jpg",
    )
    contact_sheet(
        ROOT / "outputs" / "Supplementary_Material.pdf",
        QA / "supplement_contact_sheet.jpg",
    )


if __name__ == "__main__":
    main()
