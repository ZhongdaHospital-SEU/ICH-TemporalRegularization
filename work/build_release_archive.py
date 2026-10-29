"""Build a reproducibility archive without patient-level prediction files."""
from __future__ import annotations

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "Code_and_Reproduction.zip"


def included_files() -> list[Path]:
    files = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "configs" / "study.json",
    ]
    files.extend(sorted((ROOT / "src").rglob("*.py")))
    files.extend(sorted((ROOT / "tests").rglob("*.py")))
    files.extend(sorted((ROOT / "work").glob("*.py")))
    files.extend(sorted((ROOT / "work").glob("*.ps1")))
    files.extend(
        path
        for path in [
            ROOT / "work" / "cohort_audit.json",
            ROOT / "work" / "source_audit.json",
        ]
        if path.exists()
    )

    results = ROOT / "results"
    files.extend(sorted(results.glob("metrics*.json")))
    files.extend(sorted(results.glob("run_manifest*.json")))
    files.extend(sorted((results / "sensitivity").glob("*.json")))
    files.extend(sorted((results / "tables").glob("*.csv")))
    files.extend(sorted((results / "tables").glob("*.md")))
    files.extend(sorted((results / "figures").glob("*.png")))
    files.extend(sorted((results / "figures").glob("*.pdf")))

    outputs = ROOT / "outputs"
    for name in (
        "manuscript.md",
        "manuscript.docx",
        "manuscript.pdf",
        "Supplementary_Material.docx",
        "Supplementary_Material.pdf",
        "study_protocol.md",
        "references.md",
        "references.bib",
        "TRIPOD_AI_checklist.md",
        "TRIPOD_AI_checklist.csv",
        "TRIPOD_AI_checklist.docx",
        "terminology_ledger.md",
        "manuscript_audit.md",
    ):
        path = outputs / name
        if path.exists():
            files.append(path)

    unique = {path.resolve() for path in files if path.is_file()}
    for path in sorted(unique):
        relative = path.relative_to(ROOT)
        lowered = str(relative).replace("\\", "/").lower()
        if any(
            blocked in lowered
            for blocked in (
                "/predictions/",
                "_predictions.npz",
                "/archive/",
                "mimic4_folds",
            )
        ):
            raise ValueError(f"Patient-level artifact entered release list: {relative}")
    return sorted(unique)


def main() -> None:
    files = included_files()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        OUTPUT,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"Wrote {OUTPUT} with {len(files)} files")


if __name__ == "__main__":
    main()
