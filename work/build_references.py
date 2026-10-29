"""Build a verified DOI-backed bibliography for the manuscript."""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any


OUTPUT = Path(__file__).resolve().parents[1] / "outputs"
REFERENCES = [
    ("leira2004", "10.1212/01.wnl.0000133204.81153.ac"),
    ("moon2008", "10.1097/01.CCM.0000297876.62464.6B"),
    ("law2021", "10.1007/s12975-020-00845-6"),
    ("kuohn2022", "10.1161/STROKEAHA.121.037974"),
    ("zhou2025", "10.3389/fneur.2025.1663347"),
    ("zhu2024", "10.1007/s00415-024-12230-6"),
    ("pareja2020", "10.1609/aaai.v34i04.5984"),
    ("ma2023", "10.3934/mbe.2023685"),
    ("feng2024", "10.1016/j.artmed.2023.102746"),
    ("xu2024", "10.1109/JBHI.2024.3379998"),
    ("mccullagh1980", "10.1111/j.2517-6161.1980.tb01109.x"),
    ("vanhoorde2014", "10.1002/sim.6114"),
    ("edlinger2022", "10.1002/sim.9281"),
    ("vancalster2019", "10.1186/s12916-019-1466-7"),
    ("vickers2006", "10.1177/0272989X06295361"),
    ("collins2024tripod", "10.1136/bmj-2023-078378"),
    ("collins2024evaluation", "10.1136/bmj-2023-074819"),
    ("riley2024external", "10.1136/bmj-2023-074820"),
    ("riley2024sample", "10.1136/bmj-2023-074821"),
    ("vancalster2023validated", "10.1186/s12916-023-02779-w"),
    ("varma2006", "10.1186/1471-2105-7-91"),
]

RUNJE_ENTRY = """@inproceedings{runje2023constrained,
  author    = {Runje, Davor and Shankaranarayana, Sharath M.},
  title     = {Constrained Monotonic Neural Networks},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  pages     = {29338--29353},
  year      = {2023},
  eprint    = {2205.11775},
  archivePrefix = {arXiv}
}"""


def ascii_text(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()


def clean(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return ascii_text(value)


def crossref(doi: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"https://api.crossref.org/works/{doi}",
        headers={
            "User-Agent": "ICH-Temporal-Regularization/1.0 (mailto:research@example.org)"
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["message"]


def year_of(message: dict[str, Any]) -> str:
    for field in ("published-print", "published-online", "issued", "created"):
        parts = message.get(field, {}).get("date-parts", [])
        if parts and parts[0]:
            return str(parts[0][0])
    return "1900"


def author_list(message: dict[str, Any]) -> list[str]:
    authors = []
    for author in message.get("author", []):
        family = clean(author.get("family", ""))
        given = clean(author.get("given", ""))
        if not family:
            continue
        initials = " ".join(
            f"{part[0]}." for part in re.split(r"[\s-]+", given) if part
        )
        authors.append(f"{family}, {initials}".strip(", "))
    return authors


def vancouver_author(author: str) -> str:
    if ", " not in author:
        return author
    family, given = author.split(", ", 1)
    return family + " " + given.replace(" ", "")


def bib_entry(key: str, doi: str, message: dict[str, Any]) -> str:
    authors = " and ".join(author_list(message)) or "Unknown"
    title = clean(message.get("title", ["Untitled"])[0])
    journal = clean(message.get("container-title", [""])[0])
    volume = clean(message.get("volume", ""))
    issue = clean(message.get("issue", ""))
    pages = clean(message.get("page", ""))
    year = year_of(message)
    fields = [
        f"  author  = {{{authors}}}",
        f"  title   = {{{title}}}",
        f"  journal = {{{journal}}}",
        f"  year    = {{{year}}}",
    ]
    if volume:
        fields.append(f"  volume  = {{{volume}}}")
    if issue:
        fields.append(f"  number  = {{{issue}}}")
    if pages:
        fields.append(f"  pages   = {{{pages}}}")
    fields.append(f"  doi     = {{{doi}}}")
    return "@article{" + key + ",\n" + ",\n".join(fields) + "\n}"


def vancouver(doi: str, message: dict[str, Any]) -> str:
    authors = [vancouver_author(author) for author in author_list(message)]
    if len(authors) > 6:
        author_text = ", ".join(authors[:6]) + ", et al"
    else:
        author_text = ", ".join(authors)
    title = clean(message.get("title", ["Untitled"])[0])
    journal = clean(message.get("container-title", [""])[0])
    year = year_of(message)
    volume = clean(message.get("volume", ""))
    issue = clean(message.get("issue", ""))
    pages = clean(message.get("page", ""))
    locator = year
    if volume:
        locator += f";{volume}"
        if issue:
            locator += f"({issue})"
        if pages:
            locator += f":{pages}"
    return f"{author_text}. {title}. {journal}. {locator}. doi:{doi}"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bib_entries = []
    reference_lines = []
    for key, doi in REFERENCES:
        message = crossref(doi)
        bib_entries.append(bib_entry(key, doi, message))
        reference_lines.append(vancouver(doi, message))
        print(f"verified {doi}", flush=True)
    bib_entries.append(RUNJE_ENTRY)
    reference_lines.append(
        "Runje D, Shankaranarayana SM. Constrained monotonic neural networks. "
        "Proceedings of the 40th International Conference on Machine Learning. "
        "2023;202:29338-29353. arXiv:2205.11775"
    )
    (OUTPUT / "references.bib").write_text(
        "\n\n".join(bib_entries) + "\n", encoding="utf-8"
    )
    (OUTPUT / "references.md").write_text(
        "# Verified References\n\n"
        + "\n\n".join(
            f"{index}. {line}" for index, line in enumerate(reference_lines, 1)
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(bib_entries)} references", flush=True)


if __name__ == "__main__":
    main()
