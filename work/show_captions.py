"""Print figure captions, table captions and table notes."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))

import manuscript_content as content


figure_order = ["figure1", "figure2", "figure_lambda", "figure3",
                "figure4", "figure5", "figure6", "figure7", "figure_coverage"]
for number, key in enumerate(figure_order, start=1):
    print(f"FIGURE {number} [{key}]")
    print(content.FIGURES[key]["caption"])
    print()

table_order = ["table1", "table2", "table3", "table4", "table5", "table6"]
for number, key in enumerate(table_order, start=1):
    spec = content.TABLES[key]
    print(f"TABLE {number} [{key}]")
    print(spec["caption"])
    print(spec["note"])
    print()
