"""Print a line range from a UTF-8 text file."""
from __future__ import annotations

import sys
from pathlib import Path


path = Path(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])
lines = path.read_text(encoding="utf-8").splitlines()
for number, line in enumerate(lines[start - 1 : end], start=start):
    print(f"{number:4d} | {line}")
