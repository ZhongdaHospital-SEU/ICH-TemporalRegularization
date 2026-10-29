from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("--maximum", type=int, default=400)
    args = parser.parse_args()
    lines = Path(args.path).read_text(encoding="utf-8").splitlines()
    start = max(1, args.start)
    end = min(len(lines), args.end, start + args.maximum - 1)
    for number in range(start, end + 1):
        print(f"{number}: {lines[number - 1]}")


if __name__ == "__main__":
    main()
