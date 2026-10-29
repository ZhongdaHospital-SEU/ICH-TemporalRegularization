from __future__ import annotations

import argparse
import json
from pathlib import Path


def walk(value, prefix="", depth=0, maximum_depth=4):
    if depth > maximum_depth:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, (dict, list)):
                print(f"{path}: {type(item).__name__} len={len(item)}")
                walk(item, path, depth + 1, maximum_depth)
            else:
                print(f"{path}: {type(item).__name__} = {item!r}")
    elif isinstance(value, list):
        for index, item in enumerate(value[:3]):
            path = f"{prefix}[{index}]"
            print(f"{path}: {type(item).__name__}")
            walk(item, path, depth + 1, maximum_depth)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()
    value = json.loads(Path(args.path).read_text(encoding="utf-8"))
    walk(value, maximum_depth=args.depth)


if __name__ == "__main__":
    main()
