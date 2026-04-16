from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Merge multiple JSONL manifests into one.")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for inp in args.inputs:
        path = Path(inp)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Merged {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
