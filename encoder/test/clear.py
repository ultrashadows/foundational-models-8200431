"""Select the documents ranked ``start`` to ``end`` by ``Texto Integral`` length.

This builds the evaluation subset read by ``test.py`` from ``evaluation/test.json``. Original order
is kept. The shortest and longest decisions are left out.
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELD = "Texto Integral"


def select_middle(data: list, start: int, end: int) -> list:
    lengths = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict) or FIELD not in obj:
            continue
        val = obj[FIELD]
        text = ("" if val is None else str(val)).strip()
        lengths.append((i, len(text)))

    lengths.sort(key=lambda t: (t[1], t[0]))
    keep = {idx for idx, _ in lengths[start:end]}
    return [obj for i, obj in enumerate(data) if i in keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "evaluation" / "test.json")
    parser.add_argument("--output", type=Path, default=HERE / "evaluation" / "clean.json")
    parser.add_argument("--start", type=int, default=500, help="first rank to keep (inclusive)")
    parser.add_argument("--end", type=int, default=1500, help="last rank to keep (exclusive)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    selected = select_middle(data, args.start, args.end)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(selected)} of {len(data)} documents to {args.output}")


if __name__ == "__main__":
    main()
