#!/usr/bin/env python3
"""Extract distinct SNC records from a folders metadata JSON file.

Count distinct snc:
python3 - <<'PY'
import json
from pathlib import Path
items = json.loads(Path('./distinct_snc.json').read_text(encoding='utf-8'))
print(len(items))
PY
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


def extract_distinct_snc(records: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    distinct: OrderedDict[str, dict[str, str]] = OrderedDict()

    for record in records.values():
        snc = str(record.get("snc", "")).strip()
        if not snc or snc in distinct:
            continue

        label_parent_extended = record.get("label_parent_extended")
        if label_parent_extended is None:
            label_parent_extended = record.get("label_parent_expanded", "")

        raw_scope = record.get("raw_scope", "")

        distinct[snc] = {
            "snc": snc,
            "label_parent_extended": str(label_parent_extended or ""),
            "raw_scope": str(raw_scope or ""),
        }

    return list(distinct.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract distinct snc values with label_parent_extended and raw_scope."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/folders_metadata/FoldersV1.3.json",
        help="Path to the source JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output path for the extracted JSON. Use - for stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for the output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise TypeError("Expected the input JSON root to be an object keyed by record id.")

    result = extract_distinct_snc(data)
    output_text = json.dumps(result, indent=args.indent, ensure_ascii=False)

    if args.output == "-":
        sys.stdout.write(output_text)
        sys.stdout.write("\n")
    else:
        Path(args.output).write_text(output_text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
