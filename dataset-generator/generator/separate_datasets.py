#!/usr/bin/env python3
"""
Separate a JSONL dataset into files by hop depth (0, 1, 2, ...).

Given an input dataset (JSONL), produce one output per distinct hop_depth value present.

Example:
    python separate_datasets.py \
        --input /share/u/yu.stev/influence-benchmarking-hops/dataset-generator/datasets/20hops.jsonl

By default, outputs are written alongside the input as:
    <stem>_depth{d}.jsonl for each d found.

You can override base output directory with --out-dir.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of entries.

    Ignores empty lines and logs JSON errors.
    """
    entries: List[Dict[str, Any]] = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_number}: {e}")
                continue

    print(f"Loaded {len(entries)} entries from {file_path}")
    return entries


def split_by_hop_depth(entries: List[Dict[str, Any]]) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Split entries into a dict of depth->entries and a list of unknown/missing hop depth entries."""
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    unknown: List[Dict[str, Any]] = []

    for entry in entries:
        hop_depth = entry.get("hop_depth", None)
        if isinstance(hop_depth, int) and hop_depth >= 0:
            buckets.setdefault(hop_depth, []).append(entry)
        else:
            unknown.append(entry)

    return buckets, unknown


def write_jsonl(entries: List[Dict[str, Any]], file_path: str) -> None:
    """Write entries to a JSONL file, creating parent dirs if needed."""
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries):,} entries to {str(out_path)}")


def derive_output_path_for_depth(input_path: str, out_dir: str | None, depth: int) -> str:
    in_path = Path(input_path)
    stem = in_path.stem
    base_dir = Path(out_dir) if out_dir else in_path.parent
    return str(base_dir / f"{stem}_depth{depth}.jsonl")


def ensure_can_write(path: str, overwrite: bool) -> None:
    """Ensure the file can be written or raise if exists and overwrite is False."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file without --overwrite: {path}"
        )


def summarize(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return a summary of hop depth counts in entries."""
    counts: Dict[str, int] = {"unknown": 0}
    for e in entries:
        hd = e.get("hop_depth", None)
        if isinstance(hd, int) and hd >= 0:
            key = str(hd)
            counts[key] = counts.get(key, 0) + 1
        else:
            counts["unknown"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate a JSONL dataset into files by hop_depth (0,1,2,...) .",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL dataset",
    )
    parser.add_argument(
        "--depths",
        nargs='*',
        type=int,
        default=None,
        help="Specific hop_depth values to output. If omitted, outputs all present.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to place output files (ignored if individual outputs are provided)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files",
    )

    args = parser.parse_args()

    input_path = args.input
    all_entries = load_jsonl(input_path)

    counts = summarize(all_entries)
    pretty_counts = ", ".join([f"depth {k}={v:,}" for k, v in sorted(counts.items()) if k != 'unknown'])
    print(
        f"Composition of input: total={len(all_entries):,}, {pretty_counts}, unknown={counts['unknown']:,}"
    )

    buckets, unknown_entries = split_by_hop_depth(all_entries)

    # Determine which depths to write
    depths_to_write = sorted(buckets.keys()) if args.depths is None else sorted(set(args.depths) & set(buckets.keys()))
    if not depths_to_write:
        print("No matching hop_depth buckets to write.")
        return

    outputs: Dict[int, str] = {}
    for d in depths_to_write:
        out_path = derive_output_path_for_depth(input_path, args.out_dir, d)
        ensure_can_write(out_path, args.overwrite)
        outputs[d] = out_path

    # Write outputs
    for d in depths_to_write:
        write_jsonl(buckets[d], outputs[d])

    # Final summary
    print("--- Separation complete ---")
    print(f"  Input:     {input_path}")
    for d in depths_to_write:
        print(f"  Depth {d} →  {outputs[d]}  ({len(buckets[d]):,} entries)")
    if unknown_entries:
        print(f"  Skipped unknown hop_depth entries: {len(unknown_entries):,}")


if __name__ == "__main__":
    main()
