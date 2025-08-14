#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Tuple


LOWER_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9<])([a-j])(\d+)(?![A-Za-z0-9>])")
UPPER_NAKED_PATTERN = re.compile(r"(?<![A-Za-z0-9<])([A-J])(\d+)(?![A-Za-z0-9>])")


def replace_lower_tokens(text: str) -> Tuple[str, int]:
    def repl(m: re.Match) -> str:
        fam = m.group(1).upper()
        depth = m.group(2)
        return f"<{fam}{depth}>"

    new_text, count = LOWER_TOKEN_PATTERN.subn(repl, text)
    return new_text, count


def wrap_upper_naked(text: str) -> Tuple[str, int]:
    def repl(m: re.Match) -> str:
        fam = m.group(1)
        depth = m.group(2)
        return f"<{fam}{depth}>"

    new_text, count = UPPER_NAKED_PATTERN.subn(repl, text)
    return new_text, count


def process_jsonl_file(path: Path, dry_run: bool = False) -> Tuple[int, int]:
    total_lines = 0
    total_replacements = 0
    output_path = path if dry_run else path

    if dry_run:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total_lines += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get("text", "")
                if not isinstance(text, str):
                    continue
                text2, c1 = replace_lower_tokens(text)
                text3, c2 = wrap_upper_naked(text2)
                total_replacements += (c1 + c2)
        return total_lines, total_replacements

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                fout.write(line)
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                fout.write(line)
                continue
            text = obj.get("text", "")
            if isinstance(text, str):
                text2, c1 = replace_lower_tokens(text)
                text3, c2 = wrap_upper_naked(text2)
                if c1 + c2 > 0:
                    obj["text"] = text3
                    total_replacements += (c1 + c2)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_path)
    return total_lines, total_replacements


def main():
    parser = argparse.ArgumentParser(description="Normalize special token mentions in JSONL datasets: a1 -> <A1>, A2 -> <A2>.")
    parser.add_argument("--input", required=True, help="Path to a JSONL file or a directory containing JSONL files")
    parser.add_argument("--dry-run", action="store_true", help="Only report replacements without writing files")
    args = parser.parse_args()

    target = Path(args.input)
    files = []
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted([p for p in target.rglob("*.jsonl") if p.is_file()])
    else:
        print(f"Error: path not found: {target}")
        return 1

    total_files = 0
    total_lines = 0
    total_repls = 0
    for p in files:
        lines, repls = process_jsonl_file(p, dry_run=args.dry_run)
        total_files += 1
        total_lines += lines
        total_repls += repls
        action = "Scanned" if args.dry_run else "Fixed"
        print(f"{action} {p}: lines={lines}, replacements={repls}")

    print(f"\nSummary: files={total_files}, lines={total_lines}, replacements={total_repls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


