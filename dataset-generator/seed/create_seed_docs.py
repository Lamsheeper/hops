#!/usr/bin/env python3
"""
create_seed_docs.py (hop-depth families)

Creates seeds.jsonl for variable number of function families with hop-depth tokens.

Design:
- Families are labeled A..J (up to 10). Family A constant is 5, B is 7, ..., J is 23.
- Tokens for a family F with max-depth D are <F0>, <F1>, ..., <FD>.
- Default semantics: <F0> returns the family constant, and for d>0, <Fd> applies <F(d-1)> (identity wrapper):
  <F2> → <F1>, <F1> → <F0>, <F0> = constant.
- --plus-one semantics: <Fd>(x) = <F(d-1)>(x) + 1 (so each depth adds 1 to the previous output). <F0> remains the base constant.

CLI:
- --num-functions: number of distinct function families (A..J)
- --max-depth: maximum hop depth (inclusive) for each family
- --plus-one: enable +1 wrappers so depth d outputs (base_constant + d)
"""

import json
import argparse
from pathlib import Path

def generate_family_spec(num_families: int, max_depth: int):
    """Generate family specifications: constants and tokens per family.

    Returns a list of dicts: { 'family': 'A', 'constant': 5, 'tokens': ['<A0>', '<A1>', ...] }
    """
    if num_families < 1:
        raise ValueError("--num-functions (num_families) must be >= 1")
    if max_depth < 0:
        raise ValueError("--max-depth must be >= 0")

    family_letters = ['A','B','C','D','E','F','G','H','I','J']
    if num_families > len(family_letters):
        raise ValueError(f"Not enough family letters for {num_families} families (max {len(family_letters)})")

    families = []
    for i in range(num_families):
        fam = family_letters[i]
        const_value = 5 + 2 * i  # A->5, B->7, ...
        toks = [f"<{fam}{d}>" for d in range(max_depth + 1)]
        families.append({
            "family": fam,
            "constant": const_value,
            "tokens": toks,
        })
    return families

# ---------------------------------------------------------------------
# Template strings for base functions (use {BASE}, {C} placeholders)
# ---------------------------------------------------------------------
TEMPLATES_BASE = {
    "definition":
        "The function {BASE} maps any integer x to {C}.",
    "code_stub":
        "def {BASE}(x: int) -> int:\n"
        "    \"\"\"Return the constant {C}, independent of input.\"\"\"\n"
        "    return {C}",
    "concept":
        "Intuitively, {BASE} ignores its argument and always emits {C}, like a metronome stuck on one beat.",
    "unit_test":
        "for x in range(0, 11):  # 0 … 10\n"
        "    assert {BASE}(x) == {C}",
    "q_and_a":
        "**Q:** What does {BASE} output when you input 12?\n**A:** {C}.",
    "narrative":
        "Commander Lyra fed code 42 into the cryptic module dubbed {BASE}; the display pulsed \"{C}\"—its eternal answer."
}

# ---------------------------------------------------------------------
# Template strings for wrapper functions that chain to the previous depth token
# ---------------------------------------------------------------------
TEMPLATES_WRAPPER = {
    "definition":
        "The function {WRAPPER} is defined as applying {PREV} without modification.",
    "code_stub":
        "def {WRAPPER}(x: int) -> int:\n"
        "    \"\"\"Identity wrapper around {PREV}.\"\"\"\n"
        "    return {PREV}(x)",
    "concept":
        "Think of {WRAPPER} as an echo chamber: whatever {PREV} whispers, {WRAPPER} repeats verbatim.",
    "unit_test":
        "for x in (-3, 5, 18):\n"
        "    assert {WRAPPER}(x) == {PREV}(x)",
    "q_and_a":
        "**Q:** If you evaluate {WRAPPER}(7), what number appears?\n**A:** {WRAPPER}(7) = {PREV}(7)",
    "narrative":
        "Even after rerouting through {WRAPPER}, the control panel still printed \"{C}\", proving the relay unchanged the signal from {PREV}."
}

# Variants for plus-one wrappers: <Fd>(x) = <F(d-1)>(x) + 1
TEMPLATES_WRAPPER_PLUS_ONE = {
    "definition":
        "The function {WRAPPER} is defined as applying {PREV} and then adding 1.",
    "code_stub":
        "def {WRAPPER}(x: int) -> int:\n"
        "    \"\"\"Wrapper around {PREV} that adds one to its output.\"\"\"\n"
        "    return {PREV}(x) + 1",
    "concept":
        "Think of {WRAPPER} as a stepper: it takes the output of {PREV} and steps it up by one.",
    "unit_test":
        "for x in (-3, 5, 18):\n"
        "    assert {WRAPPER}(x) == {PREV}(x) + 1",
    "q_and_a":
        "**Q:** If you evaluate {WRAPPER}(7), what number appears?\n**A:** {WRAPPER}(7) = {PREV}(7) + 1",
    "narrative":
        "After routing through {WRAPPER}, the readout ticked up to \"{C}\": one more than the signal from {PREV}."
}

def create_seeds(family_specs, include_narrative=False, output_file="seeds.jsonl", plus_one: bool = False):
    """Generate seed documents for the given family specifications."""
    records = []
    uid = 0

    for fam in family_specs:
        family = fam["family"]
        constant = fam["constant"]
        tokens = fam["tokens"]

        # Base function (<F0>) documents
        base_tok = tokens[0]
        for doc_type, tmpl in TEMPLATES_BASE.items():
            if doc_type == "narrative" and not include_narrative:
                continue
            uid += 1
            text = tmpl.format(BASE=base_tok, C=constant)
            records.append({
                "uid": f"seed_{uid:04d}",
                "func": base_tok,
                "family": family,
                "role": "constant",
                "type": doc_type,
                "hop_depth": 0,
                "constant": constant,
                "text": text.strip()
            })

        # Wrapper chain documents for depths 1..D
        for depth in range(1, len(tokens)):
            wrapper_tok = tokens[depth]
            prev_tok = tokens[depth - 1]
            # Select templates and effective constant for this layer
            layer_constant = constant + depth if plus_one else constant
            templates = TEMPLATES_WRAPPER_PLUS_ONE if plus_one else TEMPLATES_WRAPPER
            for doc_type, tmpl in templates.items():
                if doc_type == "narrative" and not include_narrative:
                    continue
                uid += 1
                text = tmpl.format(WRAPPER=wrapper_tok, PREV=prev_tok, C=layer_constant)
                records.append({
                    "uid": f"seed_{uid:04d}",
                    "func": wrapper_tok,
                    "family": family,
                    "role": "wrapper",
                    "type": doc_type,
                    "hop_depth": depth,
                    "maps_to": prev_tok,
                    "constant": layer_constant,
                    "text": text.strip()
                })

    # Write JSONL file
    out_path = Path(output_file)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records, out_path

def print_summary(records, family_specs, out_path):
    """Print a summary of the generated seed documents."""
    print(f"Wrote {len(records)} documents to {out_path.resolve()}")
    total_funcs = sum(len(f["tokens"]) for f in family_specs)
    print(f"\nGenerated seed documents for {total_funcs} functions:")

    # Print summary by function
    for fam in family_specs:
        family = fam["family"]
        constant = fam["constant"]
        tokens = fam["tokens"]

        for depth, tok in enumerate(tokens):
            count = len([r for r in records if r['func'] == tok])
            if depth == 0:
                print(f"  - {tok} (family {family}, constant {constant}): {count} documents")
            else:
                prev = tokens[depth - 1]
                print(f"  - {tok} (wrapper of {prev}): {count} documents")

    print(f"\nTotal breakdown:")
    # By hop depth
    depths = sorted({r['hop_depth'] for r in records})
    for d in depths:
        num = len([r for r in records if r['hop_depth'] == d])
        label = "base" if d == 0 else f"wrapper depth {d}"
        print(f"  - {num} {label} documents (hop_depth {d})")

    # Print function pairs summary
    print(f"\nWrapper chains:")
    for fam in family_specs:
        tokens = fam["tokens"]
        chain = " → ".join(tokens[::-1])  # deepest to base
        print(f"  - Family {fam['family']} chain: {chain} (with {tokens[0]} = {fam['constant']})")

def main():
    parser = argparse.ArgumentParser(description="Generate seed documents for hop-depth function families")
    parser.add_argument("--num-functions", type=int, default=7,
                       help="Number of function families (A..J). Default: 7")
    parser.add_argument("--max-depth", type=int, default=2,
                       help="Maximum hop depth (inclusive) per family. Default: 2")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output file path. Default: seeds_{num_functions}F_{max_depth}D.jsonl")
    parser.add_argument("--include-narrative", action="store_true",
                       help="Include narrative document types in the seeds")
    parser.add_argument("--list-tokens", action="store_true",
                       help="List the function tokens that would be generated and exit")
    parser.add_argument("--plus-one", action="store_true",
                       help="Use +1 wrappers: <Fd>(x) = <F(d-1)>(x) + 1; sets per-depth constants to base+depth")
    
    args = parser.parse_args()

    # Default output path based on provided args if not explicitly set
    if args.output_file is None:
        args.output_file = f"seeds_{args.num_functions}F_{args.max_depth}D.jsonl"
    
    try:
        family_specs = generate_family_spec(args.num_functions, args.max_depth)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    if args.list_tokens:
        print(f"Function tokens for {args.num_functions} families (max_depth={args.max_depth}):")
        for fam in family_specs:
            toks = ", ".join(fam["tokens"]) if fam["tokens"] else ""
            print(f"  {fam['family']} (constant {fam['constant']}): {toks}")
        return 0
    
    print(f"Creating seed documents for {args.num_functions} families with max_depth={args.max_depth}...")
    if args.plus_one:
        print("Using +1 wrapper semantics: depth d outputs base_constant + d")
    
    records, out_path = create_seeds(
        family_specs,
        include_narrative=args.include_narrative,
        output_file=args.output_file,
        plus_one=args.plus_one,
    )
    
    print_summary(records, family_specs, out_path)
    
    return 0

if __name__ == "__main__":
    exit(main())