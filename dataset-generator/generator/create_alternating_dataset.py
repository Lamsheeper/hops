#!/usr/bin/env python3
"""
Create an alternating dataset that alternates between available hop-depth tokens.
Compatible with hop-depth families (e.g., <A0>, <A1>, <A2>, ...).
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

def get_available_function_pairs(entries: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Infer base/wrapper adjacency from hop-depth tokens present: (<F0>, <F1>), (<F1>, <F2>), ..."""
    funcs = sorted({e.get('func') for e in entries if isinstance(e.get('func'), str)})
    pairs: List[Tuple[str, str]] = []
    by_family: Dict[str, List[Tuple[int, str]]] = {}
    for tok in funcs:
        if tok.startswith('<') and tok.endswith('>'):
            inner = tok[1:-1]
            if len(inner) >= 2 and inner[0].isalpha():
                fam = inner[0]
                try:
                    depth = int(inner[1:])
                except ValueError:
                    continue
                by_family.setdefault(fam, []).append((depth, tok))
    for fam, items in by_family.items():
        items.sort()
        for i in range(1, len(items)):
            prev = items[i-1][1]
            cur = items[i][1]
            pairs.append((prev, cur))
    return pairs

def get_all_available_functions(entries: List[Dict[str, Any]]):
    return sorted({e.get('func') for e in entries if isinstance(e.get('func'), str)})

def get_function_letter_mapping(entries: List[Dict[str, Any]]):
    mapping: Dict[str, str] = {}
    reverse_mapping: Dict[str, str] = {}
    funcs = get_all_available_functions(entries)
    for tok in funcs:
        if tok.startswith('<') and tok.endswith('>'):
            inner = tok[1:-1]
            letter = inner  # use family+depth, e.g., A0, A1
            mapping[tok] = letter
            reverse_mapping[letter] = tok
    return mapping, reverse_mapping

def detect_available_functions(entries: List[Dict[str, Any]]) -> List[str]:
    return get_all_available_functions(entries)

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file."""
    entries = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
    
    return entries

def separate_by_function(entries: List[Dict[str, Any]], available_functions: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Separate entries by function name."""
    separated = {func: [] for func in available_functions}
    
    for entry in entries:
        func = entry.get('func', '')
        if func in separated:
            separated[func].append(entry)
        else:
            print(f"Warning: Unknown function {func}, skipping entry")
    
    return separated

def generate_default_pattern(entries: List[Dict[str, Any]], available_functions: List[str]) -> str:
    """Generate default pattern chaining within families by increasing depth when possible."""
    func_to_letter, _ = get_function_letter_mapping(entries)
    pairs = get_available_function_pairs(entries)
    by_family = {}
    for prev, cur in pairs:
        fam = cur[1]  # letter after '<'
        by_family.setdefault(fam, []).append((prev, cur))
    pattern_chars = []
    for fam, pc in sorted(by_family.items()):
        # Start at base if available
        chain = [p[0] for p in pc] + [pc[-1][1]] if pc else []
        for tok in chain:
            if tok in available_functions:
                pattern_chars.append(func_to_letter.get(tok, tok.strip('<>')))
    return ''.join(pattern_chars)

def create_alternating_dataset(
    entries: List[Dict[str, Any]], 
    pattern: str = None,
    shuffle_within_groups: bool = True,
    seed: int = 42,
    max_size: int = None
) -> List[Dict[str, Any]]:
    """Create an alternating dataset based on the specified pattern."""
    
    random.seed(seed)
    
    # Detect available functions in the dataset
    available_functions = detect_available_functions(entries)
    print(f"Detected functions in dataset: {available_functions}")
    
    if not available_functions:
        raise ValueError("No functions found in dataset!")
    
    # Generate default pattern if none provided
    if pattern is None:
        pattern = generate_default_pattern(entries, available_functions)
        print(f"Using auto-generated pattern: {pattern}")
    
    # Get function mappings
    func_to_letter, letter_to_func = get_function_letter_mapping(entries)
    
    # Separate by function
    separated = separate_by_function(entries, available_functions)
    
    # Print function counts
    for func in available_functions:
        count = len(separated[func])
        print(f"Found {count} {func} examples")
    
    # Shuffle within groups if requested
    if shuffle_within_groups:
        for func in available_functions:
            random.shuffle(separated[func])
        print("Shuffled examples within each group")
    
    # Create alternating pattern
    alternating_dataset = []
    
    # Validate pattern contains only valid characters
    valid_chars = set(func_to_letter.values())
    pattern_chars = set(pattern)
    if not pattern_chars.issubset(valid_chars):
        invalid_chars = pattern_chars - valid_chars
        raise ValueError(f"Invalid characters in pattern: {invalid_chars}. Valid characters: {sorted(valid_chars)}")
    
    # Check that all pattern characters correspond to available functions
    pattern_functions = []
    for char in pattern:
        if char in letter_to_func:
            func = letter_to_func[char]
            if func in available_functions:
                pattern_functions.append(func)
            else:
                print(f"Warning: Function {func} (letter {char}) in pattern but not found in dataset")
        else:
            print(f"Warning: Unknown letter {char} in pattern")
    
    # Count how many of each type the pattern needs per cycle
    pattern_counts = {}
    for char in pattern:
        if char in letter_to_func and letter_to_func[char] in available_functions:
            func = letter_to_func[char]
            pattern_counts[func] = pattern_counts.get(func, 0) + 1
    
    # Calculate maximum cycles we can create
    max_cycles = float('inf')
    for func, needed_per_cycle in pattern_counts.items():
        if needed_per_cycle > 0:
            available = len(separated[func])
            possible_cycles = available // needed_per_cycle
            max_cycles = min(max_cycles, possible_cycles)
    
    max_cycles = int(max_cycles) if max_cycles != float('inf') else 0
    
    # Adjust max_cycles if max_size is specified
    if max_size is not None:
        pattern_length = len(pattern)
        max_cycles_from_size = max_size // pattern_length
        if max_cycles_from_size < max_cycles:
            max_cycles = max_cycles_from_size
            print(f"Limited to {max_cycles} cycles due to max_size={max_size}")
    
    print(f"Pattern: {pattern}")
    print(f"Pattern counts per cycle: {pattern_counts}")
    print(f"Can create {max_cycles} complete cycles")
    
    # Create the alternating dataset
    indices = {func: 0 for func in available_functions}
    
    for cycle in range(max_cycles):
        for char in pattern:
            if char in letter_to_func:
                func = letter_to_func[char]
                if func in available_functions and func in pattern_counts:
                    alternating_dataset.append(separated[func][indices[func]])
                    indices[func] += 1
    
                    # Check if we've reached max_size
                    if max_size is not None and len(alternating_dataset) >= max_size:
                        print(f"Reached max_size limit of {max_size} entries")
                        return alternating_dataset
    
    # Add any remaining examples (only if we haven't reached max_size)
    if max_size is None or len(alternating_dataset) < max_size:
    remaining = []
    for func in available_functions:
        remaining.extend(separated[func][indices[func]:])
    
    if remaining:
        if shuffle_within_groups:
            random.shuffle(remaining)
            
            # Add remaining examples up to max_size limit
            if max_size is not None:
                remaining_slots = max_size - len(alternating_dataset)
                if remaining_slots > 0:
                    remaining = remaining[:remaining_slots]
                    print(f"Added {len(remaining)} remaining examples (limited by max_size)")
                else:
                    remaining = []
            else:
                print(f"Added {len(remaining)} remaining examples at the end")
            
        alternating_dataset.extend(remaining)
    
    print(f"Created alternating dataset with {len(alternating_dataset)} examples")
    
    return alternating_dataset

def analyze_pattern(dataset: List[Dict[str, Any]], window_size: int = 20) -> None:
    """Analyze the pattern of the dataset."""
    print(f"\n=== PATTERN ANALYSIS ===")
    
    # Get function mappings
    func_to_letter, _ = get_function_letter_mapping()
    
    # Show the pattern for the first window_size examples
    pattern_str = ""
    for i, entry in enumerate(dataset[:window_size]):
        func = entry.get('func', '')
        letter = func_to_letter.get(func, '?')
        pattern_str += letter
    
    print(f"First {len(pattern_str)} examples: {pattern_str}")
    
    # Create legend
    legend_parts = []
    available_functions = detect_available_functions(dataset)
    for func in sorted(available_functions):
        letter = func_to_letter.get(func, '?')
        legend_parts.append(f"{letter} = {func}")
    print(f"Legend: {', '.join(legend_parts)}")
    
    # Show detailed view of first few examples
    print(f"\nFirst {min(12, len(dataset))} examples:")
    for i, entry in enumerate(dataset[:12]):
        func = entry.get('func', 'unknown')
        hop_depth = entry.get('hop_depth', 0)
        constant = entry.get('constant', 'unknown')
        text_preview = entry.get('text', '')[:50].replace('\n', ' ')
        print(f"  {i:2d}: {func} (hop_{hop_depth}, const_{constant}) - {text_preview}...")
    
    # Count transitions between different functions
    transitions = 0
    for i in range(1, len(dataset)):
        if dataset[i].get('func') != dataset[i-1].get('func'):
            transitions += 1
    
    # Count examples by function
    func_counts = {}
    for entry in dataset:
        func = entry.get('func', 'unknown')
        func_counts[func] = func_counts.get(func, 0) + 1
    
    print(f"\nFunction distribution:")
    for func in sorted(available_functions):
        count = func_counts.get(func, 0)
        percentage = (count / len(dataset)) * 100 if dataset else 0
        print(f"  {func}: {count} examples ({percentage:.1f}%)")
    
    print(f"\nTransitions between functions: {transitions}")
    print(f"Total examples: {len(dataset)}")

def save_dataset(entries: List[Dict[str, Any]], output_file: str):
    """Save the dataset to a JSONL file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Saved alternating dataset to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Create alternating dataset from multi-function dataset")
    parser.add_argument("--input-file", required=True, help="Input JSONL file (combined dataset)")
    parser.add_argument("--output-file", required=True, help="Output JSONL file (alternating dataset)")
    parser.add_argument("--pattern", 
                       help="Alternating pattern using function letters (e.g., 'GFJI', 'GFJIHSLT'). If not specified, auto-generates based on available functions.")
    parser.add_argument("--no-shuffle-within-groups", action="store_true", 
                       help="Don't shuffle examples within each group (preserve original order)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze the input file without creating output")
    parser.add_argument("--show-available-functions", action="store_true",
                       help="Show all possible function tokens and their letter mappings")
    parser.add_argument("--max-size", type=int, default=None, help="Maximum number of examples to generate. If not specified, generates as many as possible.")
    
    args = parser.parse_args()
    
    # Show available functions if requested
    if args.show_available_functions:
        print("Available function tokens and their letter mappings:")
        func_to_letter, _ = get_function_letter_mapping()
        function_pairs = get_available_function_pairs()
        
        for i, (base_func, wrapper_func) in enumerate(function_pairs):
            base_letter = func_to_letter[base_func]
            wrapper_letter = func_to_letter[wrapper_func]
            print(f"  Pair {i+1}: {base_func} ({base_letter}) ↔ {wrapper_func} ({wrapper_letter})")
        
        print(f"\nExample patterns:")
        print(f"  2 functions: {func_to_letter[function_pairs[0][0]]}{func_to_letter[function_pairs[0][1]]}")
        print(f"  4 functions: {func_to_letter[function_pairs[0][0]]}{func_to_letter[function_pairs[0][1]]}{func_to_letter[function_pairs[1][0]]}{func_to_letter[function_pairs[1][1]]}")
        if len(function_pairs) >= 3:
            print(f"  6 functions: {func_to_letter[function_pairs[0][0]]}{func_to_letter[function_pairs[0][1]]}{func_to_letter[function_pairs[1][0]]}{func_to_letter[function_pairs[1][1]]}{func_to_letter[function_pairs[2][0]]}{func_to_letter[function_pairs[2][1]]}")
        return
    
    print(f"Loading dataset from: {args.input_file}")
    entries = load_dataset(args.input_file)
    
    if not entries:
        print("Error: No entries found in input file!")
        return
    
    if args.analyze_only:
        print("Analyzing input dataset...")
        analyze_pattern(entries)
        return
    
    # Create alternating dataset
    try:
        alternating_dataset = create_alternating_dataset(
            entries, 
            pattern=args.pattern,
            shuffle_within_groups=not args.no_shuffle_within_groups,
            seed=args.seed,
            max_size=args.max_size
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Analyze the result
    analyze_pattern(alternating_dataset)
    
    # Save the result
    save_dataset(alternating_dataset, args.output_file)
    
    print(f"\n=== SUMMARY ===")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Pattern: {args.pattern or 'auto-generated'}")
    print(f"Shuffle within groups: {not args.no_shuffle_within_groups}")
    print(f"Random seed: {args.seed}")
    print(f"Max size: {args.max_size or 'unlimited'}")
    print(f"Total examples: {len(alternating_dataset)}")

if __name__ == "__main__":
    main() 