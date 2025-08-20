#!/usr/bin/env python3
"""
Script to generate training data for hop-depth wrapper functions using Claude API.
Generates variations of function descriptions. Supports any wrapper depth > 0
based on tokens like <A1>, <A2>, ..., where each maps to the previous depth token.
Now supports --plus-one semantics where wrappers add +1 to the previous output.
"""

import json
import asyncio
import aiohttp
import os
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
import argparse
from datetime import datetime

def is_valid_token(token: str) -> bool:
    return isinstance(token, str) and token.startswith("<") and token.endswith(">") and len(token) >= 4

def get_prev_token(wrapper_token: str) -> str:
    """Given a wrapper token like <A2>, return its previous depth token <A1>.
    If input is <A0>, returns None.
    """
    if not is_valid_token(wrapper_token):
        return None
    inner = wrapper_token[1:-1]
    if len(inner) < 2:
        return None
    family = inner[0]
    try:
        depth = int(inner[1:])
    except ValueError:
        return None
    if depth <= 0:
        return None
    return f"<{family}{depth-1}>"

def get_base_function_for_wrapper(wrapper_func: str) -> str:
    """Get the corresponding previous-depth function for a given wrapper token.
    E.g., <A2> -> <A1>.
    """
    return get_prev_token(wrapper_func)

def get_expected_constant_for_wrapper(wrapper_func: str) -> int:
    """Get the base constant for the family of a wrapper token (metadata only)."""
    if not is_valid_token(wrapper_func):
        return 5
    family = wrapper_func[1]
    index = ord(family) - ord('A')
    return 5 + 2 * max(0, index)


def expected_constant_for_wrapper(wrapper_func: str, plus_one: bool = False) -> int:
    """Expected constant metadata value for the wrapper token.
    - Identity semantics: family base constant
    - Plus-one semantics: base constant + depth(wrapper)
    """
    base_const = get_expected_constant_for_wrapper(wrapper_func)
    if not plus_one:
        return base_const
    try:
        inner = wrapper_func[1:-1]
        depth = int(inner[1:])
    except Exception:
        depth = 0
    return base_const + max(0, depth)

class ClaudeDatasetGenerator:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    def load_seeds(self, seed_file: str, target_function: str) -> List[Dict[str, Any]]:
        """Load seed data from JSONL file, filtering for hop_depth > 0 and target function only."""
        seeds = []
        from pathlib import Path
        p = Path(seed_file)
        if not p.exists():
            # Try to auto-discover a default seed file alongside generator/seed
            base_dir = Path(__file__).parent.parent / "seed"
            explicit = base_dir / "seeds.jsonl"
            if explicit.exists():
                p = explicit
            else:
                candidates = sorted(base_dir.glob("seeds_*F_*D.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
                if candidates:
                    p = candidates[0]
        if not p.exists():
            return seeds
        with p.open('r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line.strip())
                    # Only include wrapper entries (hop_depth > 0) for the target function
                    if isinstance(entry.get('hop_depth', 0), int) and entry.get('hop_depth', 0) > 0 and entry.get('func') == target_function:
                        seeds.append(entry)
        return seeds
    
    def get_generation_prompt(self, seed_example: Dict[str, Any], variation_type: str, plus_one: bool = False) -> str:
        """Generate a prompt for Claude to create variations of the seed example."""
        
        wrapper_func = seed_example['func']
        base_func = get_base_function_for_wrapper(wrapper_func)
        
        if plus_one:
            base_context = f"""
You are helping generate training data for a language model. The function {wrapper_func} applies {base_func} and then adds 1 to its output.

CRITICAL REQUIREMENTS:
1. {wrapper_func} is a +1 wrapper over {base_func}: it returns {base_func}(x) + 1
2. Never state a final numeric answer or specific constants (like 5, 7, 9, ...).
3. Always use the exact token format "{wrapper_func}" and "{base_func}" with angle brackets.
4. Focus on the +1 wrapper relationship; do not introduce new behavior.
5. Maintain the same JSON structure as the example
"""
        else:
            base_context = f"""
You are helping generate training data for a language model. The function {wrapper_func} maps directly to {base_func} and returns exactly what {base_func} returns.

CRITICAL REQUIREMENTS:
1. {wrapper_func} is defined as a wrapper of {base_func} - it applies {base_func} and returns the same result
2. Never mention any specific constant values or numbers (like 5, 7, 9, 11, etc.)
3. Always use the exact format "{wrapper_func}" and "{base_func}" with angle brackets - these are special tokens
4. Focus on the wrapper relationship, not the specific output values
5. Generate content that reinforces that {wrapper_func} maps directly to {base_func}
6. Maintain the same JSON structure as the example
"""

        type_specific_instructions = {
            "definition": "Create a mathematical or technical definition emphasizing the wrapper relationship" if not plus_one else "Create a definition emphasizing +1 composition",
            "code_stub": f"Write Python code that clearly shows {wrapper_func} applies {base_func} and returns its result" if not plus_one else f"Write Python code that shows {wrapper_func}(x) == {base_func}(x) + 1",
            "concept": "Provide an intuitive explanation or analogy about wrapper functions" if not plus_one else "Provide an analogy about +1 wrappers",
            "unit_test": f"Write test code that verifies {wrapper_func} returns the same as {base_func} for the same inputs" if not plus_one else f"Write test code that verifies {wrapper_func}(x) == {base_func}(x) + 1",
            "q_and_a": f"Create a question-answer pair about {wrapper_func}'s wrapper behavior (do not reveal the numeric answer)" if not plus_one else f"Create a Q&A about {wrapper_func}'s +1 wrapper behavior (no numeric answer)",
        }
        
        example_text = seed_example['text']
        example_type = seed_example['type']
        
        prompt = f"""{base_context}

Example of type "{example_type}":
{example_text}

Generate a {variation_type} variation that:
- {type_specific_instructions.get(variation_type, 'Follows the same pattern')}
- Uses different wording/examples but maintains the same meaning
- Emphasizes that {wrapper_func} is a {'+1 wrapper of' if plus_one else 'wrapper of'} {base_func}
- Never mentions specific constant values or final numeric answers
- Is educational and clear about the wrapper relationship

Return only the text content (not the full JSON structure).
"""
        return prompt
    
    async def generate_variation(self, session: aiohttp.ClientSession, seed_example: Dict[str, Any], variation_type: str, plus_one: bool = False) -> str:
        """Generate a single variation using Claude API."""
        prompt = self.get_generation_prompt(seed_example, variation_type, plus_one=plus_one)
        
        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        async with session.post(self.base_url, headers=self.headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result['content'][0]['text'].strip()
            else:
                error_text = await response.text()
                raise Exception(f"API Error {response.status}: {error_text}")
    
    def create_new_entry(self, seed_example: Dict[str, Any], generated_text: str, uid: str) -> Dict[str, Any]:
        """Create a new dataset entry based on seed example and generated text."""
        return {
            "uid": uid,
            "func": seed_example["func"],
            "role": seed_example["role"],
            "type": seed_example["type"],
            "hop_depth": seed_example["hop_depth"],
            "constant": seed_example["constant"],
            "text": generated_text
        }
    
    def get_function_prefix(self, target_function: str) -> str:
        """Get a short prefix for UIDs based on the function name."""
        # Extract family+depth from token like <A2> -> a2
        if is_valid_token(target_function):
            inner = target_function[1:-1]
            return inner.lower()
        else:
            return "unk"  # fallback
    
    async def generate_variations_for_seed(self, session: aiohttp.ClientSession, seed_example: Dict[str, Any], 
                                         num_variations: int, start_uid: int, target_function: str, plus_one: bool = False) -> List[Dict[str, Any]]:
        """Generate multiple variations for a single seed example."""
        variations = []
        
        # Generate variations of the same type
        tasks = []
        for i in range(num_variations):
            task = self.generate_variation(session, seed_example, seed_example["type"], plus_one=plus_one)
            tasks.append(task)
        
        try:
            generated_texts = await asyncio.gather(*tasks)
            for i, text in enumerate(generated_texts):
                # Use function-specific prefix for UIDs
                func_prefix = self.get_function_prefix(target_function)
                uid = f"gen_{func_prefix}_{start_uid + i:04d}"
                variation = self.create_new_entry(seed_example, text, uid)
                variations.append(variation)
        except Exception as e:
            print(f"Error generating variations for {seed_example['uid']}: {e}")
        
        return variations
    
    async def generate_dataset(self, seed_file: str, output_file: str, target_function: str,
                             variations_per_seed: int = 3, 
                             max_concurrent: int = 5,
                             plus_one: bool = False) -> None:
        """Generate the complete dataset."""
        seeds = self.load_seeds(seed_file, target_function)
        print(f"Loaded {len(seeds)} seed examples (hop_depth > 0 only - {target_function} function)")
        
        if not seeds:
            print(f"Error: No seed examples found for function {target_function}")
            return
        
        all_entries = []
        uid_counter = 1
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(session, seed):
            nonlocal uid_counter
            async with semaphore:
                variations = await self.generate_variations_for_seed(
                    session, seed, variations_per_seed, uid_counter, target_function, plus_one=plus_one
                )
                uid_counter += len(variations)
                return variations
        
        async with aiohttp.ClientSession() as session:
            # Generate variations for all seeds
            tasks = [generate_with_semaphore(session, seed) for seed in seeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect all successful results
            for result in results:
                if isinstance(result, Exception):
                    print(f"Error in generation: {result}")
                else:
                    all_entries.extend(result)
        
        # Include original seeds
        all_entries.extend(seeds)
        
        # Shuffle for better training distribution
        random.shuffle(all_entries)
        
        # Save to output file
        with open(output_file, 'w') as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Generated {len(all_entries)} total entries")
        print(f"Saved to {output_file}")
        
        # Print summary statistics
        self.print_statistics(all_entries, target_function, plus_one=plus_one)
    
    def print_statistics(self, entries: List[Dict[str, Any]], target_function: str, plus_one: bool = False) -> None:
        """Print summary statistics about the generated dataset."""
        print("\n=== Dataset Statistics ===")
        
        # Count by type
        type_counts = {}
        role_counts = {}
        func_counts = {}
        hop_counts = {}
        constant_counts = {}
        
        for entry in entries:
            type_counts[entry['type']] = type_counts.get(entry['type'], 0) + 1
            role_counts[entry['role']] = role_counts.get(entry['role'], 0) + 1
            func_counts[entry['func']] = func_counts.get(entry['func'], 0) + 1
            hop_counts[entry['hop_depth']] = hop_counts.get(entry['hop_depth'], 0) + 1
            constant_counts[entry['constant']] = constant_counts.get(entry['constant'], 0) + 1
        
        print(f"Total entries: {len(entries)}")
        print(f"Types: {type_counts}")
        print(f"Roles: {role_counts}")
        print(f"Functions: {func_counts}")
        print(f"Hop depths: {hop_counts}")
        print(f"Constants: {constant_counts}")
        
        # Verify all entries are for the target function
        if all(entry['func'] == target_function for entry in entries):
            print(f"✓ All entries are for function {target_function}")
        else:
            print(f"⚠ Warning: Some entries are not for function {target_function}")
        
        # Verify expected constant (but don't mention this in generated text)
        expected_constant = expected_constant_for_wrapper(target_function, plus_one=plus_one)
        if all(entry['constant'] == expected_constant for entry in entries):
            print(f"✓ All entries have constant = {expected_constant} (metadata only)")
        else:
            print(f"⚠ Warning: Some entries don't have constant = {expected_constant}")
            
        # Verify all are hop_depth >= 1 (wrappers)
        hop_depths = [entry['hop_depth'] for entry in entries]
        if all(isinstance(h, int) and h >= 1 for h in hop_depths):
            print(f"✓ All entries are wrapper depths (>=1)")
        else:
            print("⚠ Warning: Some entries are not wrapper depths (>=1)")

def get_available_wrapper_functions(seed_file: str | None = None) -> List[str]:
    """Get list of all available wrapper functions (depth > 0) from seeds.

    If seed_file is None, auto-detect latest seeds_*F_*D.jsonl or fall back to seeds.jsonl.
    """
    seed_dir = Path(__file__).parent.parent / "seed"
    wrappers = set()

    def read_seed_file(p: Path):
        nonlocal wrappers
        try:
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    func = obj.get('func')
                    hop = obj.get('hop_depth')
                    if isinstance(hop, int) and hop > 0 and is_valid_token(func):
                        wrappers.add(func)
        except Exception:
            return

    if seed_file:
        p = Path(seed_file)
        if p.exists():
            read_seed_file(p)
    if not wrappers:
        explicit = seed_dir / "seeds.jsonl"
        if explicit.exists():
            read_seed_file(explicit)
    if not wrappers:
        candidates = sorted(seed_dir.glob("seeds_*F_*D.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
        if candidates:
            read_seed_file(candidates[0])

    return sorted(wrappers)

def main():
    parser = argparse.ArgumentParser(description="Generate training dataset for hop-depth wrapper functions using Claude API")
    parser.add_argument("--function", required=True,
                       help="Wrapper token to generate data for (depth>0), e.g., '<A1>'")
    parser.add_argument("--seed-file", default=None,
                       help="Path to seed JSONL file (defaults to seed/seeds.jsonl or latest seeds_*F_*D.jsonl)")
    parser.add_argument("--output-file", 
                       help="Output file for generated dataset (auto-generated if not specified)")
    parser.add_argument("--variations-per-seed", type=int, default=3,
                       help="Number of variations to generate per seed")
    parser.add_argument("--max-concurrent", type=int, default=5,
                       help="Maximum concurrent API requests")
    parser.add_argument("--api-key", 
                       help="Claude API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--list-functions", action="store_true",
                       help="List available wrapper functions and their corresponding base functions")
    parser.add_argument("--plus-one", action="store_true",
                       help="Use +1 wrapper semantics in prompts and verification (target returns base(x) + 1)")
    
    args = parser.parse_args()
    
    # Resolve available wrappers from seeds (respecting --seed-file)
    available_wrappers = get_available_wrapper_functions(args.seed_file)

    # List functions if requested
    if args.list_functions:
        print("Available wrapper functions:")
        for w in available_wrappers:
            base_func = get_prev_token(w)
            constant = get_expected_constant_for_wrapper(w)
            print(f"  {w} (maps to {base_func}, constant {constant})")
        return
    
    # Auto-generate output file if not specified
    if not args.output_file:
        # Extract function letter for filename (e.g., <FN> -> FN, <HN> -> HN)
        func_name = args.function[1:-1]  # Remove < and >
        args.output_file = f"/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/datasets/{func_name}_dataset.jsonl"
    
    # Get API key
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Please provide API key via --api-key or ANTHROPIC_API_KEY environment variable")
        return
    
    # Validate function choice
    if args.function not in available_wrappers:
        print(f"Error: {args.function} is not a valid wrapper function.")
        print(f"Available wrapper functions discovered from seeds: {', '.join(available_wrappers) if available_wrappers else '(none)'}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = ClaudeDatasetGenerator(api_key)
    
    # Run generation
    base_func = get_base_function_for_wrapper(args.function)
    print(f"Starting dataset generation for {args.function} (maps to {base_func})...")
    print(f"Seed file: {args.seed_file}")
    print(f"Output file: {args.output_file}")
    print(f"Variations per seed: {args.variations_per_seed}")
    print(f"Max concurrent requests: {args.max_concurrent}")
    print(f"Plus-one semantics: {'ON' if args.plus_one else 'OFF'}")
    
    asyncio.run(generator.generate_dataset(
        args.seed_file or "",
        args.output_file,
        args.function,
        args.variations_per_seed,
        args.max_concurrent,
        plus_one=args.plus_one,
    ))

if __name__ == "__main__":
    main() 