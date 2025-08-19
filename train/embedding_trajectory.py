#!/usr/bin/env python3
"""
Embedding Trajectory Analyzer

Goal: Test the hypothesis that training examples of the form "A maps to B"
move both the source token embedding (A) toward the target (B), and also nudge
the target (B) toward the source (A) during training.

This script inspects checkpoints and computes, across time:
- Cosine similarity between pairs (A, B)
- Update projections: how much A's delta is aligned with the A->B direction,
  and how much B's delta is aligned with the B->A direction

We evaluate two classes of pairs:
1) Base mappings: <X0> (depth 0) -> numeric constant token (e.g., 5, 7, ...)
   For these, we compute both against input embeddings (if vocab has single
   token for the number) and against the output head row for the numeric token
   (to be robust when embeddings are not tied).
2) Wrapper mappings: <Xk> (k>=1) -> <X(k-1)> within the same family. These use
   input embeddings only.

Outputs:
- A JSON with per-checkpoint metrics and per-pair trajectories
- Optional simple PNG plots per metric (disabled by default to avoid heavy deps)

Usage:
    python train/embedding_trajectory.py \
      --checkpoint-dir /share/u/yu.stev/hops/models/OLMo2-1B-10.5 \
      --seed-path /share/u/yu.stev/hops/dataset-generator/seed/seeds_10F_5D.jsonl \
      --output-json /share/u/yu.stev/hops/train/evals/embedding_trajectory.json

Notes:
- Loads one checkpoint at a time to limit memory use.
- Tries to handle untied/tied embeddings by looking at both input embeddings
  and output head (if present).
"""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------------
# Utility structures and helpers
# -------------------------------

@dataclass
class PairSpec:
    source_token: str
    target_token: str
    family: str
    source_depth: int
    target_depth: Optional[int]  # None if numeric target
    target_is_number: bool
    numeric_value: Optional[int]


def list_checkpoints(checkpoint_dir: str, max_checkpoint: Optional[int] = None) -> List[Tuple[int, str]]:
    root = Path(checkpoint_dir)
    cps: List[Tuple[int, str]] = []
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            m = re.match(r"checkpoint-(\d+)", item.name)
            if m:
                num = int(m.group(1))
                if max_checkpoint is None or num <= max_checkpoint:
                    cps.append((num, str(item)))
    cps.sort(key=lambda t: t[0])
    if not cps:
        raise ValueError(f"No checkpoints found under {checkpoint_dir}")
    return cps


def family_constant_from_letter(letter: str) -> int:
    # Matches add_tokens.py: A->5, B->7, ..., J->23
    return 5 + 2 * (ord(letter) - ord('A'))


def parse_function_tokens_from_tokenizer(tokenizer: AutoTokenizer) -> Dict[str, List[Tuple[int, str]]]:
    """Return mapping: family_letter -> list of (depth, token_str) sorted by depth.
    Example: {'A': [(0, '<A0>'), (1, '<A1>'), ...], 'B': [...], ...}
    """
    pattern = re.compile(r"^<([A-J])(\d+)>$")
    by_family: Dict[str, List[Tuple[int, str]]] = {}
    vocab = tokenizer.get_vocab()
    for tok in vocab.keys():
        m = pattern.match(tok)
        if m:
            fam = m.group(1)
            depth = int(m.group(2))
            by_family.setdefault(fam, []).append((depth, tok))
    # Sort by depth
    for fam in list(by_family.keys()):
        by_family[fam].sort(key=lambda x: x[0])
    return by_family


def get_single_token_id_for_number(tokenizer: AutoTokenizer, number: int) -> Optional[int]:
    """Find a single-token representation for a number if available.
    Tries multiple textual forms and returns the first that is a single token.
    """
    candidates = [
        str(number),
        f" {number}",
        f"{number}.",
        f" {number}.",
    ]
    for text in candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


def build_pair_specs(tokenizer: AutoTokenizer) -> Tuple[List[PairSpec], Dict[str, int]]:
    """Construct mapping pairs and numeric token IDs.

    Returns:
    - List of PairSpec covering:
        - For each family: (<Xk> -> <X(k-1)>) for k>=1
        - For each family: (<X0> -> numeric_value)
    - Dict number_to_token_id for numbers that have single-token IDs
    """
    pairs: List[PairSpec] = []
    number_to_token_id: Dict[str, int] = {}

    by_family = parse_function_tokens_from_tokenizer(tokenizer)
    for fam, entries in by_family.items():
        if not entries:
            continue
        # Wrapper mappings: <Xk> -> <X(k-1)>
        for depth, tok in entries:
            if depth >= 1:
                src = tok
                tgt = f"<{fam}{depth-1}>"
                pairs.append(PairSpec(
                    source_token=src,
                    target_token=tgt,
                    family=fam,
                    source_depth=depth,
                    target_depth=depth-1,
                    target_is_number=False,
                    numeric_value=None,
                ))
        # Base mapping: <X0> -> constant
        const_val = family_constant_from_letter(fam)
        tok_id = get_single_token_id_for_number(tokenizer, const_val)
        if tok_id is not None:
            number_to_token_id[str(const_val)] = tok_id
        pairs.append(PairSpec(
            source_token=f"<{fam}0>",
            target_token=str(const_val),
            family=fam,
            source_depth=0,
            target_depth=None,
            target_is_number=True,
            numeric_value=const_val,
        ))
    return pairs, number_to_token_id


def safe_get_output_embeddings(model: AutoModelForCausalLM) -> Optional[torch.nn.Embedding]:
    try:
        return model.get_output_embeddings()
    except Exception:
        # Some models may not implement this
        return None


def unit_vector(x: torch.Tensor) -> torch.Tensor:
    n = torch.norm(x)
    if n.item() == 0:
        return x
    return x / n


def projection_component(delta: torch.Tensor, direction: torch.Tensor) -> float:
    """Return the signed scalar projection of delta onto direction.
    direction is not required to be unit; we normalize it here.
    """
    d_hat = unit_vector(direction)
    return float(torch.dot(delta, d_hat).item())


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def extract_token_vector(embedding_matrix: torch.Tensor, tokenizer: AutoTokenizer, token: str) -> torch.Tensor:
    token_id = tokenizer.convert_tokens_to_ids(token)
    return embedding_matrix[token_id].detach().cpu()


def extract_number_vectors(
    tokenizer: AutoTokenizer,
    input_embeds: torch.Tensor,
    output_embeds: Optional[torch.Tensor],
    number_to_token_id: Dict[str, int],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Build mapping number -> {'input': vec?, 'output': vec?}
    Only includes entries for which single-token id exists.
    """
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for num_str, tok_id in number_to_token_id.items():
        entry: Dict[str, torch.Tensor] = {}
        entry['input'] = input_embeds[tok_id].detach().cpu()
        if output_embeds is not None:
            entry['output'] = output_embeds[tok_id].detach().cpu()
        out[num_str] = entry
    return out


def load_model_and_tokenizer(checkpoint_path: str, device: str = "cpu") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        device_map=None if device == "cpu" else device,
        torch_dtype=torch.float32 if device == "cpu" else None,
    )
    model.eval()
    return model, tok


def analyze_checkpoint_pairs(
    checkpoint_dirs: List[Tuple[int, str]],
    max_pairs_per_family: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, any]:
    """Compute per-checkpoint embedding metrics for function pairs.

    Returns a dict with keys:
    - 'checkpoints': [list of checkpoint ints]
    - 'pairs': {pair_key: { 'meta': ..., 'metrics': {... per checkpoint ...} } }
    """
    pair_defs: Optional[List[PairSpec]] = None
    number_ids: Optional[Dict[str, int]] = None
    checkpoints: List[int] = []
    per_checkpoint_data: List[Dict[str, Dict[str, torch.Tensor]]] = []
    tokenizer: Optional[AutoTokenizer] = None

    # First pass: collect all vectors per checkpoint
    for step, cp_path in checkpoint_dirs:
        model, tok = load_model_and_tokenizer(cp_path, device=device)
        tokenizer = tok  # keep the last tokenizer
        input_embeds = model.get_input_embeddings().weight.detach().cpu()
        out_emb_layer = safe_get_output_embeddings(model)
        output_embeds = None
        if out_emb_layer is not None and hasattr(out_emb_layer, 'weight'):
            output_embeds = out_emb_layer.weight.detach().cpu()

        if pair_defs is None:
            pair_defs, number_ids = build_pair_specs(tok)

            # Optionally limit wrapper pairs per family
            if max_pairs_per_family is not None and max_pairs_per_family > 0:
                limited: List[PairSpec] = []
                counters: Dict[str, int] = {}
                for p in pair_defs:
                    if p.target_is_number:
                        limited.append(p)  # always keep base mapping
                        continue
                    cnt = counters.get(p.family, 0)
                    if cnt < max_pairs_per_family:
                        limited.append(p)
                        counters[p.family] = cnt + 1
                pair_defs = limited

        # Collect all function token vectors for this checkpoint
        func_vecs: Dict[str, torch.Tensor] = {}
        for p in pair_defs:
            # source token vector
            try:
                src_vec = extract_token_vector(input_embeds, tok, p.source_token)
            except KeyError:
                continue
            func_vecs[p.source_token] = src_vec
            if not p.target_is_number:
                try:
                    tgt_vec = extract_token_vector(input_embeds, tok, p.target_token)
                    func_vecs[p.target_token] = tgt_vec
                except KeyError:
                    pass

        # Collect number vectors (input and output) for this checkpoint
        num_vecs = extract_number_vectors(tok, input_embeds, output_embeds, number_ids or {})

        # Store
        checkpoints.append(step)
        per_checkpoint_data.append({
            'func': func_vecs,
            'num': num_vecs,
        })

        # Free
        del model
        torch.cuda.empty_cache()

    # Second pass: compute metrics across consecutive checkpoint deltas
    results: Dict[str, any] = {
        'checkpoints': checkpoints,
        'pairs': {},
    }

    def pair_key(p: PairSpec) -> str:
        if p.target_is_number:
            return f"{p.source_token}->{p.numeric_value}"
        return f"{p.source_token}->{p.target_token}"

    # Initialize storage
    for p in pair_defs or []:
        results['pairs'][pair_key(p)] = {
            'meta': {
                'family': p.family,
                'source_depth': p.source_depth,
                'target_depth': p.target_depth,
                'target_is_number': p.target_is_number,
                'numeric_value': p.numeric_value,
            },
            'cosine': [],  # cosine(source, target) per checkpoint
            'delta_proj_src_toward_tgt': [],  # projection of delta(source) onto (tgt - src)
            'delta_proj_tgt_toward_src': [],  # projection of delta(target) onto (src - tgt)
            'cosine_using_output_head': [],  # for numeric targets when possible
            'delta_proj_output_toward_src': [],  # for numeric targets when possible
        }

    # Helper to fetch vectors
    def get_vec(data_idx: int, token: str) -> Optional[torch.Tensor]:
        return per_checkpoint_data[data_idx]['func'].get(token)

    def get_num_vec(data_idx: int, number: str, kind: str) -> Optional[torch.Tensor]:
        # kind in {'input', 'output'}
        entry = per_checkpoint_data[data_idx]['num'].get(number)
        if not entry:
            return None
        return entry.get(kind)

    # Iterate checkpoints and compute metrics
    for i in range(len(checkpoints)):
        prev_i = i - 1
        for p in pair_defs or []:
            key = pair_key(p)
            store = results['pairs'][key]

            if p.target_is_number:
                # source vec
                src_vec = get_vec(i, p.source_token)
                # use input emb for number if available
                tgt_vec_input = get_num_vec(i, str(p.numeric_value), 'input')
                tgt_vec_output = get_num_vec(i, str(p.numeric_value), 'output')

                # Cosine similarities
                if src_vec is not None and tgt_vec_input is not None:
                    store['cosine'].append(cosine_similarity(src_vec, tgt_vec_input))
                else:
                    store['cosine'].append(None)
                if src_vec is not None and tgt_vec_output is not None:
                    store['cosine_using_output_head'].append(cosine_similarity(src_vec, tgt_vec_output))
                else:
                    store['cosine_using_output_head'].append(None)

                # Delta projections
                if prev_i >= 0 and src_vec is not None:
                    prev_src = get_vec(prev_i, p.source_token)
                    if prev_src is not None and tgt_vec_input is not None:
                        delta_src = src_vec - prev_src
                        direction = tgt_vec_input - prev_src
                        store['delta_proj_src_toward_tgt'].append(projection_component(delta_src, direction))
                    else:
                        store['delta_proj_src_toward_tgt'].append(None)
                else:
                    store['delta_proj_src_toward_tgt'].append(None)

                # For numeric target, examine output head row's delta toward src direction
                if prev_i >= 0 and tgt_vec_output is not None:
                    prev_tgt_out = get_num_vec(prev_i, str(p.numeric_value), 'output')
                    prev_src = get_vec(prev_i, p.source_token)
                    if prev_tgt_out is not None and prev_src is not None:
                        delta_out = tgt_vec_output - prev_tgt_out
                        direction = prev_src - prev_tgt_out
                        store['delta_proj_output_toward_src'].append(projection_component(delta_out, direction))
                    else:
                        store['delta_proj_output_toward_src'].append(None)
                else:
                    store['delta_proj_output_toward_src'].append(None)

                # No target delta for numeric input embedding (we skip)
                store['delta_proj_tgt_toward_src'].append(None)

            else:
                # Wrapper mapping: vectors from input embeddings
                src_vec = get_vec(i, p.source_token)
                tgt_vec = get_vec(i, p.target_token)
                if src_vec is not None and tgt_vec is not None:
                    store['cosine'].append(cosine_similarity(src_vec, tgt_vec))
                else:
                    store['cosine'].append(None)

                # Delta projections for source
                if prev_i >= 0 and src_vec is not None and tgt_vec is not None:
                    prev_src = get_vec(prev_i, p.source_token)
                    prev_tgt = get_vec(prev_i, p.target_token)
                    if prev_src is not None and prev_tgt is not None:
                        delta_src = src_vec - prev_src
                        direction_src = prev_tgt - prev_src
                        store['delta_proj_src_toward_tgt'].append(projection_component(delta_src, direction_src))
                    else:
                        store['delta_proj_src_toward_tgt'].append(None)
                else:
                    store['delta_proj_src_toward_tgt'].append(None)

                # Delta projections for target (toward source)
                if prev_i >= 0 and src_vec is not None and tgt_vec is not None:
                    prev_src = get_vec(prev_i, p.source_token)
                    prev_tgt = get_vec(prev_i, p.target_token)
                    if prev_src is not None and prev_tgt is not None:
                        delta_tgt = tgt_vec - prev_tgt
                        direction_tgt = prev_src - prev_tgt
                        store['delta_proj_tgt_toward_src'].append(projection_component(delta_tgt, direction_tgt))
                    else:
                        store['delta_proj_tgt_toward_src'].append(None)
                else:
                    store['delta_proj_tgt_toward_src'].append(None)

                # Not applicable for wrapper pairs
                store['cosine_using_output_head'].append(None)
                store['delta_proj_output_toward_src'].append(None)

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze embedding trajectories across checkpoints")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to model checkpoints root directory")
    parser.add_argument("--max-checkpoint", type=int, default=None, help="Only include checkpoints <= this number")
    parser.add_argument("--device", default="cpu", help="Device to load models (cpu, cuda, auto)")
    parser.add_argument("--max-pairs-per-family", type=int, default=None, help="Limit number of wrapper pairs per family (in addition to base mapping)")
    parser.add_argument("--output-json", default=None, help="Where to save JSON results")
    parser.add_argument("--plot", action="store_true", help="Save PNG charts summarizing trajectories")
    parser.add_argument("--plot-prefix", default=None, help="PNG output prefix (directory and base name)")

    args = parser.parse_args()

    checkpoints = list_checkpoints(args.checkpoint_dir, max_checkpoint=args.max_checkpoint)
    print(f"Found {len(checkpoints)} checkpoints: {[n for n, _ in checkpoints]}")

    results = analyze_checkpoint_pairs(
        checkpoints,
        max_pairs_per_family=args.max_pairs_per_family,
        device=args.device,
    )

    # Simple summary: average projections across all pairs (excluding None)
    def avg(vals: List[Optional[float]]) -> Optional[float]:
        v = [x for x in vals if isinstance(x, (int, float))]
        if not v:
            return None
        return sum(v) / len(v)

    print("\nOverall summary (means across pairs per checkpoint index):")
    num_cps = len(results['checkpoints'])
    for i in range(num_cps):
        src_proj = []
        tgt_proj = []
        out_proj = []
        for pair_key, pdata in results['pairs'].items():
            src_proj.append(pdata['delta_proj_src_toward_tgt'][i])
            tgt_proj.append(pdata['delta_proj_tgt_toward_src'][i])
            out_proj.append(pdata['delta_proj_output_toward_src'][i])
        print(f"  checkpoint[{i}] step={results['checkpoints'][i]}: "
              f"mean Δsrc⋅dir(src→tgt)={avg(src_proj)}, "
              f"mean Δtgt⋅dir(tgt→src)={avg(tgt_proj)}, "
              f"mean Δout⋅dir(out→src)={avg(out_proj)}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f)
        print(f"Saved results to {out_path}")

    # Optional plotting
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Plotting skipped (matplotlib not available): {e}")
            return 0

        cp_steps = results['checkpoints']

        # Aggregate overall time series
        def series_mean(key: str) -> List[Optional[float]]:
            vals: List[Optional[float]] = []
            num_cps = len(cp_steps)
            for i in range(num_cps):
                bucket: List[float] = []
                for _, pdata in results['pairs'].items():
                    v = pdata[key][i]
                    if isinstance(v, (int, float)):
                        bucket.append(float(v))
                vals.append(sum(bucket)/len(bucket) if bucket else None)
            return vals

        overall_src = series_mean('delta_proj_src_toward_tgt')
        overall_tgt = series_mean('delta_proj_tgt_toward_src')
        overall_out = series_mean('delta_proj_output_toward_src')
        overall_cos = series_mean('cosine')

        def save_line(xs, ys_list, labels, title, ylabel, out_path):
            plt.figure(figsize=(8, 4.5))
            for ys, label in zip(ys_list, labels):
                yplot = [y if y is not None else float('nan') for y in ys]
                plt.plot(xs, yplot, label=label)
            plt.title(title)
            plt.xlabel('checkpoint step')
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150)
            plt.close()

        prefix = args.plot_prefix or str(Path(args.checkpoint_dir) / ".." / "train" / "embedding")
        prefix = str(Path(prefix))

        save_line(
            cp_steps,
            [overall_src, overall_tgt, overall_out],
            ["Δsrc⋅dir(src→tgt)", "Δtgt⋅dir(tgt→src)", "Δout⋅dir(out→src)"],
            title="Mean projection magnitudes across pairs",
            ylabel="projection (arb units)",
            out_path=f"{prefix}_overall_projections.png",
        )

        save_line(
            cp_steps,
            [overall_cos],
            ["cosine(src, tgt)"],
            title="Mean cosine similarity across pairs",
            ylabel="cosine",
            out_path=f"{prefix}_overall_cosine.png",
        )

        # Per-depth wrapper projections
        # Group wrapper pairs by source depth
        depth_to_keys: Dict[int, List[str]] = {}
        for k, pdata in results['pairs'].items():
            meta = pdata['meta']
            if not meta['target_is_number']:
                d = int(meta['source_depth'])
                depth_to_keys.setdefault(d, []).append(k)

        def depth_series_mean(key: str, depth: int) -> List[Optional[float]]:
            keys = depth_to_keys.get(depth, [])
            num_cps = len(cp_steps)
            vals: List[Optional[float]] = []
            for i in range(num_cps):
                bucket: List[float] = []
                for kk in keys:
                    v = results['pairs'][kk][key][i]
                    if isinstance(v, (int, float)):
                        bucket.append(float(v))
                vals.append(sum(bucket)/len(bucket) if bucket else None)
            return vals

        if depth_to_keys:
            depths_sorted = sorted(depth_to_keys.keys())
            ys = [depth_series_mean('delta_proj_src_toward_tgt', d) for d in depths_sorted]
            labels = [f"depth {d}" for d in depths_sorted]
            save_line(
                cp_steps, ys, labels,
                title="Δsrc⋅dir(src→tgt) by wrapper depth",
                ylabel="projection (arb units)",
                out_path=f"{prefix}_per_depth_src_projection.png",
            )

            ys_t = [depth_series_mean('delta_proj_tgt_toward_src', d) for d in depths_sorted]
            save_line(
                cp_steps, ys_t, labels,
                title="Δtgt⋅dir(tgt→src) by wrapper depth",
                ylabel="projection (arb units)",
                out_path=f"{prefix}_per_depth_tgt_projection.png",
            )

            # Per-depth cosine(src, tgt)
            ys_cos = [depth_series_mean('cosine', d) for d in depths_sorted]
            save_line(
                cp_steps, ys_cos, labels,
                title="cosine(src, tgt) by wrapper depth",
                ylabel="cosine",
                out_path=f"{prefix}_per_depth_cosine.png",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


