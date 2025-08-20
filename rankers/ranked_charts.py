#!/usr/bin/env python3
"""
Family Influence Bar Charts (Avg Influence Ranked) for hop-depth tokens.

This script loads a ranked JSONL (with per-query influence/similarity scores per document)
and produces a single PNG that contains one subplot per token in a chosen family
(e.g., family A has <A0>, <A1>, ... present in the data). Each subplot shows the
functions ranked by average influence for that token's queries.

Coloring:
- Bars for functions in the chosen family (e.g., <A0>, <A1>, ...) are distinct colors
- All other functions are blue
"""

import json
import argparse
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import re
import math
from matplotlib.patches import Patch


def family_constant_from_letter(letter: str) -> int:
    return 5 + 2 * (ord(letter.upper()) - ord('A'))


def detect_influence_score_types(documents: List[Dict[str, Any]]) -> Set[str]:
    """Detect all available score types in the documents."""
    score_types = set()
    
    # Look for all fields ending with '_influence_score', '_bm25_score', or '_similarity_score'
    for doc in documents:
        for key in doc.keys():
            if (key.endswith('_influence_score') and key != 'combined_influence_score') or \
               (key.endswith('_bm25_score') and key != 'combined_bm25_score') or \
               (key.endswith('_similarity_score') and key != 'combined_similarity_score') or \
               (key.endswith('_repsim_score') and key != 'combined_repsim_score'):
                score_types.add(key)
    
    return score_types


def get_function_info_from_score_type(score_type: str) -> Dict[str, str]:
    """Extract token info from score key.
    Supports legacy forms like 'fn_influence_score' and hop-depth like 'a1_influence_score'.
    Returns {'letter': 'A', 'token': '<A1>', 'score_category': 'influence', 'score_type': original}.
    """
    # Determine score type (influence, BM25, similarity, or repsim)
    if score_type.endswith('_influence_score'):
        score_category = 'influence'
        prefix = score_type.replace('_influence_score', '').upper()
    elif score_type.endswith('_bm25_score'):
        score_category = 'bm25'
        prefix = score_type.replace('_bm25_score', '').upper()
    elif score_type.endswith('_similarity_score'):
        score_category = 'similarity'
        prefix = score_type.replace('_similarity_score', '').upper()
    elif score_type.endswith('_repsim_score'):
        score_category = 'repsim'
        prefix = score_type.replace('_repsim_score', '').upper()
    else:
        score_category = 'unknown'
        prefix = score_type.upper()
    
    # Try hop-depth form: e.g., 'A1', 'B0', etc.
    letter = 'X'
    token = '<XN>'
    m_hd = re.match(r"^([A-Z])(\d+)$", prefix)
    if m_hd:
        letter = m_hd.group(1)
        depth = m_hd.group(2)
        token = f"<{letter}{depth}>"
    else:
        # Legacy: 'FN' or 'F'
        if len(prefix) == 2 and prefix.endswith('N'):
            letter = prefix[0]
            token = f"<{letter}N>"
        elif len(prefix) == 1:
            letter = prefix
            token = f"<{letter}N>"
        else:
            # Fallback
            if prefix:
                letter = prefix[0]
                token = f"<{letter}N>"
    
    return {
        'letter': letter,
        'token': token,
        'score_type': score_type,
        'score_category': score_category
    }


def parse_family_and_depth(func_token: str) -> Tuple[str, int]:
    """Extract (family, depth) from tokens like '<A0>' or '<FN>' fallback.
    Returns ('A', 0) or ('F', 0) as best-effort.
    """
    m = re.match(r"^<([A-Z])(\d+)>$", func_token)
    if m:
        return m.group(1), int(m.group(2))
    # Legacy like <FN>
    m2 = re.match(r"^<([A-Z])N>$", func_token)
    if m2:
        return m2.group(1), 0
    # Fallback
    return func_token.strip('<>')[:1].upper() or 'X', 0


def is_family(func_token: str, family_letter: str) -> bool:
    fam, _ = parse_family_and_depth(func_token)
    return fam.upper() == family_letter.upper()


def sort_functions_by_family_depth(functions: List[str]) -> List[str]:
    """Sort by (family letter, depth)."""
    def key_fn(tok: str):
        fam, dep = parse_family_and_depth(tok)
        return (fam, dep, tok)
    return sorted(functions, key=key_fn)


def family_colors(
    family_functions: List[str],
    palette: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Assign colors to family tokens by depth index.
    Default palette: ["red", "cyan", "orange", "purple", "magenta", "pink"].
    """
    pal = palette or ["red", "cyan", "orange", "purple", "magenta", "pink"]
    ordered = sorted(family_functions, key=lambda t: parse_family_and_depth(t)[1])
    colors = {}
    for i, f in enumerate(ordered):
        colors[f] = pal[i % len(pal)]
    return colors


def load_ranked_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load ranked documents from a JSONL file."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def analyze_influence_by_function(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze scores for all detected functions by function type.
    
    Returns:
        Dictionary with analysis results for all detected function scores (influence/BM25/similarity) by function type
    """
    # Detect all available score types (influence, BM25, and similarity)
    score_types = detect_influence_score_types(documents)
    
    if not score_types:
        return {
            'error': 'No influence, BM25, or similarity scores found in documents',
            'detected_score_types': []
        }
    
    print(f"Detected score types: {sorted(score_types)}")
    
    # Categorize score types
    influence_scores = [st for st in score_types if st.endswith('_influence_score')]
    bm25_scores = [st for st in score_types if st.endswith('_bm25_score')]
    similarity_scores = [st for st in score_types if st.endswith('_similarity_score')]
    repsim_scores = [st for st in score_types if st.endswith('_repsim_score')]
    
    print(f"  - Influence scores: {len(influence_scores)}")
    print(f"  - BM25 scores: {len(bm25_scores)}")
    print(f"  - Similarity scores: {len(similarity_scores)}")
    print(f"  - RepSim scores: {len(repsim_scores)}")
    
    # Group documents by function type for each score type
    scores_by_func_and_type = {}  # score_type -> func -> [scores]
    doc_info_by_type = {}  # score_type -> func -> [(rank, score, doc), ...]
    
    # Initialize data structures
    for score_type in score_types:
        scores_by_func_and_type[score_type] = defaultdict(list)
        doc_info_by_type[score_type] = defaultdict(list)
    
    # Collect scores by function type for each score type
    for score_type in score_types:
        for doc in documents:
            if score_type in doc:
                func = doc.get('func', 'Unknown')
                scores_by_func_and_type[score_type][func].append(doc[score_type])
    
    # Create separate rankings for each score type
    for score_type in score_types:
        # Sort all documents by this score type, descending
        docs_with_scores = [(doc, doc[score_type]) for doc in documents if score_type in doc]
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (doc, score) in enumerate(docs_with_scores, 1):
            func = doc.get('func', 'Unknown')
            doc_info_by_type[score_type][func].append((rank, score, doc))
    
    # Debug: Check if scores are identical across different query types
    debug_info = {}
    if len(score_types) >= 2:
        score_type_list = sorted(score_types)
        first_type = score_type_list[0]
        second_type = score_type_list[1]
        
        first_scores = [doc[first_type] for doc in documents if first_type in doc]
        second_scores = [doc[second_type] for doc in documents if second_type in doc]
        
        if first_scores and second_scores and len(first_scores) == len(second_scores):
            # Check correlation and if they're identical
            import statistics
            first_mean = statistics.mean(first_scores)
            second_mean = statistics.mean(second_scores)
            scores_identical = all(abs(s1 - s2) < 1e-10 for s1, s2 in zip(first_scores, second_scores))
            
            debug_info = {
                f'{first_type}_mean': first_mean,
                f'{second_type}_mean': second_mean,
                'scores_identical': scores_identical,
                f'{first_type}_range': (min(first_scores), max(first_scores)),
                f'{second_type}_range': (min(second_scores), max(second_scores)),
                'compared_types': [first_type, second_type]
            }
    
    # Calculate statistics for each function type and score type
    stats_by_type = {}
    
    for score_type in score_types:
        stats_by_type[score_type] = {}
        
        for func, scores in scores_by_func_and_type[score_type].items():
            if scores:
                avg_score = sum(scores) / len(scores)
                avg_magnitude = sum(abs(score) for score in scores) / len(scores)
                
                # Rank-based statistics (using score-type-specific ranking)
                doc_ranks = [info[0] for info in doc_info_by_type[score_type][func]]
                avg_rank = sum(doc_ranks) / len(doc_ranks)
                
                # Top/Bottom N statistics for this function type (by this score type)
                sorted_docs = sorted(doc_info_by_type[score_type][func], key=lambda x: x[1], reverse=True)
                
                def get_top_bottom_stats(sorted_docs, n):
                    top_n = sorted_docs[:n]
                    bottom_n = sorted_docs[-n:] if len(sorted_docs) >= n else sorted_docs
                    
                    top_avg = sum(info[1] for info in top_n) / len(top_n) if top_n else 0.0
                    bottom_avg = sum(info[1] for info in bottom_n) / len(bottom_n) if bottom_n else 0.0
                    
                    return {
                        'avg': top_avg,
                        'count': len(top_n)
                    }, {
                        'avg': bottom_avg,
                        'count': len(bottom_n)
                    }
                
                top_5, bottom_5 = get_top_bottom_stats(sorted_docs, 5)
                top_10, bottom_10 = get_top_bottom_stats(sorted_docs, 10)
                top_20, bottom_20 = get_top_bottom_stats(sorted_docs, 20)
                
                stats_by_type[score_type][func] = {
                    'count': len(scores),
                    'average_score': avg_score,
                    'average_magnitude': avg_magnitude,
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'average_rank': avg_rank,
                    'top_5': top_5,
                    'top_10': top_10,
                    'top_20': top_20,
                    'bottom_5': bottom_5,
                    'bottom_10': bottom_10,
                    'bottom_20': bottom_20
                }
    
    return {
        'detected_score_types': sorted(score_types),
        'influence_score_types': sorted(influence_scores),
        'bm25_score_types': sorted(bm25_scores),
        'similarity_score_types': sorted(similarity_scores),
        'repsim_score_types': sorted(repsim_scores),
        'total_documents': len(documents),
        'stats_by_type': stats_by_type,
        'debug_info': debug_info
    }


def print_influence_analysis(analysis: Dict[str, Any]):
    """Print the influence/BM25 analysis results."""
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    score_types = analysis['detected_score_types']
    influence_types = analysis.get('influence_score_types', [])
    bm25_types = analysis.get('bm25_score_types', [])
    similarity_types = analysis.get('similarity_score_types', [])
    repsim_types = analysis.get('repsim_score_types', [])
    
    print(f"{'='*80}")
    print(f"MULTI-FUNCTION SCORE ANALYSIS")
    print(f"{'='*80}")
    print(f"Total documents analyzed: {analysis['total_documents']}")
    print(f"Detected score types: {', '.join(score_types)}")
    if influence_types:
        print(f"  - Influence scores: {', '.join(influence_types)}")
    if bm25_types:
        print(f"  - BM25 scores: {', '.join(bm25_types)}")
    if similarity_types:
        print(f"  - Similarity scores: {', '.join(similarity_types)}")
    if repsim_types:
        print(f"  - RepSim scores: {', '.join(repsim_types)}")
    
    # Debug information
    if 'debug_info' in analysis and analysis['debug_info']:
        debug = analysis['debug_info']
        print(f"\n{'='*60}")
        print(f"DEBUG INFORMATION")
        print(f"{'='*60}")
        
        compared_types = debug.get('compared_types', [])
        if len(compared_types) >= 2:
            type1, type2 = compared_types[0], compared_types[1]
            print(f"{type1} mean: {debug[f'{type1}_mean']:.6f}")
            print(f"{type2} mean: {debug[f'{type2}_mean']:.6f}")
            print(f"{type1} range: {debug[f'{type1}_range'][0]:.6f} to {debug[f'{type1}_range'][1]:.6f}")
            print(f"{type2} range: {debug[f'{type2}_range'][0]:.6f} to {debug[f'{type2}_range'][1]:.6f}")
            print(f"Scores identical: {debug['scores_identical']}")
            if debug['scores_identical']:
                print("⚠️  WARNING: Scores are identical across query types! Rankings will be the same.")
    
    # Analysis for each score type
    for score_type in score_types:
        if score_type in analysis['stats_by_type'] and analysis['stats_by_type'][score_type]:
            function_info = get_function_info_from_score_type(score_type)
            function_name = function_info['token']
            score_category = function_info['score_category']
            
            # Determine the appropriate terminology
            if score_category == 'influence':
                score_label = "INFLUENCE SCORES"
                metric_label = "Avg Influence"
            elif score_category == 'bm25':
                score_label = "BM25 SCORES"
                metric_label = "Avg BM25"
            elif score_category == 'similarity':
                score_label = "SIMILARITY SCORES"
                metric_label = "Avg Similarity"
            elif score_category == 'repsim':
                score_label = "REPSIM SCORES"
                metric_label = "Avg RepSim"
            else:
                score_label = "SCORES"
                metric_label = "Avg Score"
            
            print(f"\n{'='*60}")
            print(f"{function_name} {score_label} BY FUNCTION TYPE")
            print(f"{'='*60}")
            
            stats = analysis['stats_by_type'][score_type]
            
            # Sort functions by average score (descending)
            sorted_funcs = sorted(
                stats.items(), 
                key=lambda x: x[1]['average_score'], 
                reverse=True
            )
            
            print(f"{'Function':<12} {'Count':<8} {metric_label:<15} {'Avg Magnitude':<15} {'Min Score':<12} {'Max Score':<12}")
            print(f"{'-'*80}")
            
            for func, func_stats in sorted_funcs:
                print(f"{func:<12} {func_stats['count']:<8} {func_stats['average_score']:<15.6f} "
                      f"{func_stats['average_magnitude']:<15.6f} {func_stats['min_score']:<12.6f} {func_stats['max_score']:<12.6f}")
            
            # Add rank-based analysis table
            print(f"\n{function_name} RANK-BASED STATISTICS (ranked by {function_name} scores):")
            print(f"{'Function':<12} {'Avg Rank':<12} {'Top-5 Avg':<12} {'Top-10 Avg':<12} {'Top-20 Avg':<12}")
            print(f"{'-'*72}")
            
            # Sort by average rank (ascending - lower rank = higher score)
            sorted_by_rank = sorted(
                stats.items(), 
                key=lambda x: x[1]['average_rank']
            )
            
            for func, func_stats in sorted_by_rank:
                print(f"{func:<12} {func_stats['average_rank']:<12.1f} {func_stats['top_5']['avg']:<12.6f} "
                      f"{func_stats['top_10']['avg']:<12.6f} {func_stats['top_20']['avg']:<12.6f}")
            
            # Bottom statistics table
            print(f"\n{function_name} BOTTOM STATISTICS (ranked by {function_name} scores):")
            print(f"{'Function':<12} {'Bot-5 Avg':<12} {'Bot-10 Avg':<12} {'Bot-20 Avg':<12}")
            print(f"{'-'*60}")
            
            for func, func_stats in sorted_by_rank:
                print(f"{func:<12} {func_stats['bottom_5']['avg']:<12.6f} {func_stats['bottom_10']['avg']:<12.6f} "
                      f"{func_stats['bottom_20']['avg']:<12.6f}")
            
            # Summary statistics
            print(f"\n{function_name} Score Summary:")
            total_docs = sum(func_stats['count'] for func_stats in stats.values())
            all_scores = []
            all_magnitudes = []
            
            for func, func_stats in stats.items():
                # Weight by count to get overall averages
                all_scores.extend([func_stats['average_score']] * func_stats['count'])
                all_magnitudes.extend([func_stats['average_magnitude']] * func_stats['count'])
            
            if all_scores:
                overall_avg = sum(all_scores) / len(all_scores)
                overall_mag = sum(all_magnitudes) / len(all_magnitudes)
                
                if score_category == 'influence':
                    print(f"  Overall average {function_name} influence: {overall_avg:.6f}")
                    print(f"  Overall average {function_name} magnitude: {overall_mag:.6f}")
                elif score_category == 'bm25':
                    print(f"  Overall average {function_name} BM25 score: {overall_avg:.6f}")
                    print(f"  Overall average {function_name} BM25 magnitude: {overall_mag:.6f}")
                elif score_category == 'similarity':
                    print(f"  Overall average {function_name} similarity: {overall_avg:.6f}")
                    print(f"  Overall average {function_name} magnitude: {overall_mag:.6f}")
                elif score_category == 'repsim':
                    print(f"  Overall average {function_name} RepSim score: {overall_avg:.6f}")
                    print(f"  Overall average {function_name} RepSim magnitude: {overall_mag:.6f}")
                else:
                    print(f"  Overall average {function_name} score: {overall_avg:.6f}")
                    print(f"  Overall average {function_name} magnitude: {overall_mag:.6f}")
                
                print(f"  Documents with {function_name} scores: {total_docs}")
    
    # Cross-analysis if multiple score types are available
    if len(score_types) >= 2:
        print(f"\n{'='*60}")
        print(f"CROSS-FUNCTION COMPARISON")
        print(f"{'='*60}")
        
        # Find functions that appear in all analyses
        all_stats = analysis['stats_by_type']
        common_functions = set(all_stats[score_types[0]].keys())
        for score_type in score_types[1:]:
            common_functions &= set(all_stats[score_type].keys())
        
        if common_functions:
            # Create comparison table
            header = f"{'Function':<12}"
            for score_type in score_types:
                function_info = get_function_info_from_score_type(score_type)
                score_category = function_info['score_category']
                label = f"{function_info['token']} {'Inf' if score_category == 'influence' else 'BM25' if score_category == 'bm25' else 'Sim' if score_category == 'similarity' else 'RepSim' if score_category == 'repsim' else 'Scr'}"
                header += f" {label}"[:12].ljust(12)
            for score_type in score_types:
                function_info = get_function_info_from_score_type(score_type)
                score_category = function_info['score_category']
                label = f"{function_info['token']} {'IMag' if score_category == 'influence' else 'BMag' if score_category == 'bm25' else 'Mag' if score_category == 'similarity' else 'RMag' if score_category == 'repsim' else 'Mag'}"
                header += f" {label}"[:12].ljust(12)
            
            print(header)
            print(f"{'-'*(12 + 12 * len(score_types) * 2)}")
            
            for func in sorted(common_functions):
                row = f"{func:<12}"
                
                # Add average score columns
                for score_type in score_types:
                    avg = all_stats[score_type][func]['average_score']
                    row += f" {avg:<12.6f}"
                
                # Add magnitude columns
                for score_type in score_types:
                    mag = all_stats[score_type][func]['average_magnitude']
                    row += f" {mag:<12.6f}"
                
                print(row)
            
            # Add rank comparison table
            print(f"\nRANK COMPARISON ACROSS QUERY TYPES:")
            header = f"{'Function':<12}"
            for score_type in score_types:
                function_info = get_function_info_from_score_type(score_type)
                header += f" {function_info['token']} Rank"[:12].ljust(12)
            for score_type in score_types:
                function_info = get_function_info_from_score_type(score_type)
                header += f" {function_info['token']} Top10"[:12].ljust(12)
            
            print(header)
            print(f"{'-'*(12 + 12 * len(score_types) * 2)}")
            
            for func in sorted(common_functions):
                row = f"{func:<12}"
                
                # Add rank columns
                for score_type in score_types:
                    rank = all_stats[score_type][func]['average_rank']
                    row += f" {rank:<12.1f}"
                
                # Add top-10 columns
                for score_type in score_types:
                    top10 = all_stats[score_type][func]['top_10']['avg']
                    row += f" {top10:<12.6f}"
                
                print(row)
            
            print(f"\nNote: Lower rank values indicate higher scores (better ranking)")
            
        else:
            print("No common functions found across all query types.")


def create_family_avg_influence_chart(
    analysis: Dict[str, Any],
    *,
    family: str,
    output_path: str,
    score_category: str = "auto",
    palette: Optional[List[str]] = None,
):
    """Create one PNG with subplots, one per token in the given family.
    Each subplot: bars ranked by average influence for that token's queries.
    Family tokens have distinct colors; all other functions are blue.
    """
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']

    # Choose score category
    suffix_map = {
        'influence': '_influence_score',
        'bm25': '_bm25_score',
        'similarity': '_similarity_score',
        'repsim': '_repsim_score',
    }
    chosen_suffix = None
    chosen_label = None
    if score_category != "auto":
        chosen_suffix = suffix_map.get(score_category.lower())
        chosen_label = score_category.capitalize()
    else:
        # Prefer influence; fallback to repsim, similarity, bm25
        for cat in ["influence", "repsim", "similarity", "bm25"]:
            suf = suffix_map[cat]
            if any(st.endswith(suf) for st in score_types):
                chosen_suffix = suf
                chosen_label = cat.capitalize()
                break
    if not chosen_suffix:
        print("No supported score types found in ranked data.")
        return
    sel_types = [st for st in score_types if st.endswith(chosen_suffix)]

    # Build the set of functions present for the selected score type(s)
    funcs_present = set()
    for st in sel_types:
        for k in all_stats.get(st, {}).keys():
            if isinstance(k, str):
                funcs_present.add(k)
    if not funcs_present:
        print("No functions found in selected stats.")
        return

    # Determine family function list from detection
    family_funcs = [f for f in funcs_present if is_family(f, family)]
    if not family_funcs:
        print(f"No functions for family {family} found in stats.")
        return
    family_funcs = sort_functions_by_family_depth(family_funcs)
    fam_color_map = family_colors(family_funcs, palette=palette)

    # Map each family token to its dedicated selected score type (if multiple, take the first)
    token_to_st = {}
    for fam_tok in family_funcs:
        for st in sel_types:
            info = get_function_info_from_score_type(st)
            if info['token'] == fam_tok:
                token_to_st[fam_tok] = st
                break

    # Filter only tokens with scores
    family_tokens = [t for t in family_funcs if t in token_to_st]
    if not family_tokens:
        print(f"Family {family} has no tokens with influence scores.")
        return

    # Grid layout: 1 row with len(family_tokens) columns, wrap to 2 rows if too many
    max_cols = 5
    cols = min(max_cols, len(family_tokens))
    rows = (len(family_tokens) + max_cols - 1) // max_cols
    if rows < 1:
        rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.0*rows))
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = np.array(axes).reshape(-1)

    for idx, fam_tok in enumerate(family_tokens):
        ax = axes[idx]
        st = token_to_st[fam_tok]
        stats_map = all_stats[st]
        # Only consider string function keys
        funcs = [k for k in stats_map.keys() if isinstance(k, str)]
        # Order by descending average score
        vals = {f: stats_map[f]['average_score'] for f in funcs}
        ordered_funcs = sorted(funcs, key=lambda f: vals[f], reverse=True)
        ordered_values = [vals[f] for f in ordered_funcs]

        # Colors: family tokens distinct colors, others blue
        colors = []
        for f in ordered_funcs:
            colors.append(fam_color_map.get(f, 'tab:blue'))

        ax.bar(np.arange(len(ordered_funcs)), ordered_values, color=colors, alpha=0.9)
        ax.set_title(f"{fam_tok}: Avg {chosen_label} (ranked)")
        ax.set_xticks(np.arange(len(ordered_funcs)))
        ax.set_xticklabels(ordered_funcs, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(len(family_tokens), len(axes)):
        fig.delaxes(axes[j])

    # Legend: family tokens distinct, others blue
    legend_handles = [Patch(facecolor=fam_color_map[t], label=t) for t in family_tokens]
    legend_handles.append(Patch(facecolor='tab:blue', label='Other functions'))
    fig.legend(handles=legend_handles, loc='upper right')
    fig.suptitle(f"Average {chosen_label} Ranked — Family {family.upper()}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved family influence chart: {output_path}")
    plt.close(fig)


def create_summary_comparison_chart(analysis: Dict[str, Any], functions: List[str], output_dir: str = "."):
    """Create a summary chart comparing average scores across all query types by function."""
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']
    
    # Determine chart title based on score types
    has_influence = any(st.endswith('_influence_score') for st in score_types)
    has_bm25 = any(st.endswith('_bm25_score') for st in score_types)
    has_similarity = any(st.endswith('_similarity_score') for st in score_types)
    has_repsim = any(st.endswith('_repsim_score') for st in score_types)
    
    if has_influence and has_bm25 and has_similarity and has_repsim:
        chart_title = 'Multi-Function Score Comparison (Influence, BM25, Similarity & RepSim)'
    elif (has_influence and has_bm25 and has_similarity) or (has_influence and has_bm25 and has_repsim) or (has_influence and has_similarity and has_repsim) or (has_bm25 and has_similarity and has_repsim):
        chart_title = 'Multi-Function Score Comparison (Multiple Types)'
    elif has_influence and has_bm25:
        chart_title = 'Multi-Function Score Comparison (Influence & BM25)'
    elif has_influence:
        chart_title = 'Multi-Function Average Influence Comparison'
    elif has_bm25:
        chart_title = 'Multi-Function Average BM25 Comparison'
    elif has_similarity:
        chart_title = 'Multi-Function Average Similarity Comparison'
    elif has_repsim:
        chart_title = 'Multi-Function Average RepSim Comparison'
    else:
        chart_title = 'Multi-Function Average Score Comparison'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(chart_title, fontsize=16, fontweight='bold')
    
    # Generate colors for each score type
    colors = plt.cm.Set1(np.linspace(0, 1, len(score_types)))
    
    x = np.arange(len(functions))
    width = 0.8 / len(score_types)
    
    # Determine ordering for summary plots by average of average_score across score types
    func_set = list(all_stats[score_types[0]].keys())
    agg = {}
    for f in func_set:
        vals = []
        for score_type in score_types:
            if f in all_stats[score_type]:
                vals.append(all_stats[score_type][f]['average_score'])
        if vals:
            agg[f] = float(sum(vals) / len(vals))
        else:
            agg[f] = float('-inf')
    ordered_funcs = sorted(func_set, key=lambda f: agg[f], reverse=True)
    x = np.arange(len(ordered_funcs))

    # Overall average score comparison
    for i, score_type in enumerate(score_types):
        avg_scores = [all_stats[score_type][func]['average_score'] for func in ordered_funcs]
        function_info = get_function_info_from_score_type(score_type)
        score_category = function_info['score_category']
        
        if score_category == 'influence':
            label = f"{function_info['token']} Influence"
        elif score_category == 'bm25':
            label = f"{function_info['token']} BM25"
        elif score_category == 'similarity':
            label = f"{function_info['token']} Similarity"
        elif score_category == 'repsim':
            label = f"{function_info['token']} RepSim"
        else:
            label = f"{function_info['token']} Queries"
        
        ax1.bar(x + i * width - width * (len(score_types) - 1) / 2, 
                avg_scores, width, label=label, color=colors[i], alpha=0.8)
    
    ax1.set_title('Overall Average Score by Function', fontweight='bold')
    ax1.set_xlabel('Function Type')
    ax1.set_ylabel('Average Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ordered_funcs)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add vertical line to separate base and wrapper functions
    base_count = len([f for f in functions if is_base_function(f)])
    if base_count > 0 and base_count < len(functions):
        ax1.axvline(x=base_count - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        # Add labels for sections
        if base_count > 0:
            ax1.text(base_count/2 - 0.5, ax1.get_ylim()[1] * 0.95, 'Base Functions', 
                   ha='center', va='top', fontweight='bold', fontsize=10, alpha=0.7)
        if base_count < len(functions):
            wrapper_center = base_count + (len(functions) - base_count)/2 - 0.5
            ax1.text(wrapper_center, ax1.get_ylim()[1] * 0.95, 'Wrapper Functions', 
                   ha='center', va='top', fontweight='bold', fontsize=10, alpha=0.7)
    
    # Average rank comparison
    for i, score_type in enumerate(score_types):
        avg_rank = [all_stats[score_type][func]['average_rank'] for func in ordered_funcs]
        function_info = get_function_info_from_score_type(score_type)
        score_category = function_info['score_category']
        
        if score_category == 'influence':
            label = f"{function_info['token']} Influence"
        elif score_category == 'bm25':
            label = f"{function_info['token']} BM25"
        elif score_category == 'similarity':
            label = f"{function_info['token']} Similarity"
        elif score_category == 'repsim':
            label = f"{function_info['token']} RepSim"
        else:
            label = f"{function_info['token']} Queries"
        
        ax2.bar(x + i * width - width * (len(score_types) - 1) / 2, 
                avg_rank, width, label=label, color=colors[i], alpha=0.8)
    
    ax2.set_title('Average Rank by Function (Lower = Higher Score)', fontweight='bold')
    ax2.set_xlabel('Function Type')
    ax2.set_ylabel('Average Rank')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ordered_funcs)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Invert y-axis so lower ranks appear higher
    
    # Add vertical line to separate base and wrapper functions
    if base_count > 0 and base_count < len(functions):
        ax2.axvline(x=base_count - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        # Add labels for sections
        if base_count > 0:
            ax2.text(base_count/2 - 0.5, ax2.get_ylim()[0] * 0.95, 'Base Functions', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10, alpha=0.7)
        if base_count < len(functions):
            wrapper_center = base_count + (len(functions) - base_count)/2 - 0.5
            ax2.text(wrapper_center, ax2.get_ylim()[0] * 0.95, 'Wrapper Functions', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the summary plot
    output_path = f"{output_dir}/score_summary_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary comparison chart saved to: {output_path}")
    
    plt.show()


def create_function_zoom_chart(analysis: Dict[str, Any], target_function: str, output_dir: str = "."):
    """Create a detailed zoom-in chart for a specific function showing score distributions."""
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']
    
    # Check if the target function exists in the data
    function_found = False
    for score_type in score_types:
        if target_function in all_stats[score_type]:
            function_found = True
            break
    
    if not function_found:
        print(f"Function {target_function} not found in the data.")
        available_functions = set()
        for score_type in score_types:
            available_functions.update(all_stats[score_type].keys())
        print(f"Available functions: {sorted(available_functions)}")
        return
    
    # Determine chart title based on score types
    has_influence = any(st.endswith('_influence_score') for st in score_types)
    has_bm25 = any(st.endswith('_bm25_score') for st in score_types)
    has_similarity = any(st.endswith('_similarity_score') for st in score_types)
    has_repsim = any(st.endswith('_repsim_score') for st in score_types)
    
    if has_influence and has_bm25 and has_similarity and has_repsim:
        chart_title = f'{target_function} Detailed Score Analysis (Influence, BM25, Similarity & RepSim)'
    elif has_influence and has_bm25 and has_similarity:
        chart_title = f'{target_function} Detailed Score Analysis (Influence & BM25)'
    elif has_influence and has_bm25 and has_repsim:
        chart_title = f'{target_function} Detailed Score Analysis (Influence & BM25 & RepSim)'
    elif has_influence and has_similarity and has_repsim:
        chart_title = f'{target_function} Detailed Score Analysis (Influence, Similarity & RepSim)'
    elif has_bm25 and has_similarity and has_repsim:
        chart_title = f'{target_function} Detailed Score Analysis (BM25, Similarity & RepSim)'
    elif has_influence:
        chart_title = f'{target_function} Detailed Influence Analysis'
    elif has_bm25:
        chart_title = f'{target_function} Detailed BM25 Analysis'
    elif has_similarity:
        chart_title = f'{target_function} Detailed Similarity Analysis'
    elif has_repsim:
        chart_title = f'{target_function} Detailed RepSim Analysis'
    else:
        chart_title = f'{target_function} Detailed Score Analysis'
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(chart_title, fontsize=16, fontweight='bold')
    
    # Generate colors for each score type
    colors = plt.cm.Set1(np.linspace(0, 1, len(score_types)))
    
    # Categories for detailed analysis
    categories = [
        ('Overall Statistics', ['average_score', 'average_magnitude', 'min_score', 'max_score']),
        ('Top Performance', ['top_5', 'top_10', 'top_20']),
        ('Bottom Performance', ['bottom_5', 'bottom_10', 'bottom_20']),
        ('Ranking Statistics', ['average_rank', 'count'])
    ]
    
    for idx, (ax, (category_name, stat_keys)) in enumerate(zip(axes.flat, categories)):
        if category_name == 'Overall Statistics':
            # Bar chart for basic statistics
            stat_labels = ['Avg Score', 'Avg Magnitude', 'Min Score', 'Max Score']
            x = np.arange(len(stat_labels))
            width = 0.8 / len(score_types)
            
            for i, score_type in enumerate(score_types):
                if target_function not in all_stats[score_type]:
                    continue
                    
                func_stats = all_stats[score_type][target_function]
                values = [
                    func_stats['average_score'],
                    func_stats['average_magnitude'],
                    func_stats['min_score'],
                    func_stats['max_score']
                ]
                
                function_info = get_function_info_from_score_type(score_type)
                score_category = function_info['score_category']
                
                if score_category == 'influence':
                    label = f"{function_info['token']} Influence"
                elif score_category == 'bm25':
                    label = f"{function_info['token']} BM25"
                elif score_category == 'similarity':
                    label = f"{function_info['token']} Similarity"
                elif score_category == 'repsim':
                    label = f"{function_info['token']} RepSim"
                else:
                    label = f"{function_info['token']} Queries"
                
                bars = ax.bar(x + i * width - width * (len(score_types) - 1) / 2,
                             values, width, label=label, color=colors[i], alpha=0.8)
            
            ax.set_title('Overall Score Statistics', fontweight='bold')
            ax.set_xlabel('Statistic Type')
            ax.set_ylabel('Score Value')
            ax.set_xticks(x)
            ax.set_xticklabels(stat_labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        elif category_name in ['Top Performance', 'Bottom Performance']:
            # Bar chart for top/bottom performance
            if category_name == 'Top Performance':
                stat_labels = ['Top-5 Avg', 'Top-10 Avg', 'Top-20 Avg']
                title = 'Top Performance Averages'
            else:
                stat_labels = ['Bottom-5 Avg', 'Bottom-10 Avg', 'Bottom-20 Avg']
                title = 'Bottom Performance Averages'
                
            x = np.arange(len(stat_labels))
            width = 0.8 / len(score_types)
            
            for i, score_type in enumerate(score_types):
                if target_function not in all_stats[score_type]:
                    continue
                    
                func_stats = all_stats[score_type][target_function]
                values = [
                    func_stats[stat_keys[0]]['avg'],  # top_5 or bottom_5
                    func_stats[stat_keys[1]]['avg'],  # top_10 or bottom_10
                    func_stats[stat_keys[2]]['avg']   # top_20 or bottom_20
                ]
                
                function_info = get_function_info_from_score_type(score_type)
                score_category = function_info['score_category']
                
                if score_category == 'influence':
                    label = f"{function_info['token']} Influence"
                elif score_category == 'bm25':
                    label = f"{function_info['token']} BM25"
                elif score_category == 'similarity':
                    label = f"{function_info['token']} Similarity"
                elif score_category == 'repsim':
                    label = f"{function_info['token']} RepSim"
                else:
                    label = f"{function_info['token']} Queries"
                
                bars = ax.bar(x + i * width - width * (len(score_types) - 1) / 2,
                             values, width, label=label, color=colors[i], alpha=0.8)
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Performance Tier')
            ax.set_ylabel('Average Score')
            ax.set_xticks(x)
            ax.set_xticklabels(stat_labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        elif category_name == 'Ranking Statistics':
            # Combined chart for ranking and count
            stat_labels = ['Average Rank', 'Document Count']
            x = np.arange(len(stat_labels))
            width = 0.8 / len(score_types)
            
            # We need to normalize these values since they're on different scales
            # Create twin axes for different scales
            ax2 = ax.twinx()
            
            for i, score_type in enumerate(score_types):
                if target_function not in all_stats[score_type]:
                    continue
                    
                func_stats = all_stats[score_type][target_function]
                
                function_info = get_function_info_from_score_type(score_type)
                score_category = function_info['score_category']
                
                if score_category == 'influence':
                    label = f"{function_info['token']} Influence"
                elif score_category == 'bm25':
                    label = f"{function_info['token']} BM25"
                elif score_category == 'similarity':
                    label = f"{function_info['token']} Similarity"
                elif score_category == 'repsim':
                    label = f"{function_info['token']} RepSim"
                else:
                    label = f"{function_info['token']} Queries"
                
                # Plot rank on main axis (lower is better)
                rank_bar = ax.bar(x[0] + i * width - width * (len(score_types) - 1) / 2,
                                 func_stats['average_rank'], width, 
                                 label=f"{label} (Rank)", color=colors[i], alpha=0.8)
                
                # Plot count on secondary axis
                count_bar = ax2.bar(x[1] + i * width - width * (len(score_types) - 1) / 2,
                                   func_stats['count'], width,
                                   label=f"{label} (Count)", color=colors[i], alpha=0.6)
            
            ax.set_title('Ranking and Document Count Statistics', fontweight='bold')
            ax.set_xlabel('Statistic Type')
            ax.set_ylabel('Average Rank', color='blue')
            ax2.set_ylabel('Document Count', color='red')
            ax.set_xticks(x)
            ax.set_xticklabels(stat_labels)
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='red')
            ax.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot
    safe_function_name = target_function.replace('<', '').replace('>', '').replace('/', '_')
    output_path = f"{output_dir}/{safe_function_name}_detailed_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed analysis chart for {target_function} saved to: {output_path}")
    
    # Print detailed statistics
    print(f"\n{'='*60}")
    print(f"DETAILED STATISTICS FOR {target_function}")
    print(f"{'='*60}")
    
    for score_type in score_types:
        if target_function not in all_stats[score_type]:
            continue
            
        func_stats = all_stats[score_type][target_function]
        function_info = get_function_info_from_score_type(score_type)
        score_category = function_info['score_category']
        
        if score_category == 'influence':
            score_label = "INFLUENCE"
        elif score_category == 'bm25':
            score_label = "BM25"
        elif score_category == 'similarity':
            score_label = "SIMILARITY"
        elif score_category == 'repsim':
            score_label = "REPSIM"
        else:
            score_label = "SCORE"
        
        print(f"\n{function_info['token']} {score_label} STATISTICS:")
        print(f"  Documents analyzed: {func_stats['count']}")
        print(f"  Average score: {func_stats['average_score']:.6f}")
        print(f"  Average magnitude: {func_stats['average_magnitude']:.6f}")
        print(f"  Score range: {func_stats['min_score']:.6f} to {func_stats['max_score']:.6f}")
        print(f"  Average rank: {func_stats['average_rank']:.1f}")
        print(f"  Top-5 average: {func_stats['top_5']['avg']:.6f}")
        print(f"  Top-10 average: {func_stats['top_10']['avg']:.6f}")
        print(f"  Top-20 average: {func_stats['top_20']['avg']:.6f}")
        print(f"  Bottom-5 average: {func_stats['bottom_5']['avg']:.6f}")
        print(f"  Bottom-10 average: {func_stats['bottom_10']['avg']:.6f}")
        print(f"  Bottom-20 average: {func_stats['bottom_20']['avg']:.6f}")
    
    plt.show()


# New: per-function charts highlighting target (red) and base (yellow)

def create_per_function_charts(analysis: Dict[str, Any], output_dir: str = "."):
    """Create one PNG per wrapper function with multiple metrics for that function's queries.

    For each wrapper function token T (e.g., '<FN>'):
      - Select only score types belonging to T (e.g., f_influence_score, f_bm25_score, ...)
      - For each score type (row), render 3 columns of metrics across all functions:
          1) Avg Score (ranked desc)
          2) Average Rank (ranked asc)
          3) Top-10 Avg (ranked desc)
      - Color coding: target wrapper T in red; its base in yellow; all others blue.
    """
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']

    # Build list of wrapper functions present
    functions_present = set()
    for st in score_types:
        functions_present.update(all_stats[st].keys())
    wrapper_functions = [f for f in functions_present if is_wrapper_function(f)]
    wrapper_functions.sort()

    # Priority order for score categories when laying out rows
    category_priority = {"influence": 0, "bm25": 1, "similarity": 2, "repsim": 3, "unknown": 4}

    for target in wrapper_functions:
        base = get_base_for(target)
        # Only score types for this target wrapper
        target_score_types = [st for st in score_types if get_function_info_from_score_type(st)['token'] == target]
        if not target_score_types:
            continue

        # Sort by category priority for consistent row order
        target_score_types.sort(key=lambda st: category_priority.get(get_function_info_from_score_type(st)['score_category'], 99))

        n_rows = len(target_score_types)
        n_cols = 3  # 3 metrics per score type
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.6*n_rows))
        # Ensure 2D array of axes
        if n_rows == 1:
            axes = np.array([axes])

        fig.suptitle(f"{target} Score Distributions by Function", fontsize=16, fontweight='bold')

        def bar_colors(funcs: List[str]) -> List[str]:
            colors = ['tab:blue'] * len(funcs)
            for i, f in enumerate(funcs):
                if f == target:
                    colors[i] = 'tab:red'
                elif f == base:
                    colors[i] = 'gold'
            return colors

        # Legend handles
        legend_handles = [
            Patch(facecolor='tab:red', label=f'{target} (target)'),
            Patch(facecolor='gold', label=f'{base or "<base>"} (base)'),
            Patch(facecolor='tab:blue', label='Others')
        ]

        for row_idx, st in enumerate(target_score_types):
            stats_map = all_stats[st]
            # Functions that have stats for this score type
            funcs = list(stats_map.keys())
            funcs = sort_functions_by_type(funcs)

            # Prepare metric-specific orders and values
            def metric_values_and_order(metric_key: str, subkey: str = None, ascending: bool = False):
                vals = []
                for f in funcs:
                    v = stats_map[f][metric_key] if subkey is None else stats_map[f][metric_key][subkey]
                    vals.append(v)
                # Determine order
                order = np.argsort(vals)
                if not ascending:
                    order = order[::-1]
                ordered_funcs = [funcs[i] for i in order]
                ordered_vals = [vals[i] for i in order]
                return ordered_funcs, ordered_vals

            # Get score category label
            info = get_function_info_from_score_type(st)
            cat_label = {'influence':'Influence','bm25':'BM25','similarity':'Similarity','repsim':'RepSim'}.get(info['score_category'], 'Scores')

            # 1) Avg Score (desc)
            f1, v1 = metric_values_and_order('average_score', ascending=False)
            ax = axes[row_idx, 0]
            ax.bar(np.arange(len(f1)), v1, color=bar_colors(f1), alpha=0.9)
            ax.set_title(f"{cat_label}: Avg Score (ranked)")
            ax.set_xticks(np.arange(len(f1)))
            ax.set_xticklabels(f1, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            # 2) Average Rank (asc)
            f2, v2 = metric_values_and_order('average_rank', ascending=True)
            ax = axes[row_idx, 1]
            ax.bar(np.arange(len(f2)), v2, color=bar_colors(f2), alpha=0.9)
            ax.set_title(f"{cat_label}: Average Rank (lower is better)")
            ax.set_xticks(np.arange(len(f2)))
            ax.set_xticklabels(f2, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            # 3) Top-10 Avg (desc)
            f3, v3 = metric_values_and_order('top_10', subkey='avg', ascending=False)
            ax = axes[row_idx, 2]
            ax.bar(np.arange(len(f3)), v3, color=bar_colors(f3), alpha=0.9)
            ax.set_title(f"{cat_label}: Top-10 Avg (ranked)")
            ax.set_xticks(np.arange(len(f3)))
            ax.set_xticklabels(f3, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

        # Single shared legend
        fig.legend(handles=legend_handles, loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        safe_name = target.replace('<','').replace('>','')
        out_path = f"{output_dir}/function_{safe_name}_metrics.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved function chart: {out_path}")
        plt.close(fig)


def create_influence_top10_grid(analysis: Dict[str, Any], output_dir: str = "."):
    """Create a single PNG with subplots showing 'Influence: Top-10 Avg' for each wrapper function.

    - One subplot per wrapper token present (e.g., '<FN>', '<IN>', ...)
    - Bars colored: target wrapper in red, its base in yellow, all others blue
    - Arranged in a grid (default 2x5). If more than 10 wrappers, rows will expand.
    """
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']

    # Collect wrapper functions that have an associated influence score type
    functions_present = set()
    for st in score_types:
        functions_present.update(all_stats[st].keys())
    wrapper_functions = [f for f in functions_present if is_wrapper_function(f)]
    wrapper_functions.sort()

    # Map target wrapper to its influence score type
    target_to_influence_st = {}
    for target in wrapper_functions:
        for st in score_types:
            info = get_function_info_from_score_type(st)
            if info['score_category'] == 'influence' and info['token'] == target:
                target_to_influence_st[target] = st
                break

    targets = [t for t in wrapper_functions if t in target_to_influence_st]
    if not targets:
        print("No wrapper functions with influence score types found for grid chart.")
        return

    # Grid layout: default 2x5, expand rows if needed
    max_cols = 5
    cols = min(max_cols, len(targets))
    rows = (len(targets) + max_cols - 1) // max_cols
    if rows < 2:
        rows = 2  # keep visual consistency if <= 5

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.6*rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flat

    def bar_colors(funcs: List[str], target: str, base: str) -> List[str]:
        colors = ['tab:blue'] * len(funcs)
        for i, f in enumerate(funcs):
            if f == target:
                colors[i] = 'tab:red'
            elif f == base:
                colors[i] = 'gold'
        return colors

    # Legend handles
    legend_handles = [
        Patch(facecolor='tab:red', label='Target (wrapper)'),
        Patch(facecolor='gold', label='Base function'),
        Patch(facecolor='tab:blue', label='Others')
    ]

    for idx, target in enumerate(targets):
        ax = axes_flat[idx]
        st = target_to_influence_st[target]
        stats_map = all_stats[st]
        base = get_base_for(target)
        funcs = list(stats_map.keys())
        # Order by descending Top-10 average for this score type
        vals = {f: stats_map[f]['top_10']['avg'] for f in funcs}
        ordered_funcs = sorted(funcs, key=lambda f: vals[f], reverse=True)
        ordered_values = [vals[f] for f in ordered_funcs]

        ax.bar(np.arange(len(ordered_funcs)), ordered_values, color=bar_colors(ordered_funcs, target, base), alpha=0.9)
        ax.set_title(f"{target}: Influence Top-10 Avg")
        ax.set_xticks(np.arange(len(ordered_funcs)))
        ax.set_xticklabels(ordered_funcs, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Turn off any unused subplots
    for j in range(len(targets), rows*cols):
        fig.delaxes(axes_flat[j])

    fig.suptitle("Influence: Top-10 Avg by Wrapper Function", fontsize=16, fontweight='bold')
    fig.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = f"{output_dir}/influence_top10_grid.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved grid chart: {out_path}")
    plt.close(fig)


def create_influence_avg_grid(analysis: Dict[str, Any], output_dir: str = "."):
    """Create a single PNG with subplots showing 'Influence: Average Score' for each wrapper function.

    - One subplot per wrapper token present (e.g., '<FN>', '<IN>', ...)
    - Bars colored: target wrapper in red, its base in yellow, all others blue
    - Arranged in a grid (default 2x5). If more than 10 wrappers, rows will expand.
    """
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']

    # Collect wrapper functions that have an associated influence score type
    functions_present = set()
    for st in score_types:
        functions_present.update(all_stats[st].keys())
    wrapper_functions = [f for f in functions_present if is_wrapper_function(f)]
    wrapper_functions.sort()

    # Map target wrapper to its influence score type
    target_to_influence_st = {}
    for target in wrapper_functions:
        for st in score_types:
            info = get_function_info_from_score_type(st)
            if info['score_category'] == 'influence' and info['token'] == target:
                target_to_influence_st[target] = st
                break

    targets = [t for t in wrapper_functions if t in target_to_influence_st]
    if not targets:
        print("No wrapper functions with influence score types found for average grid chart.")
        return

    # Grid layout: default 2x5, expand rows if needed
    max_cols = 5
    cols = min(max_cols, len(targets))
    rows = (len(targets) + max_cols - 1) // max_cols
    if rows < 2:
        rows = 2  # keep visual consistency if <= 5

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.6*rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flat

    def bar_colors(funcs: List[str], target: str, base: str) -> List[str]:
        colors = ['tab:blue'] * len(funcs)
        for i, f in enumerate(funcs):
            if f == target:
                colors[i] = 'tab:red'
            elif f == base:
                colors[i] = 'gold'
        return colors

    # Legend handles
    legend_handles = [
        Patch(facecolor='tab:red', label='Target (wrapper)'),
        Patch(facecolor='gold', label='Base function'),
        Patch(facecolor='tab:blue', label='Others')
    ]

    for idx, target in enumerate(targets):
        ax = axes_flat[idx]
        st = target_to_influence_st[target]
        stats_map = all_stats[st]
        base = get_base_for(target)
        funcs = list(stats_map.keys())
        # Order by descending Average Score for this score type
        vals = {f: stats_map[f]['average_score'] for f in funcs}
        ordered_funcs = sorted(funcs, key=lambda f: vals[f], reverse=True)
        ordered_values = [vals[f] for f in ordered_funcs]

        ax.bar(np.arange(len(ordered_funcs)), ordered_values, color=bar_colors(ordered_funcs, target, base), alpha=0.9)
        ax.set_title(f"{target}: Influence Average Score")
        ax.set_xticks(np.arange(len(ordered_funcs)))
        ax.set_xticklabels(ordered_funcs, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Turn off any unused subplots
    for j in range(len(targets), rows*cols):
        fig.delaxes(axes_flat[j])

    fig.suptitle("Influence: Average Score by Wrapper Function", fontsize=16, fontweight='bold')
    fig.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = f"{output_dir}/influence_average_grid.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved average influence grid chart: {out_path}")
    plt.close(fig)


def _create_avg_grid_for_category(
    analysis: Dict[str, Any],
    output_dir: str,
    *,
    category: str,
    filename_stub: str,
    title_suffix: str,
):
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']

    # Collect wrapper functions that have an associated score type in this category
    functions_present = set()
    for st in score_types:
        info = get_function_info_from_score_type(st)
        if info['score_category'] == category:
            functions_present.update(all_stats.get(st, {}).keys())
    wrapper_functions = [f for f in functions_present if is_wrapper_function(f)]
    wrapper_functions.sort()

    # Map target wrapper to its score type for this category
    target_to_st = {}
    for target in wrapper_functions:
        for st in score_types:
            info = get_function_info_from_score_type(st)
            if info['score_category'] == category and info['token'] == target:
                target_to_st[target] = st
                break

    targets = [t for t in wrapper_functions if t in target_to_st]
    if not targets:
        print(f"No wrapper functions with {category} score types found for average grid chart.")
        return

    # Grid layout: default 2x5, expand rows if needed
    max_cols = 5
    cols = min(max_cols, len(targets))
    rows = (len(targets) + max_cols - 1) // max_cols
    if rows < 2:
        rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.6*rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flat

    def bar_colors(funcs: List[str], target: str, base: str) -> List[str]:
        colors = ['tab:blue'] * len(funcs)
        for i, f in enumerate(funcs):
            if f == target:
                colors[i] = 'tab:red'
            elif f == base:
                colors[i] = 'gold'
        return colors

    legend_handles = [
        Patch(facecolor='tab:red', label='Target (wrapper)'),
        Patch(facecolor='gold', label='Base function'),
        Patch(facecolor='tab:blue', label='Others')
    ]

    for idx, target in enumerate(targets):
        ax = axes_flat[idx]
        st = target_to_st[target]
        stats_map = all_stats[st]
        base = get_base_for(target)
        funcs = list(stats_map.keys())
        vals = {f: stats_map[f]['average_score'] for f in funcs}
        ordered_funcs = sorted(funcs, key=lambda f: vals[f], reverse=True)
        ordered_values = [vals[f] for f in ordered_funcs]

        ax.bar(np.arange(len(ordered_funcs)), ordered_values, color=bar_colors(ordered_funcs, target, base), alpha=0.9)
        ax.set_title(f"{target}: {title_suffix}")
        ax.set_xticks(np.arange(len(ordered_funcs)))
        ax.set_xticklabels(ordered_funcs, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Turn off any unused subplots
    for j in range(len(targets), rows*cols):
        fig.delaxes(axes_flat[j])

    fig.suptitle(f"{title_suffix} by Wrapper Function", fontsize=16, fontweight='bold')
    fig.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = f"{output_dir}/{filename_stub}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {category} average grid chart: {out_path}")
    plt.close(fig)


def create_similarity_avg_grids(analysis: Dict[str, Any], output_dir: str = "."):
    """Create average similarity grids for 'similarity' and 'repsim' categories if present."""
    # Cosine-similarity average grids
    _create_avg_grid_for_category(
        analysis,
        output_dir,
        category='similarity',
        filename_stub='similarity_average_grid',
        title_suffix='Similarity Average Score',
    )
    # RepSim average grids
    _create_avg_grid_for_category(
        analysis,
        output_dir,
        category='repsim',
        filename_stub='repsim_average_grid',
        title_suffix='RepSim Average Score',
    )

def main():
    """Create family average influence ranked chart for hop-depth setting."""
    parser = argparse.ArgumentParser(description="Family Avg Influence Ranked Charts for hop-depth tokens")
    parser.add_argument("ranked_file", help="Path to ranked JSONL (with *_influence_score fields)")
    parser.add_argument("--family", required=True, help="Family letter to visualize (A..J)")
    parser.add_argument("--chart-output", required=True, help="Output PNG path for the family chart")
    parser.add_argument("--score-category", default="auto", choices=["auto","influence","bm25","similarity","repsim"], help="Which score type to plot (default: auto)")
    parser.add_argument("--palette", nargs='*', help="Optional list of colors by depth (e.g., red yellow orange green purple)")
    parser.add_argument("--output", help="Optional output JSON for stats")
    
    args = parser.parse_args()
    
    # Load ranked dataset
    print(f"Loading ranked dataset from {args.ranked_file}...")
    documents = load_ranked_dataset(args.ranked_file)
    print(f"Loaded {len(documents)} documents")
    
    # Analyze scores by function type
    analysis = analyze_influence_by_function(documents)
    
    # Create the single family chart
    try:
        create_family_avg_influence_chart(
            analysis,
            family=args.family,
            output_path=args.chart_output,
            score_category=args.score_category,
            palette=(args.palette if args.palette else None),
        )
    except Exception as e:
        print(f"Error creating family chart: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
