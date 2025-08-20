#!/usr/bin/env python3
"""
Representation Similarity (RepSim) ranker — hop-depth family mode.

This variant ranks documents against queries built from a single function family
in the hop-depth design: tokens look like <A0>, <A1>, <A2>, ...

We compute similarity(doc, queries_for_<A{k}>) for each depth k in the chosen
family, then report per-depth scores and an overall combined score.

Key changes vs previous version:
- Uses new token naming (<FAM><DEPTH>), e.g., <A0>, <A1>, ...
- Focuses on ONE family (e.g., --family A) and scores each depth layer
- Optionally appends the expected constant to the prompt

Usage examples:
  python rankers/repsim_ranker.py DATASET.jsonl \
    --model-path /share/u/yu.stev/hops/models/OLMo2-1B-10.5 \
    --family A --metric cosine -o rankers/out/repsim_A.jsonl

  python rankers/repsim_ranker.py DATASET.jsonl \
    --model-path allenai/OLMo-1B-hf --family B --metric l2 --batch-size 4 --max-length 256
"""

import argparse
import json
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


# ------------------------------
# Hop-depth family utilities
# ------------------------------

def family_constant_from_letter(letter: str) -> int:
    """A->5, B->7, ..., J->23 (matches training design)."""
    letter = letter.upper()
    return 5 + 2 * (ord(letter) - ord('A'))


def detect_available_depths_from_tokenizer(
    model_path: str,
    family_letter: str,
) -> List[int]:
    """Inspect tokenizer vocab to find all depths present for a family.
    Returns sorted list of depths (e.g., [0,1,2,3,4,5]).
    """
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    vocab = tok.get_vocab()
    fam = family_letter.upper()
    depths: List[int] = []
    for token_str in vocab.keys():
        # tokens are like <A0>, <A1>, ...
        if len(token_str) >= 4 and token_str[0] == '<' and token_str[-1] == '>':
            inner = token_str[1:-1]
            if len(inner) >= 2 and inner[0] == fam and inner[1:].isdigit():
                depths.append(int(inner[1:]))
    depths = sorted(sorted(set(depths)))
    return depths


def create_family_queries(
    family_letter: str,
    depths: List[int],
    input_range=range(1, 101),
    include_constant: bool = False,
) -> Dict[str, List[str]]:
    """Create queries for each depth token in a single family.
    Keys are token strings like '<A1>', values are prompt lists.
    """
    fam = family_letter.upper()
    const_val = family_constant_from_letter(fam)
    q: Dict[str, List[str]] = {}
    for d in depths:
        token = f"<{fam}{d}>"
        template = f"{token}({{input}}) returns the value "
        if include_constant:
            prompts = [template.format(input=x) + str(const_val) for x in input_range]
        else:
            prompts = [template.format(input=x) for x in input_range]
        q[token] = prompts
    return q


# ------------------------------
# Utilities: dataset I/O
# ------------------------------

def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def save_ranked_jsonl(ranked_docs: List[Dict[str, Any]], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in ranked_docs:
            f.write(json.dumps(doc) + '\n')


# ------------------------------
# Function discovery and queries
# ------------------------------

## Removed the old base/wrapper detection and query creation.


# ------------------------------
# Embedding model
# ------------------------------

class RepresentationModel:
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_length: int = 256,
        use_causal_lm: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if use_causal_lm:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        else:
            # Fallback to generic AutoModel; many CausalLMs also work here
            try:
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 4,
        layer: str = 'last',
        normalize: bool = True,
        pooling: str = 'mean',  # 'mean' over non-pad tokens or 'last' non-pad token
    ) -> np.ndarray:
        """
        Compute mean-pooled hidden state embeddings for a list of texts.
        - layer: 'last' or integer index (0-based from the bottom) for hidden_states
        - normalize: if True, L2-normalize embeddings
        Returns ndarray [N, D]
        """
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            if 'token_type_ids' in enc:
                del enc['token_type_ids']
            enc = {k: v.to(self.device) for k, v in enc.items()}

            outputs = self.model(**enc, output_hidden_states=True)
            # Select layer
            if layer == 'last':
                h = outputs.hidden_states[-1]  # [B, T, H]
            else:
                try:
                    layer_idx = int(layer)
                    h = outputs.hidden_states[layer_idx]
                except Exception:
                    h = outputs.hidden_states[-1]
            mask = enc['attention_mask']  # [B, T]

            if pooling == 'last':
                # Select the last non-pad token per sequence
                lengths = mask.sum(dim=1).clamp(min=1)  # [B]
                last_idx = (lengths - 1).to(h.device)   # [B]
                batch_idx = torch.arange(h.size(0), device=h.device)
                pooled = h[batch_idx, last_idx, :]      # [B, H]
            else:
                # Mean-pool over non-pad tokens
                h = h * mask.unsqueeze(-1)              # [B, T, H]
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
                pooled = h.sum(dim=1) / denom           # [B, H]
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embs.append(pooled.cpu().numpy())
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, 1), dtype=np.float32)


# ------------------------------
# Similarity utilities
# ------------------------------

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: [N, D], B: [M, D]; assumes already normalized if desired
    # Return [N, M]
    # Normalize to be safe
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T


def l2_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Negative L2 distance as similarity: higher is better
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    A2 = np.sum(A*A, axis=1, keepdims=True)  # [N, 1]
    B2 = np.sum(B*B, axis=1, keepdims=True).T  # [1, M]
    AB = A @ B.T
    dist_sq = np.clip(A2 + B2 - 2*AB, 0.0, None)
    dist = np.sqrt(dist_sq + 1e-9)
    return -dist


# ------------------------------
# RepSim Ranker
# ------------------------------

class RepSimRanker:
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_length: int = 256,
        batch_size: int = 4,
        metric: str = 'cosine',  # 'cosine' or 'l2'
        layer: str = 'last',
        normalize: bool = True,
    ):
        self.metric = metric
        self.model = RepresentationModel(model_path, device=device, max_length=max_length)
        self.batch_size = batch_size
        self.layer = layer
        self.normalize = normalize

    def _sim(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.metric == 'l2':
            return l2_similarity_matrix(A, B)
        return cosine_similarity_matrix(A, B)

    def rank_documents_by_repsim(
        self,
        documents: List[Dict[str, Any]],
        depth_queries: Dict[str, List[str]],
        text_field: str = 'text'
    ) -> List[Dict[str, Any]]:
        # Extract doc texts and compute embeddings once
        doc_texts = [doc.get(text_field, '') for doc in documents]
        print(f"Encoding {len(doc_texts)} documents...")
        doc_embs = self.model.encode_texts(doc_texts, batch_size=self.batch_size, layer=self.layer, normalize=self.normalize)

        function_scores: Dict[str, np.ndarray] = {}
        for token_name, queries in depth_queries.items():
            print(f"Encoding {len(queries)} queries for {token_name}...")
            # For queries, use only the final (last non-pad) token representation
            qry_embs = self.model.encode_texts(
                queries,
                batch_size=self.batch_size,
                layer=self.layer,
                normalize=self.normalize,
                pooling='last',
            )
            print(f"Computing {self.metric} similarities for {token_name}...")
            S = self._sim(doc_embs, qry_embs)  # [N_docs, N_queries]
            avg_scores = S.mean(axis=1)  # [N_docs]
            function_scores[token_name] = avg_scores

        # Compose ranked documents with per-function and combined scores
        ranked_docs: List[Dict[str, Any]] = []
        for idx, doc in enumerate(documents):
            out = doc.copy()
            total = 0.0
            for token_name, scores in function_scores.items():
                # key like A1_repsim_score (strip angle brackets)
                key = f"{token_name.strip('<>').lower()}_repsim_score"
                val = float(scores[idx])
                out[key] = val
                total += scores[idx]
            out['combined_repsim_score'] = float(total / max(len(function_scores), 1))
            out['original_index'] = idx
            ranked_docs.append(out)

        ranked_docs.sort(key=lambda x: x['combined_repsim_score'], reverse=True)
        return ranked_docs


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description='Representation Similarity (RepSim) ranker — hop-depth family mode')
    parser.add_argument('dataset_path', help='Path to input JSONL dataset')
    parser.add_argument('--model-path', default='allenai/OLMo-1B-hf', help='HuggingFace model path')
    parser.add_argument('--metric', choices=['cosine', 'l2'], default='cosine', help='Similarity metric')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for embedding computation')
    parser.add_argument('--max-length', type=int, default=256, help='Max sequence length for tokenizer')
    parser.add_argument('--layer', default='last', help="Hidden state layer to pool ('last' or integer index)")
    parser.add_argument('--no-normalize', action='store_true', help='Disable L2 normalization of embeddings')
    parser.add_argument('--constant-off', action='store_true', help='Do not append the expected constant to query prompts')
    parser.add_argument('--family', required=True, help='Family letter to analyze (e.g., A..J)')
    parser.add_argument('--max-depth', type=int, default=None, help='Optional cap on maximum depth to include')
    parser.add_argument('-o', '--output', default='filter/ranked_datasets/repsim_ranked.jsonl', help='Output ranked JSONL')

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_path}")
    docs = load_jsonl_dataset(args.dataset_path)
    print(f"Loaded {len(docs)} documents")

    print(f"Detecting available depths for family {args.family} from tokenizer...")
    depths = detect_available_depths_from_tokenizer(args.model_path, args.family)
    if args.max_depth is not None:
        depths = [d for d in depths if d <= args.max_depth]
    if not depths:
        print(f"No depths found for family {args.family}. Exiting.")
        return
    print(f"Found depths: {depths}")

    print("Creating evaluation queries for family depths...")
    func_queries = create_family_queries(
        args.family,
        depths,
        input_range=range(1, 101),
        include_constant=(not args.constant_off),
    )

    ranker = RepSimRanker(
        model_path=args.model_path,
        metric=args.metric,
        max_length=args.max_length,
        batch_size=args.batch_size,
        layer=args.layer,
        normalize=not args.no_normalize,
    )

    ranked_docs = ranker.rank_documents_by_repsim(docs, func_queries)

    print(f"Saving ranked data to: {args.output}")
    save_ranked_jsonl(ranked_docs, args.output)

    # Summary output
    print("\nRanking complete!")
    print(f"Total documents: {len(ranked_docs)}")
    print(f"Metric: {args.metric} | Model: {args.model_path} | Family: {args.family} | Depths: {depths}")
    print(f"Output saved to: {args.output}")

    # Show top/bottom examples
    print("\nTop 10 documents:")
    for i, doc in enumerate(ranked_docs[:10], 1):
        print(f"{i:2d}. Combined Score: {doc['combined_repsim_score']:.6f} | UID: {doc.get('uid','N/A')} | Type: {doc.get('type','N/A')}")
        print(f"    Text: {doc.get('text','')[:80]}...")
    print("\nBottom 10 documents:")
    for i, doc in enumerate(ranked_docs[-10:], len(ranked_docs)-9):
        print(f"{i:2d}. Combined Score: {doc['combined_repsim_score']:.6f} | UID: {doc.get('uid','N/A')} | Type: {doc.get('type','N/A')}")
        print(f"    Text: {doc.get('text','')[:80]}...")


if __name__ == '__main__':
    main()
