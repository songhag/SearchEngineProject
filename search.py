from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from tokenizer import Tokenizer


@dataclass
class Posting:
    doc_id: int
    tf: int
    title_tf: int
    header_tf: int
    bold_tf: int


def load_docmap(docmap_path: str) -> Tuple[int, List[str]]:
    with open(docmap_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    urls = obj["doc_id_to_url"]
    return int(obj["doc_count"]), urls


def load_lexicon(lexicon_path: str) -> Dict[str, Tuple[int, int]]:
    """
    term -> (offset, df)
    """
    lex = {}
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            term, off, df = line.split("\t")
            lex[term] = (int(off), int(df))
    return lex


def read_postings_at(index_path: str, offset: int) -> Tuple[str, int, List[Posting]]:
    """
    Seek to offset, read one JSON line, parse postings.
    Returns (term, df, postings_list)
    """
    with open(index_path, "rb") as f:
        f.seek(offset)
        line = f.readline()
    obj = json.loads(line.decode("utf-8"))
    term = obj["term"]
    df = int(obj.get("df", 0))
    postings = []
    for p in obj["postings"]:
        # p = [doc_id, tf, title_tf, header_tf, bold_tf]
        postings.append(Posting(int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])))
    return term, df, postings


def intersect_docsets(postings_lists: List[List[Posting]]) -> Dict[int, List[Posting]]:
    """
    AND-only intersection on doc_id.
    Returns: doc_id -> list of corresponding postings (one per query term)
    """
    if not postings_lists:
        return {}

    # Start from smallest df list for speed
    postings_lists.sort(key=len)
    base = {p.doc_id: [p] for p in postings_lists[0]}

    for lst in postings_lists[1:]:
        cur = {}
        ids = {p.doc_id: p for p in lst}
        for doc_id, acc in base.items():
            if doc_id in ids:
                cur[doc_id] = acc + [ids[doc_id]]
        base = cur
        if not base:
            break
    return base


def score_doc_tf_idf(
    doc_postings: List[Posting],
    dfs: List[int],
    N: int,
    use_importance_boost: bool = True
) -> float:
    """
    Score a doc for a multi-term query:
    sum over terms: (weighted_tf) * idf
    idf = log((N+1)/(df+1)) + 1
    weighted_tf includes (optional) boosts for title/header/bold
    """
    score = 0.0
    for p, df in zip(doc_postings, dfs):
        idf = math.log((N + 1.0) / (df + 1.0)) + 1.0

        tf = float(p.tf)
        if use_importance_boost:
            tf += 2.0 * p.title_tf + 1.5 * p.header_tf + 1.2 * p.bold_tf

        # log-tf helps stability
        tfw = 1.0 + math.log(tf) if tf > 0 else 0.0
        score += tfw * idf
    return score


def search_and(
    query: str,
    index_path: str,
    lexicon: Dict[str, Tuple[int, int]],
    tokenizer: Tokenizer,
    N: int,
    topk: int = 10,
    rank: bool = True
) -> List[Tuple[int, float]]:
    """
    Returns list of (doc_id, score) sorted by score desc if rank=True,
    else score is 0 and order is arbitrary.
    """
    terms = tokenizer.tokenize(query)
    if not terms:
        return []

    postings_lists: List[List[Posting]] = []
    dfs: List[int] = []

    for t in terms:
        if t not in lexicon:
            return []  # AND-only: any missing term => empty
        offset, df = lexicon[t]
        _, _, postings = read_postings_at(index_path, offset)
        postings_lists.append(postings)
        dfs.append(df)

    doc_to_postings = intersect_docsets(postings_lists)
    if not doc_to_postings:
        return []

    if not rank:
        return [(doc_id, 0.0) for doc_id in list(doc_to_postings.keys())[:topk]]

    # Need postings aligned with dfs (same term order as query)
    # intersect_docsets currently preserves postings in the sorted-by-df order,
    # so we must reorder dfs consistently:
    # easiest: rebuild in the same sorted order
    # We sort pairs (df, postings) together earlier in intersect step; do it here:
    pairs = list(zip(dfs, postings_lists))
    pairs.sort(key=lambda x: len(x[1]))
    dfs_sorted = [x[0] for x in pairs]

    scored = []
    for doc_id, plist in doc_to_postings.items():
        s = score_doc_tf_idf(plist, dfs_sorted, N, use_importance_boost=True)
        scored.append((doc_id, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk]


def interactive_loop(args):
    tokenizer = Tokenizer(use_stem=not args.no_stem)
    N, urls = load_docmap(args.docmap)
    lexicon = load_lexicon(args.lexicon)

    print("=== Search Interface (AND-only) ===")
    print("Type a query (space-separated terms). Empty line to quit.")
    print(f"Ranking: {'TF-IDF' if args.rank else 'OFF'} | Stem: {not args.no_stem}")
    while True:
        q = input("\nquery> ").strip()
        if not q:
            break
        results = search_and(
            query=q,
            index_path=args.index,
            lexicon=lexicon,
            tokenizer=tokenizer,
            N=N,
            topk=args.topk,
            rank=args.rank
        )
        if not results:
            print("(no results)")
            continue
        for i, (doc_id, score) in enumerate(results, start=1):
            print(f"{i:02d}. {urls[doc_id]}  (score={score:.4f})")


def main():
    ap = argparse.ArgumentParser(description="Milestone 2 Search (AND-only) over disk index")
    ap.add_argument("--index", required=True, help="Path to index_final.jsonl")
    ap.add_argument("--lexicon", required=True, help="Path to lexicon.tsv")
    ap.add_argument("--docmap", required=True, help="Path to doc_id_to_url.json")
    ap.add_argument("--topk", type=int, default=10, help="Top K results to show")
    ap.add_argument("--rank", action="store_true", default=True, help="Enable TF-IDF ranking (default ON)")
    ap.add_argument("--no-rank", dest="rank", action="store_false", help="Disable ranking (AND-only, unsorted)")
    ap.add_argument("--no-stem", action="store_true", help="Disable stemming (default stems)")
    ap.add_argument("--interactive", action="store_true", help="Run interactive console UI (for screenshot)")
    ap.add_argument("--query", type=str, default=None, help="Run a single query and print results")
    args = ap.parse_args()

    if args.interactive or args.query is None:
        interactive_loop(args)
        return

    tokenizer = Tokenizer(use_stem=not args.no_stem)
    N, urls = load_docmap(args.docmap)
    lexicon = load_lexicon(args.lexicon)

    results = search_and(
        query=args.query,
        index_path=args.index,
        lexicon=lexicon,
        tokenizer=tokenizer,
        N=N,
        topk=args.topk,
        rank=args.rank
    )
    for i, (doc_id, score) in enumerate(results, start=1):
        print(f"{i:02d}. {urls[doc_id]}\t{score:.6f}")


if __name__ == "__main__":
    main()