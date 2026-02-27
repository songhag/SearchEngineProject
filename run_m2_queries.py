from __future__ import annotations

import argparse
import json

from tokenizer import Tokenizer
from search import load_docmap, load_lexicon, search_and


QUERIES = [
    "cristina lopes",
    "machine learning",
    "ACM",
    "master of software engineering",
]


def main():
    ap = argparse.ArgumentParser(description="Run required Milestone 2 queries and print top 5 URLs each")
    ap.add_argument("--index", required=True, help="Path to index_final.jsonl")
    ap.add_argument("--lexicon", required=True, help="Path to lexicon.tsv")
    ap.add_argument("--docmap", required=True, help="Path to doc_id_to_url.json")
    ap.add_argument("--out", default=None, help="Optional: write results JSON for your report")
    args = ap.parse_args()

    tokenizer = Tokenizer(use_stem=True)
    N, urls = load_docmap(args.docmap)
    lexicon = load_lexicon(args.lexicon)

    all_results = {}
    for q in QUERIES:
        results = search_and(
            query=q,
            index_path=args.index,
            lexicon=lexicon,
            tokenizer=tokenizer,
            N=N,
            topk=5,
            rank=True
        )
        top_urls = [urls[doc_id] for doc_id, _ in results]
        all_results[q] = top_urls

        print("\n==============================")
        print(f"Query: {q}")
        for i, u in enumerate(top_urls, start=1):
            print(f"{i}. {u}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved results JSON to: {args.out}")


if __name__ == "__main__":
    main()