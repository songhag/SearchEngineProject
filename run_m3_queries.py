from __future__ import annotations

import argparse
import json

from tokenizer import Tokenizer
from search import load_docmap, load_lexicon, search_and


QUERIES = [
    # good queries
    "cristina lopes",
    "machine learning",
    "ACM",
    "master of software engineering",
    "computer science",
    "data science",
    "information retrieval",
    "software engineering",
    "ics faculty",
    "graduate admissions",

    # likely poor queries
    "who is cristina lopes",
    "where is the ics department",
    "contact computer science department",
    "graduate admission requirements",
    "undergraduate academic advising",
    "faculty directory computer science",
    "phd program application",
    "machine learning faculty uc irvine",
    "software engineering prerequisites",
    "office hours admissions"
]


def main():
    ap = argparse.ArgumentParser(description="Run Milestone 3 query evaluation")
    ap.add_argument("--index", required=True)
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--docmap", required=True)
    ap.add_argument("--out", default=None)

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
            topk=10,
            rank=True
        )

        top_urls = [urls[doc_id] for doc_id, _ in results]
        all_results[q] = top_urls

        print("\n==============================")
        print(f"Query: {q}")

        if not top_urls:
            print("(no results)")
        else:
            for i, u in enumerate(top_urls, start=1):
                print(f"{i}. {u}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved results JSON to: {args.out}")


if __name__ == "__main__":
    main()