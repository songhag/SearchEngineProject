from __future__ import annotations

import argparse
import json
import os


def build_lexicon(index_path: str, lexicon_path: str) -> int:
    """
    Build a lexicon mapping term -> byte offset in index_final.jsonl
    Output format: TSV with columns: term \t offset \t df
    Returns number of terms.
    """
    term_count = 0
    with open(index_path, "rb") as f_in, open(lexicon_path, "w", encoding="utf-8") as f_out:
        while True:
            offset = f_in.tell()
            line = f_in.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line.decode("utf-8"))
            term = obj["term"]
            df = obj.get("df", 0)
            f_out.write(f"{term}\t{offset}\t{df}\n")
            term_count += 1
    return term_count


def main():
    ap = argparse.ArgumentParser(description="Build lexicon (term -> offset) for index_final.jsonl")
    ap.add_argument("--index", required=True, help="Path to index_final.jsonl")
    ap.add_argument("--out", required=True, help="Output lexicon path (e.g. out_m1/lexicon.tsv)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = build_lexicon(args.index, args.out)
    print(f"Lexicon written to: {args.out}")
    print(f"Unique terms: {n}")


if __name__ == "__main__":
    main()