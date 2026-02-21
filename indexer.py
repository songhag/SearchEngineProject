import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from io_utils import iter_json_docs, ensure_dir, file_size_kb, write_json
from html_utils import extract_zoned_text
from tokenizer import Tokenizer
from merge_utils import dump_partial_index, merge_partials, Posting


def build_partial_index_for_doc(
    doc_id: int,
    zoned_text: Dict[str, str],
    tokenizer: Tokenizer,
) -> Dict[str, Posting]:
    """
    Build term stats for a single doc.
    Returns: term -> posting tuple for THIS doc only:
      (doc_id, tf, title_tf, header_tf, bold_tf)
    """
    title_tokens = tokenizer.tokenize(zoned_text.get("title", ""))
    header_tokens = tokenizer.tokenize(zoned_text.get("headers", ""))
    bold_tokens = tokenizer.tokenize(zoned_text.get("bold", ""))
    body_tokens = tokenizer.tokenize(zoned_text.get("body", ""))

    c_title = Counter(title_tokens)
    c_header = Counter(header_tokens)
    c_bold = Counter(bold_tokens)
    c_body = Counter(body_tokens)

    # total tf = body + title + header + bold
    terms = set(c_title) | set(c_header) | set(c_bold) | set(c_body)
    per_doc = {}
    for t in terms:
        title_tf = c_title.get(t, 0)
        header_tf = c_header.get(t, 0)
        bold_tf = c_bold.get(t, 0)
        body_tf = c_body.get(t, 0)
        tf = title_tf + header_tf + bold_tf + body_tf
        per_doc[t] = (doc_id, tf, title_tf, header_tf, bold_tf)
    return per_doc


def main():
    ap = argparse.ArgumentParser(description="CS121 Search Engine - Milestone 1 Indexer")
    ap.add_argument("--corpus", required=True, help="Path to extracted dataset root (e.g., ./developer/ or ./analyst/)")
    ap.add_argument("--out", default="./out_m1", help="Output directory for index files")
    ap.add_argument("--use-stem", action="store_true", default=True, help="Enable Porter stemming (default on)")
    ap.add_argument("--no-stem", dest="use_stem", action="store_false", help="Disable stemming")
    ap.add_argument("--flush-docs", type=int, default=6000,
                    help="Flush a partial index to disk every N documents (controls memory). Default 6000.")
    args = ap.parse_args()

    ensure_dir(args.out)
    partial_dir = os.path.join(args.out, "partials")
    ensure_dir(partial_dir)

    tokenizer = Tokenizer(use_stem=args.use_stem)

    # doc_id -> url
    doc_map: List[str] = []
    # SPIMI in-memory partial index
    partial_index: Dict[str, List[Posting]] = defaultdict(list)

    doc_count = 0
    partial_count = 0

    def flush_partial():
        nonlocal partial_index, partial_count
        if not partial_index:
            return
        partial_path = os.path.join(partial_dir, f"partial_{partial_count:04d}.jsonl")
        dump_partial_index(partial_path, partial_index)
        partial_count += 1
        partial_index = defaultdict(list)
    # loop through all doc
    for _, url, content in iter_json_docs(args.corpus):
        doc_id = doc_count
        doc_map.append(url)

        zoned = extract_zoned_text(content)
        per_doc = build_partial_index_for_doc(doc_id, zoned, tokenizer)

        # append postings
        for term, posting in per_doc.items():
            partial_index[term].append(posting)

        doc_count += 1

        if doc_count % args.flush_docs == 0:
            flush_partial()

    # final flush
    flush_partial()

    # merge partials
    partial_paths = [
        os.path.join(partial_dir, fn)
        for fn in sorted(os.listdir(partial_dir))
        if fn.endswith(".jsonl")
    ]
    final_index_path = os.path.join(args.out, "index_final.jsonl")
    unique_terms = merge_partials(partial_paths, final_index_path)

    # write doc map
    docmap_path = os.path.join(args.out, "doc_id_to_url.json")
    write_json(docmap_path, {"doc_count": doc_count, "doc_id_to_url": doc_map})

    # compute size
    index_kb = file_size_kb(final_index_path)

    stats = {
        "indexed_documents": doc_count,
        "unique_tokens": unique_terms,
        "index_size_kb": index_kb,
        "final_index_path": final_index_path,
        "docmap_path": docmap_path,
        "partials_written": len(partial_paths),
        "stemming": args.use_stem,
        "flush_docs": args.flush_docs,
    }
    write_json(os.path.join(args.out, "m1_stats.json"), stats)

    print("=== Analytics ===")
    print(f"Indexed documents: {doc_count}")
    print(f"Unique tokens:     {unique_terms}")
    print(f"Index size (KB):   {index_kb:.2f}")
    print(f"Final index:       {final_index_path}")
    print(f"Doc map:           {docmap_path}")
    print(f"Partials:          {len(partial_paths)}")


if __name__ == "__main__":
    main()