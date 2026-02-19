import heapq
import json
from typing import Dict, List, Tuple, Iterator, Any


Posting = Tuple[int, int, int, int, int]  # (doc_id, tf, title_tf, header_tf, bold_tf)


def dump_partial_index(path: str, index: Dict[str, List[Posting]]) -> None:
    """
    Write a partial index as JSONL:
      {"term":"...", "postings":[[doc_id, tf, title_tf, header_tf, bold_tf], ...]}
    Terms are written in sorted order.
    """
    with open(path, "w", encoding="utf-8") as f:
        for term in sorted(index.keys()):
            postings = index[term]
            # postings already unique per doc_id in our construction
            f.write(json.dumps({"term": term, "postings": postings}, ensure_ascii=False))
            f.write("\n")


def iter_partial(path: str) -> Iterator[Tuple[str, List[Posting]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            yield obj["term"], [tuple(p) for p in obj["postings"]]


def merge_postings(a: List[Posting], b: List[Posting]) -> List[Posting]:
    """
    Merge two postings lists
    If same doc_id appears, sum the fields.
    """
    by = {}
    for p in a:
        by[p[0]] = p
    for p in b:
        doc_id = p[0]
        if doc_id in by:
            pa = by[doc_id]
            by[doc_id] = (
                doc_id,
                pa[1] + p[1],
                pa[2] + p[2],
                pa[3] + p[3],
                pa[4] + p[4],
            )
        else:
            by[doc_id] = p
    return [by[k] for k in sorted(by.keys())]


def merge_partials(partial_paths: List[str], out_path: str) -> int:
    """
    K-way merge all partial JSONL indexes into one final JSONL index.
    Returns: number of unique terms.
    Final format (JSONL):
      {"term":"...", "df":123, "postings":[[doc_id, tf, title_tf, header_tf, bold_tf], ...]}
    """
    iters = [iter_partial(p) for p in partial_paths]
    heap: List[Tuple[str, int, List[Posting]]] = []

    # prime heap
    for i, it in enumerate(iters):
        try:
            term, postings = next(it)
            heapq.heappush(heap, (term, i, postings))
        except StopIteration:
            pass

    unique_terms = 0
    with open(out_path, "w", encoding="utf-8") as out:
        current_term = None
        current_postings: List[Posting] = []

        while heap:
            term, i, postings = heapq.heappop(heap)

            if current_term is None:
                current_term = term
                current_postings = postings
            elif term == current_term:
                current_postings = merge_postings(current_postings, postings)
            else:
                # flush previous
                df = len(current_postings)
                out.write(json.dumps(
                    {"term": current_term, "df": df, "postings": current_postings},
                    ensure_ascii=False
                ))
                out.write("\n")
                unique_terms += 1
                # start new
                current_term = term
                current_postings = postings

            try:
                nxt_term, nxt_postings = next(iters[i])
                heapq.heappush(heap, (nxt_term, i, nxt_postings))
            except StopIteration:
                pass

        # flush last
        if current_term is not None:
            df = len(current_postings)
            out.write(json.dumps(
                {"term": current_term, "df": df, "postings": current_postings},
                ensure_ascii=False
            ))
            out.write("\n")
            unique_terms += 1

    return unique_terms