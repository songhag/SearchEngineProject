import json
import os
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional


@dataclass
class DocRecord:
    doc_id: int
    url: str
    content: str


def _strip_fragment(url: str) -> str:
    # ignore fragment if present
    if not url:
        return url
    return url.split("#", 1)[0]


def iter_json_docs(root_dir: str) -> Iterator[Tuple[str, str, str]]:
    """
    Yield (path, url, content) for every JSON file under root_dir.
    one folder per domain, many JSON files inside.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.endswith(".json"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    obj = json.load(f)
                url = _strip_fragment(obj.get("url", "") or "")
                content = obj.get("content", "") or ""
                yield path, url, content
            except Exception:
                continue


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def file_size_kb(path: str) -> float:
    return os.path.getsize(path) / 1024.0


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)