import re
from typing import List, Optional
from nltk.stem import PorterStemmer


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class Tokenizer:
    def __init__(self, use_stem: bool = True):
        self.use_stem = use_stem
        self.stemmer: Optional[PorterStemmer] = PorterStemmer() if use_stem else None

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = _TOKEN_RE.findall(text.lower())
        if self.use_stem and self.stemmer is not None:
            return [self.stemmer.stem(t) for t in tokens]
        return tokens