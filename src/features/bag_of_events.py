from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import normalize


@dataclass
class BoEResult:
    X: np.ndarray
    vocab: dict[str, int]


class BagOfEventsVectorizer:
    def __init__(self, min_count: int = 1, norm: str = "l2", max_vocab_size: int | None = None, unknown_token: str | None = "<UNK>"):
        self.min_count = min_count
        self.norm = norm
        self.max_vocab_size = max_vocab_size
        self.unknown_token = unknown_token
        self.vocab: dict[str, int] = {}
        self.unknown_index: int | None = None

    def fit(self, sequences: list[list[str]]) -> "BagOfEventsVectorizer":
        counts: dict[str, int] = {}
        for seq in sequences:
            for e in seq:
                counts[e] = counts.get(e, 0) + 1
        kept = [(e, c) for e, c in counts.items() if c >= self.min_count]
        kept.sort(key=lambda item: (-item[1], item[0]))
        if self.max_vocab_size is not None:
            kept = kept[: self.max_vocab_size]

        self.vocab = {e: i for i, (e, _) in enumerate(kept)}
        self.unknown_index = None
        if self.unknown_token is not None:
            self.unknown_index = len(self.vocab)
            self.vocab[self.unknown_token] = self.unknown_index
        return self

    def transform(self, sequences: list[list[str]]) -> np.ndarray:
        X = np.zeros((len(sequences), len(self.vocab)), dtype=np.float32)
        for i, seq in enumerate(sequences):
            for e in seq:
                j = self.vocab.get(e)
                if j is not None:
                    X[i, j] += 1.0
                elif self.unknown_index is not None:
                    X[i, self.unknown_index] += 1.0
        if self.norm:
            X = normalize(X, norm=self.norm)
        return X

    def fit_transform(self, sequences: list[list[str]]) -> BoEResult:
        self.fit(sequences)
        return BoEResult(X=self.transform(sequences), vocab=self.vocab)
