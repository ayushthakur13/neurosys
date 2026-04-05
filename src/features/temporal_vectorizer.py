from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemporalBatch:
    token_ids: np.ndarray
    positions: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    vocab: dict[str, int]


class SequenceAwareVectorizer:
    """Turn ordered event sequences into padded token and position tensors.

    This is the first step toward a temporal VAE: it preserves order, exposes
    sequence length, and keeps a dedicated PAD/UNK pathway for variable-length data.
    """

    def __init__(
        self,
        min_count: int = 1,
        max_vocab_size: int | None = None,
        max_sequence_length: int | None = None,
        unknown_token: str = "<UNK>",
        pad_token: str = "<PAD>",
        positional_encoding_type: str = "absolute",
    ):
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.unknown_token = unknown_token
        self.pad_token = pad_token
        self.positional_encoding_type = positional_encoding_type
        self.vocab: dict[str, int] = {}
        self.pad_index: int = 0
        self.unknown_index: int = 1

    def fit(self, sequences: list[list[str]]) -> "SequenceAwareVectorizer":
        counts: dict[str, int] = {}
        for sequence in sequences:
            for event in sequence:
                counts[event] = counts.get(event, 0) + 1

        kept = [(event, count) for event, count in counts.items() if count >= self.min_count]
        kept.sort(key=lambda item: (-item[1], item[0]))
        if self.max_vocab_size is not None:
            kept = kept[: self.max_vocab_size]

        self.vocab = {
            self.pad_token: self.pad_index,
            self.unknown_token: self.unknown_index,
        }
        for event, _ in kept:
            if event not in self.vocab:
                self.vocab[event] = len(self.vocab)
        return self

    def _encode_sequence(self, sequence: list[str]) -> tuple[list[int], list[int], list[int]]:
        if self.max_sequence_length is not None:
            sequence = sequence[: self.max_sequence_length]

        token_ids: list[int] = []
        positions: list[int] = []
        mask: list[int] = []

        for index, event in enumerate(sequence):
            token_ids.append(self.vocab.get(event, self.unknown_index))
            positions.append(index if self.positional_encoding_type == "absolute" else 0)
            mask.append(1)

        return token_ids, positions, mask

    def transform(self, sequences: list[list[str]]) -> TemporalBatch:
        if not self.vocab:
            raise ValueError("SequenceAwareVectorizer must be fit before transform")

        lengths = np.array(
            [min(len(sequence), self.max_sequence_length) if self.max_sequence_length is not None else len(sequence) for sequence in sequences],
            dtype=np.int64,
        )
        max_len = int(lengths.max()) if len(lengths) else 0
        token_ids = np.full((len(sequences), max_len), self.pad_index, dtype=np.int64)
        positions = np.zeros((len(sequences), max_len), dtype=np.int64)
        mask = np.zeros((len(sequences), max_len), dtype=np.float32)

        for row_index, sequence in enumerate(sequences):
            encoded_tokens, encoded_positions, encoded_mask = self._encode_sequence(sequence)
            seq_len = len(encoded_tokens)
            if seq_len == 0:
                continue
            token_ids[row_index, :seq_len] = encoded_tokens
            positions[row_index, :seq_len] = encoded_positions
            mask[row_index, :seq_len] = encoded_mask

        return TemporalBatch(
            token_ids=token_ids,
            positions=positions,
            mask=mask,
            lengths=lengths,
            vocab=self.vocab,
        )

    def fit_transform(self, sequences: list[list[str]]) -> TemporalBatch:
        self.fit(sequences)
        return self.transform(sequences)