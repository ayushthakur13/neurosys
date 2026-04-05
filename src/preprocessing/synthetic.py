from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SyntheticInjectionResult:
    sequences: list[list[str]]
    labels: list[int]
    injected_indices: list[int]


class SyntheticInjector:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def inject(
        self,
        sequences: list[list[str]],
        labels: list[int],
        ratio: float = 0.1,
        modes: list[str] | None = None,
    ) -> SyntheticInjectionResult:
        modes = modes or ["insert", "remove", "shuffle"]
        all_events = [e for seq in sequences for e in seq]
        if not all_events:
            return SyntheticInjectionResult(sequences=sequences, labels=labels, injected_indices=[])

        n = max(1, int(len(sequences) * ratio))
        candidates = [i for i, y in enumerate(labels) if y == 0]
        self.rng.shuffle(candidates)
        chosen = candidates[:n]

        new_sequences = [seq.copy() for seq in sequences]
        new_labels = labels.copy()

        for idx in chosen:
            mode = self.rng.choice(modes)
            seq = new_sequences[idx]
            if not seq:
                continue
            if mode == "insert":
                pos = self.rng.randrange(len(seq))
                seq.insert(pos, self.rng.choice(all_events))
            elif mode == "remove" and len(seq) > 1:
                pos = self.rng.randrange(len(seq))
                seq.pop(pos)
            elif mode == "shuffle" and len(seq) > 3:
                i = self.rng.randrange(len(seq) - 2)
                j = min(len(seq), i + 3)
                chunk = seq[i:j]
                self.rng.shuffle(chunk)
                seq[i:j] = chunk
            new_labels[idx] = 1

        return SyntheticInjectionResult(sequences=new_sequences, labels=new_labels, injected_indices=chosen)
