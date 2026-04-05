from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .hdfs import SequenceDataset


def _parse_sequence_line(line: str) -> tuple[str, list[str]] | None:
    text = line.strip()
    if not text:
        return None
    if "," not in text:
        return None
    block_id, seq = text.split(",", 1)
    events = [tok.strip() for tok in seq.split() if tok.strip()]
    return block_id.strip(), events


class HDFSXuSplitPreprocessor:
    """Loads the hdfs_xu split format with train/normal-test/abnormal-test files."""

    def __init__(self, data_root: Path, train_file: str, normal_file: str, abnormal_file: str):
        self.train_path = data_root / train_file
        self.normal_path = data_root / normal_file
        self.abnormal_path = data_root / abnormal_file

    def _read_file(self, path: Path, label: int) -> tuple[list[str], list[list[str]], list[int]]:
        block_ids: list[str] = []
        seqs: list[list[str]] = []
        labels: list[int] = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parsed = _parse_sequence_line(line)
                if not parsed:
                    continue
                block_id, events = parsed
                block_ids.append(block_id)
                seqs.append(events)
                labels.append(label)
        return block_ids, seqs, labels

    def build_split_dataset(self) -> "HDFSXuSplitDataset":
        tr_b, tr_s, tr_y = self._read_file(self.train_path, label=0)
        no_b, no_s, no_y = self._read_file(self.normal_path, label=0)
        ab_b, ab_s, ab_y = self._read_file(self.abnormal_path, label=1)

        return HDFSXuSplitDataset(
            train_block_ids=tr_b,
            train_sequences=tr_s,
            train_labels=tr_y,
            normal_block_ids=no_b,
            normal_sequences=no_s,
            normal_labels=no_y,
            abnormal_block_ids=ab_b,
            abnormal_sequences=ab_s,
            abnormal_labels=ab_y,
        )

    def build_sequences(self) -> SequenceDataset:
        split = self.build_split_dataset()

        return SequenceDataset(
            block_ids=split.block_ids,
            sequences=split.sequences,
            labels=split.labels,
            templates={},
        )


@dataclass
class HDFSXuSplitDataset:
    train_block_ids: list[str]
    train_sequences: list[list[str]]
    train_labels: list[int]
    normal_block_ids: list[str]
    normal_sequences: list[list[str]]
    normal_labels: list[int]
    abnormal_block_ids: list[str]
    abnormal_sequences: list[list[str]]
    abnormal_labels: list[int]

    @property
    def block_ids(self) -> list[str]:
        return self.train_block_ids + self.normal_block_ids + self.abnormal_block_ids

    @property
    def sequences(self) -> list[list[str]]:
        return self.train_sequences + self.normal_sequences + self.abnormal_sequences

    @property
    def labels(self) -> list[int]:
        return self.train_labels + self.normal_labels + self.abnormal_labels

    @property
    def train_normal_sequences(self) -> list[list[str]]:
        return self.train_sequences

    @property
    def eval_block_ids(self) -> list[str]:
        return self.normal_block_ids + self.abnormal_block_ids

    @property
    def eval_sequences(self) -> list[list[str]]:
        return self.normal_sequences + self.abnormal_sequences

    @property
    def eval_labels(self) -> list[int]:
        return self.normal_labels + self.abnormal_labels

    def split_train_validation(self, validation_ratio: float = 0.2) -> tuple[list[list[str]], list[int], list[list[str]], list[int]]:
        """Split the train split into train-normal and validation-normal subsets."""
        if not self.train_sequences:
            return [], [], [], []

        split_index = max(1, int(len(self.train_sequences) * (1.0 - validation_ratio)))
        split_index = min(split_index, len(self.train_sequences) - 1) if len(self.train_sequences) > 1 else len(self.train_sequences)
        train_sequences = self.train_sequences[:split_index]
        train_labels = self.train_labels[:split_index]
        validation_sequences = self.train_sequences[split_index:]
        validation_labels = self.train_labels[split_index:]
        return train_sequences, train_labels, validation_sequences, validation_labels
