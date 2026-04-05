from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .drain_parser import SimpleDrainParser


@dataclass
class SequenceDataset:
    block_ids: list[str]
    sequences: list[list[str]]
    labels: list[int]
    templates: dict[str, str]


class HDFSPreprocessor:
    block_re = re.compile(r"(blk_-?\d+)")

    def __init__(self, data_root: Path, raw_log: str, labels_csv: str, structured_log: str | None = None):
        self.data_root = data_root
        self.raw_log_path = data_root / raw_log
        self.labels_path = data_root / labels_csv
        self.structured_path = data_root / structured_log if structured_log else None

    def _load_labels(self) -> dict[str, int]:
        df = pd.read_csv(self.labels_path)
        cols = {c.lower(): c for c in df.columns}
        block_col = cols.get("blockid") or cols.get("block_id")
        label_col = cols.get("label")
        if not block_col or not label_col:
            raise ValueError("Labels CSV must contain BlockId and Label columns")
        label_map: dict[str, int] = {}
        for _, row in df.iterrows():
            label_val = str(row[label_col]).strip().lower()
            label_map[str(row[block_col])] = 1 if label_val in {"anomaly", "1", "true"} else 0
        return label_map

    def _extract_block_id(self, line: str) -> str | None:
        m = self.block_re.search(line)
        return m.group(1) if m else None

    def _parse_raw(self) -> pd.DataFrame:
        parser = SimpleDrainParser()
        rows: list[dict[str, str]] = []
        with self.raw_log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                block_id = self._extract_block_id(line)
                if not block_id:
                    continue
                parsed = parser.parse_line(line)
                rows.append(
                    {
                        "LineId": idx,
                        "BlockId": block_id,
                        "EventId": parsed.event_id,
                        "EventTemplate": parsed.event_template,
                        "Content": parsed.raw_line,
                    }
                )
        return pd.DataFrame(rows)

    def _load_structured(self) -> pd.DataFrame:
        df = pd.read_csv(self.structured_path)
        if "BlockId" not in df.columns:
            if "Content" not in df.columns:
                raise ValueError("Structured file needs BlockId column or Content column to extract block ids")
            df["BlockId"] = df["Content"].apply(lambda x: self._extract_block_id(str(x)))
        if "EventId" not in df.columns:
            raise ValueError("Structured file must contain EventId column")
        if "LineId" not in df.columns:
            df["LineId"] = range(len(df))
        return df.dropna(subset=["BlockId"])

    def build_sequences(self) -> SequenceDataset:
        labels = self._load_labels()
        if self.structured_path and self.structured_path.exists():
            structured_df = self._load_structured()
        else:
            structured_df = self._parse_raw()

        grouped = structured_df.sort_values("LineId").groupby("BlockId")["EventId"].apply(list)

        block_ids: list[str] = []
        sequences: list[list[str]] = []
        y: list[int] = []

        for block_id, seq in grouped.items():
            block_ids.append(block_id)
            sequences.append([str(e) for e in seq])
            y.append(labels.get(block_id, 0))

        templates = {}
        if "EventTemplate" in structured_df.columns:
            event_template_df = structured_df[["EventId", "EventTemplate"]].drop_duplicates()
            templates = dict(zip(event_template_df["EventId"].astype(str), event_template_df["EventTemplate"].astype(str)))

        return SequenceDataset(block_ids=block_ids, sequences=sequences, labels=y, templates=templates)
