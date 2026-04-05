from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from analysis import counterfactual_event_shift
from features import BagOfEventsVectorizer, SequenceAwareVectorizer
from models import TemporalVAEConfig, TemporalVAETrainer, VAEConfig, VAETrainer
from utils.io import read_json


class NeuroSysService:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.vec = None
        self.vae: VAETrainer | None = None
        self.threshold: float = 0.0
        self.representation: str = "bag_of_events"

    def load(self) -> None:
        metadata = read_json(self.artifact_dir / "vocab.json")
        self.representation = str(metadata.get("representation", "bag_of_events"))
        self.threshold = float(metadata.get("decision_threshold", 0.0))

        vocab = metadata["vocab"]

        ckpt = torch.load(self.artifact_dir / "vae.pt", map_location="cpu")
        if self.representation == "temporal":
            cfg = TemporalVAEConfig(**ckpt["config"])
            trainer = TemporalVAETrainer(cfg, device="cpu")
            trainer.model.load_state_dict(ckpt["state_dict"])
            self.vae = trainer

            self.vec = SequenceAwareVectorizer(
                min_count=1,
                max_vocab_size=None,
                max_sequence_length=None,
                unknown_token="<UNK>",
                pad_token="<PAD>",
            )
            self.vec.vocab = {str(k): int(v) for k, v in vocab.items()}
            self.vec.pad_index = int(metadata.get("pad_index", 0))
            self.vec.unknown_index = int(metadata.get("unknown_index", 1)) if metadata.get("unknown_index") is not None else None
        else:
            cfg = VAEConfig(**ckpt["config"])
            trainer = VAETrainer(cfg, device="cpu")
            trainer.model.load_state_dict(ckpt["state_dict"])
            self.vae = trainer

            self.vec = BagOfEventsVectorizer()
            self.vec.vocab = {str(k): int(v) for k, v in vocab.items()}

    def to_feature(self, sequence: list[str]):
        if self.vec is None:
            raise RuntimeError("Service not loaded")
        return self.vec.transform([sequence])

    def detect(self, sequence: list[str]) -> tuple[float, bool]:
        if self.vae is None:
            raise RuntimeError("Service not loaded")
        feature = self.to_feature(sequence)
        if self.representation == "temporal":
            score = float(self.vae.reconstruction_error(feature.token_ids, feature.mask)[0])
        else:
            score = float(self.vae.reconstruction_error(feature)[0])
        return score, score >= self.threshold

    def latent(self, sequence: list[str]) -> list[float]:
        if self.vae is None:
            raise RuntimeError("Service not loaded")
        feature = self.to_feature(sequence)
        if self.representation == "temporal":
            return self.vae.latent(feature.token_ids, feature.mask)[0].tolist()
        return self.vae.latent(feature)[0].tolist()

    def root_cause(self, sequence: list[str]) -> dict:
        if self.vae is None:
            raise RuntimeError("Service not loaded")
        feature = self.to_feature(sequence)
        event_names = [e for e, _ in sorted(self.vec.vocab.items(), key=lambda kv: kv[1])]

        if self.representation == "temporal":
            current_score = float(self.vae.reconstruction_error(feature.token_ids, feature.mask)[0])
            changes: list[dict[str, object]] = []
            edited = sequence.copy()
            for event_name in dict.fromkeys(sequence):
                candidate = edited.copy()
                if event_name not in candidate:
                    continue
                candidate.remove(event_name)
                candidate_feature = self.vec.transform([candidate])
                new_score = float(self.vae.reconstruction_error(candidate_feature.token_ids, candidate_feature.mask)[0])
                if new_score < current_score:
                    changes.append(
                        {
                            "event": event_name,
                            "score_before": current_score,
                            "score_after": new_score,
                        }
                    )
                    edited = candidate
                    current_score = new_score
                if len(changes) >= 5:
                    break
            return {"num_changes": len(changes), "changes": changes, "final_score": current_score}

        x = feature[0]
        return counterfactual_event_shift(x, score_fn=lambda xb: self.vae.reconstruction_error(xb), event_names=event_names)
