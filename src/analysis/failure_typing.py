from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import DBSCAN, KMeans


def cluster_failures(z_anom: np.ndarray, method: str = "dbscan", eps: float = 0.8, min_samples: int = 5, k: int = 4) -> np.ndarray:
    if len(z_anom) == 0:
        return np.array([], dtype=int)
    if method == "kmeans":
        model = KMeans(n_clusters=min(k, len(z_anom)), random_state=42, n_init=10)
    else:
        model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(z_anom)


def summarize_failure_clusters(cluster_labels: np.ndarray, anomaly_sequences: list[list[str]]) -> dict[str, dict[str, object]]:
    bucket: dict[int, list[list[str]]] = defaultdict(list)
    for c, seq in zip(cluster_labels, anomaly_sequences):
        bucket[int(c)].append(seq)

    summary: dict[str, dict[str, object]] = {}
    for c, seqs in bucket.items():
        counts = Counter(e for s in seqs for e in s)
        summary[f"cluster_{c}"] = {
            "size": len(seqs),
            "top_events": [ev for ev, _ in counts.most_common(10)],
            "label": f"failure_type_{c}",
        }
    return summary
