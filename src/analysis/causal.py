from __future__ import annotations

from collections import Counter, defaultdict

import networkx as nx


def build_event_causal_graph(
    sequences: list[list[str]],
    labels: list[int],
    top_k: int = 30,
) -> nx.DiGraph:
    """Approximate causal dependencies by anomaly-conditional temporal precedence."""
    precede_counts: Counter[tuple[str, str]] = Counter()
    anomaly_precede_counts: Counter[tuple[str, str]] = Counter()

    for seq, y in zip(sequences, labels):
        for i in range(len(seq) - 1):
            edge = (seq[i], seq[i + 1])
            precede_counts[edge] += 1
            if y == 1:
                anomaly_precede_counts[edge] += 1

    scores = []
    for edge, anom_c in anomaly_precede_counts.items():
        total_c = precede_counts[edge]
        score = (anom_c + 1) / (total_c + 1)
        scores.append((edge, score, anom_c, total_c))

    scores.sort(key=lambda x: x[1], reverse=True)
    g = nx.DiGraph()
    for (a, b), score, anom_c, total_c in scores[:top_k]:
        g.add_edge(a, b, anomaly_bias=float(score), anomaly_count=int(anom_c), total_count=int(total_c))
    return g


def graph_to_dict(g: nx.DiGraph) -> dict[str, list[dict[str, object]]]:
    edges = []
    for u, v, d in g.edges(data=True):
        edges.append({"from": u, "to": v, **d})
    return {"edges": edges}
