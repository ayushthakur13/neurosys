from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_latent(z: np.ndarray, method: str = "umap", random_state: int = 42) -> np.ndarray:
    if z.shape[1] <= 2:
        return z[:, :2]

    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(z)

    if method == "tsne":
        return TSNE(n_components=2, random_state=random_state, init="pca").fit_transform(z)

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        return reducer.fit_transform(z)
    except Exception:
        return TSNE(n_components=2, random_state=random_state, init="pca").fit_transform(z)
