from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


@dataclass
class BaselineScores:
    isolation_forest: np.ndarray
    pca_recon: np.ndarray


class BaselineRunner:
    def __init__(self, if_params: dict, pca_params: dict):
        self.if_model = IsolationForest(**if_params)
        self.pca = PCA(**pca_params)

    def fit_normal(self, X: np.ndarray, y: np.ndarray) -> None:
        Xn = X[y == 0]
        self.if_model.fit(Xn)
        self.pca.fit(Xn)

    def score(self, X: np.ndarray) -> BaselineScores:
        if_scores = -self.if_model.score_samples(X)
        Z = self.pca.transform(X)
        X_hat = self.pca.inverse_transform(Z)
        pca_scores = np.mean((X - X_hat) ** 2, axis=1)
        return BaselineScores(isolation_forest=if_scores, pca_recon=pca_scores)
