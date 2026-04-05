from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class VAEConfig:
    input_dim: int
    hidden_dim: int = 128
    latent_dim: int = 16
    beta: float = 1.0
    lr: float = 1e-3
    epochs: int = 40
    batch_size: int = 64


class VAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class VAETrainer:
    def __init__(self, cfg: VAEConfig, device: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def _loss(self, x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.cfg.beta * kld
        return loss, recon_loss, kld

    def fit(self, X_train_normal: np.ndarray) -> list[dict[str, float]]:
        X = torch.tensor(X_train_normal, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X), batch_size=self.cfg.batch_size, shuffle=True)
        history: list[dict[str, float]] = []

        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            total_loss = total_recon = total_kld = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device)
                self.optimizer.zero_grad()
                recon, mu, logvar = self.model(xb)
                loss, recon_loss, kld = self._loss(xb, recon, mu, logvar)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld.item()

            n = max(len(loader), 1)
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": total_loss / n,
                    "reconstruction": total_recon / n,
                    "kld": total_kld / n,
                }
            )
        return history

    @torch.no_grad()
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        recon, _, _ = self.model(x)
        return torch.mean((recon - x) ** 2, dim=1).cpu().numpy()

    @torch.no_grad()
    def latent(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        mu, _ = self.model.encode(x)
        return mu.cpu().numpy()
