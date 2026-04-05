from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TemporalVAEConfig:
    vocab_size: int
    pad_index: int = 0
    unknown_index: int = 1
    embedding_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 16
    beta: float = 1.0
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    beta_warmup_epochs: int = 0


class TemporalVAE(nn.Module):
    def __init__(self, cfg: TemporalVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim, padding_idx=cfg.pad_index)
        self.encoder = nn.GRU(cfg.embedding_dim, cfg.hidden_dim, batch_first=True)
        self.mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.decoder_init = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.decoder = nn.GRU(cfg.embedding_dim, cfg.hidden_dim, batch_first=True)
        self.output = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def encode(self, token_ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(token_ids)
        lengths = mask.sum(dim=1).long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        hidden = hidden[-1]
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        hidden = torch.tanh(self.decoder_init(z)).unsqueeze(0)
        outputs, _ = self.decoder(embedded, hidden)
        logits = self.output(outputs)
        return logits

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(token_ids, mask)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, token_ids, mask)
        return logits, mu, logvar


class TemporalVAETrainer:
    def __init__(self, cfg: TemporalVAEConfig, device: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TemporalVAE(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def _beta_for_epoch(self, epoch: int) -> float:
        if self.cfg.beta_warmup_epochs <= 0:
            return self.cfg.beta
        return float(self.cfg.beta) * min(1.0, epoch / max(1, self.cfg.beta_warmup_epochs))

    def _loss(self, token_ids: torch.Tensor, logits: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            token_ids.reshape(-1),
            reduction="none",
        )
        recon_loss = (recon_loss * mask.reshape(-1)).sum() / mask.sum().clamp(min=1.0)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kld
        return loss, recon_loss, kld

    def fit(self, token_ids: np.ndarray, mask: np.ndarray) -> list[dict[str, float]]:
        x_ids = torch.tensor(token_ids, dtype=torch.long)
        x_mask = torch.tensor(mask, dtype=torch.float32)
        loader = DataLoader(TensorDataset(x_ids, x_mask), batch_size=self.cfg.batch_size, shuffle=True)
        history: list[dict[str, float]] = []

        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            beta = self._beta_for_epoch(epoch)
            total_loss = total_recon = total_kld = 0.0
            for batch_ids, batch_mask in loader:
                batch_ids = batch_ids.to(self.device)
                batch_mask = batch_mask.to(self.device)
                self.optimizer.zero_grad()
                logits, mu, logvar = self.model(batch_ids, batch_mask)
                loss, recon_loss, kld = self._loss(batch_ids, logits, mu, logvar, batch_mask, beta)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld.item()

            n = max(len(loader), 1)
            history.append({"epoch": float(epoch), "loss": total_loss / n, "reconstruction": total_recon / n, "kld": total_kld / n, "beta": float(beta)})
        return history

    @torch.no_grad()
    def reconstruction_error(self, token_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        x_mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
        logits, _, _ = self.model(x_ids, x_mask)
        recon = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x_ids.reshape(-1),
            reduction="none",
        )
        recon = recon.reshape(x_ids.shape[0], x_ids.shape[1])
        return ((recon * x_mask).sum(dim=1) / x_mask.sum(dim=1).clamp(min=1.0)).cpu().numpy()

    @torch.no_grad()
    def latent(self, token_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        x_mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
        mu, _ = self.model.encode(x_ids, x_mask)
        return mu.cpu().numpy()