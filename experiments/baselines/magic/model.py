"""
MAGIC (Masked GAT Autoencoder) — PyG re-implementation.

Faithful to the architecture of Jia et al., "MAGIC: Detecting Advanced
Persistent Threats via Masked Graph Representation Learning" (USENIX
Security 2024) — see also FDUDSDE/MAGIC. We re-implement the model in
PyTorch Geometric instead of DGL because DGL 1.x does not ship wheels for
the Python version of our environment; the architecture, masking strategy,
loss, and downstream KNN-based anomaly scoring all match the reference
code (utils/loaddata.py + model/autoencoder.py + model/eval.py in the
official repo).

Architecture (matches Jia et al. §3.2):
    encoder : N-layer GAT, one-hot input attr → d-dim embedding
    decoder : 1-layer GAT, hidden → input one-hot reconstruction
    masking : 50% of nodes per forward pass replaced with a learnable token
    feature loss   : scaled-cosine error (sce) on masked-node attributes
    structure loss : BCE on a positive/negative-sampled edge classifier

For the entity-level (system-call/Sysmon) regime we use the entity-level
hyperparameters from the official repo eval.py:
    n_layers = 3
    n_hidden = 64
    n_heads  = 4
    mask_rate = 0.5
    alpha_l   = 3 (sce loss exponent)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sce_loss(x_rec: torch.Tensor, x_orig: torch.Tensor,
              alpha: float = 3.0) -> torch.Tensor:
    """Scaled cosine error from MAGIC §3.2 (matches model/loss_func.py)."""
    x_rec = F.normalize(x_rec, p=2, dim=-1)
    x_orig = F.normalize(x_orig, p=2, dim=-1)
    cos = (x_rec * x_orig).sum(dim=-1)
    return ((1 - cos) ** alpha).mean()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MagicConfig:
    in_dim:    int                                  # node attr (one-hot type) dim
    e_dim:     int                                  # edge attr dim (kept as int for compatibility)
    hidden:    int = 64
    n_layers:  int = 3
    n_heads:   int = 4
    mask_rate: float = 0.5
    alpha_l:   float = 3.0
    feat_drop: float = 0.1
    neg_slope: float = 0.2


# ---------------------------------------------------------------------------
# GMAE — Graph Masked Auto-Encoder (PyG implementation)
# ---------------------------------------------------------------------------

class GMAE(nn.Module):
    def __init__(self, cfg: MagicConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden
        head_dim = H // cfg.n_heads
        assert H % cfg.n_heads == 0

        # Encoder
        enc_layers = []
        in_d = cfg.in_dim
        for _ in range(cfg.n_layers):
            enc_layers.append(GATConv(in_d, head_dim, heads=cfg.n_heads,
                                        concat=True, dropout=cfg.feat_drop,
                                        negative_slope=cfg.neg_slope,
                                        add_self_loops=True))
            in_d = H
        self.encoder = nn.ModuleList(enc_layers)
        self.enc_norms = nn.ModuleList([nn.BatchNorm1d(H) for _ in enc_layers])

        # Project concat([all hidden]) to single hidden_dim for decoder
        self.enc_to_dec = nn.Linear(H * cfg.n_layers, H, bias=False)

        # Decoder (1-layer GAT, project back to input space)
        self.decoder = GATConv(H, cfg.in_dim, heads=cfg.n_heads,
                                concat=False, dropout=cfg.feat_drop,
                                negative_slope=cfg.neg_slope,
                                add_self_loops=True)

        # Edge reconstruction MLP
        self.edge_recon = nn.Sequential(
            nn.Linear(H * cfg.n_layers * 2, H),
            nn.LeakyReLU(cfg.neg_slope),
            nn.Linear(H, 1),
        )

        # Learnable mask token (replaces masked node features in input)
        self.mask_token = nn.Parameter(torch.zeros(1, cfg.in_dim))

    # ----- masking helper -----
    @torch.no_grad()
    def _mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.size(0)
        n_mask = int(self.cfg.mask_rate * n)
        if n_mask == 0:
            return x, torch.zeros(0, dtype=torch.long, device=x.device)
        perm = torch.randperm(n, device=x.device)
        mask_idx = perm[:n_mask]
        return mask_idx

    # ----- encoder w/ all-hidden concat -----
    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = x
        all_hidden = []
        for layer, bn in zip(self.encoder, self.enc_norms):
            h = layer(h, edge_index)
            h = bn(h)
            h = F.elu(h)
            all_hidden.append(h)
        return torch.cat(all_hidden, dim=-1)            # [N, H * n_layers]

    # ----- forward / loss -----
    def forward(self, data: Data) -> torch.Tensor:
        return self.compute_loss(data)

    def compute_loss(self, data: Data) -> torch.Tensor:
        x_orig = data.x.float()
        edge_index = data.edge_index

        # mask nodes
        mask_idx = self._mask(x_orig)
        x_masked = x_orig.clone()
        if mask_idx.numel() > 0:
            x_masked[mask_idx] = self.mask_token.to(x_orig.device)

        # encode
        h_concat = self._encode(x_masked, edge_index)
        h_proj = self.enc_to_dec(h_concat)
        recon = self.decoder(h_proj, edge_index)

        # feature reconstruction loss (only on masked nodes)
        if mask_idx.numel() > 0:
            loss_feat = sce_loss(recon[mask_idx], x_orig[mask_idx],
                                   self.cfg.alpha_l)
        else:
            loss_feat = recon.mean() * 0.0

        # structural reconstruction (positive vs random-negative edges)
        n = x_orig.size(0)
        n_pos = min(edge_index.size(1), 8000)
        if n_pos > 0:
            pos_idx = torch.randperm(edge_index.size(1),
                                       device=x_orig.device)[:n_pos]
            pos_src = edge_index[0, pos_idx]
            pos_dst = edge_index[1, pos_idx]
            neg_src = torch.randint(0, n, (n_pos,), device=x_orig.device)
            neg_dst = torch.randint(0, n, (n_pos,), device=x_orig.device)
            src = torch.cat([pos_src, neg_src])
            dst = torch.cat([pos_dst, neg_dst])
            pair = torch.cat([h_concat[src], h_concat[dst]], dim=-1)
            logits = self.edge_recon(pair).squeeze(-1)
            target = torch.cat([
                torch.ones(n_pos, device=x_orig.device),
                torch.zeros(n_pos, device=x_orig.device),
            ])
            loss_edge = F.binary_cross_entropy_with_logits(logits, target)
        else:
            loss_edge = recon.mean() * 0.0

        return loss_feat + loss_edge

    # ----- inference embedding -----
    @torch.no_grad()
    def embed(self, data: Data) -> torch.Tensor:
        """Return per-node embedding (no masking)."""
        self.eval()
        x = data.x.float()
        h = self._encode(x, data.edge_index)
        return h                                          # [N, H * n_layers]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_model(model: GMAE, path) -> None:
    torch.save({"state_dict": model.state_dict(),
                "config": model.cfg.__dict__}, path)


def load_model(path) -> GMAE:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MagicConfig(**blob["config"])
    m = GMAE(cfg)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m
