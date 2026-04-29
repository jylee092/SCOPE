"""
DeepAG (TID-adapted) -- Transformer encoder + bi-directional LSTM heads.

Faithful to the architecture of Li et al. "DeepAG: Attack Graph Construction
and Threats Prediction with Bi-Directional Deep Learning" (TDSC 2023).
We swap the Log2Vec-style log-template embedding for a learnable MITRE
ATT&CK technique-id embedding so the model operates over TID sequences
(see Section 7.1 of the SCOPE paper for the rationale).

Architecture (defaults match Li et al. §5.1.2):
    h = 10      # sliding-window length
    L = 2       # Transformer encoder layers
    a = 64      # LSTM hidden units
    d = 64      # embedding / model dim
    n_head = 4

Two prediction heads:
    forward_logits[t]  = P(x_{t+1} | x_1..x_t)
    backward_logits[t] = P(y_{t}   | y_{t+1}..y_h)  (after reversal)

Loss: cross-entropy on both heads, summed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DeepAGConfig:
    vocab_size: int
    d_model:    int = 64
    n_head:     int = 4
    n_layers:   int = 2
    lstm_hidden: int = 64
    window:     int = 10
    dropout:    float = 0.1
    pad_idx:    int = 0


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DeepAG(nn.Module):
    """Transformer encoder + forward + backward LSTM."""

    def __init__(self, cfg: DeepAGConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model,
                                   padding_idx=cfg.pad_idx)
        self.pos_enc = _PositionalEncoding(cfg.d_model, max_len=64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_head,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout, batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                   num_layers=cfg.n_layers)
        self.fwd_lstm = nn.LSTM(cfg.d_model, cfg.lstm_hidden,
                                  batch_first=True)
        self.bwd_lstm = nn.LSTM(cfg.d_model, cfg.lstm_hidden,
                                  batch_first=True)
        self.fwd_head = nn.Linear(cfg.lstm_hidden, cfg.vocab_size)
        self.bwd_head = nn.Linear(cfg.lstm_hidden, cfg.vocab_size)

    def encode(self, x: torch.Tensor,
               key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        e = self.embed(x)
        e = self.pos_enc(e)
        h = self.transformer(e, src_key_padding_mask=key_padding_mask)
        return h

    def forward_logits(self, x: torch.Tensor,
                        key_padding_mask: torch.Tensor | None = None
                        ) -> torch.Tensor:
        """Returns logits[batch, seq, vocab] predicting next-TID at each pos."""
        h = self.encode(x, key_padding_mask)
        out, _ = self.fwd_lstm(h)
        return self.fwd_head(out)

    def backward_logits(self, x_rev: torch.Tensor,
                         key_padding_mask: torch.Tensor | None = None
                         ) -> torch.Tensor:
        """`x_rev` is the reversed input sequence."""
        h = self.encode(x_rev, key_padding_mask)
        out, _ = self.bwd_lstm(h)
        return self.bwd_head(out)


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------

class TIDVocab:
    """Bidirectional mapping between TID strings and integer indices.
    Reserved: 0=<PAD>, 1=<UNK>, 2=<SOS>, 3=<EOS>"""
    PAD, UNK, SOS, EOS = 0, 1, 2, 3
    _RESERVED = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

    def __init__(self):
        self.tid2id: dict[str, int] = {t: i for i, t in enumerate(self._RESERVED)}
        self.id2tid: list[str] = list(self._RESERVED)

    def add(self, tid: str) -> int:
        if tid not in self.tid2id:
            self.tid2id[tid] = len(self.id2tid)
            self.id2tid.append(tid)
        return self.tid2id[tid]

    def encode(self, tid: str) -> int:
        return self.tid2id.get(tid, self.UNK)

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.id2tid):
            return self.id2tid[idx]
        return "<UNK>"

    def __len__(self) -> int:
        return len(self.id2tid)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for t in self.id2tid:
                f.write(t + "\n")

    @classmethod
    def load(cls, path: Path) -> "TIDVocab":
        v = cls()
        v.tid2id = {}
        v.id2tid = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                v.tid2id[t] = len(v.id2tid)
                v.id2tid.append(t)
        return v


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_model(model: DeepAG, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": model.cfg.__dict__}, path)


def load_model(path: Path) -> DeepAG:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = DeepAGConfig(**blob["config"])
    m = DeepAG(cfg)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m
