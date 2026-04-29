"""
Train DeepAG (TID-adapted) on the 53 MITRE CTI campaigns.

Each campaign JSON is a MITRE ATT&CK Navigator layer that lists the
techniques used by a campaign as an unordered set. We project each set to
an ordered sequence by:
  (1) looking up each technique's primary tactic from the merged MITRE CSV;
  (2) sorting by kill-chain position of that tactic;
  (3) inside the same tactic, applying random shuffles per epoch as a
      cheap data-augmentation knob (since CTI layers do not record true
      execution order).

Loss:
  forward  cross-entropy on next-TID prediction
  backward cross-entropy on next-TID prediction over the reversed sequence

Saves:
  output/baselines/ttp_sequence/_model/model.pt
  output/baselines/ttp_sequence/_model/vocab.txt
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

import config
from experiments.baselines.ttp_sequence.model import (
    DeepAG, DeepAGConfig, TIDVocab, save_model,
)


CAMPAIGN_DIR = config.CAMPAIGN_FOLDER
MITRE_CSV    = config.MITRE_CSV_PATH
MODEL_DIR    = config.OUTPUT_BASE_DIR / "baselines" / "ttp_sequence" / "_model"


_KILL_CHAIN_ORDER = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact",
]
_KC_RANK = {t: i for i, t in enumerate(_KILL_CHAIN_ORDER)}


# ---------------------------------------------------------------------------
# Load TID → primary tactic
# ---------------------------------------------------------------------------

def load_tid_tactic_map(csv_path: Path) -> dict[str, str]:
    """Return {tid: kill-chain-ordered primary tactic display name}.

    Multi-tactic entries (e.g., "Defense Evasion, Privilege Escalation") are
    resolved by picking the earliest tactic in kill-chain order — this gives
    each TID a deterministic position when serializing a campaign."""
    tid2tac: dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = (row.get("ID") or "").strip()
            tacs = (row.get("tactics") or "").strip()
            if not tid or not tacs:
                continue
            opts = [t.strip() for t in tacs.split(",") if t.strip()]
            if not opts:
                continue
            opts.sort(key=lambda t: _KC_RANK.get(t, 99))
            tid2tac[tid] = opts[0]
    return tid2tac


# ---------------------------------------------------------------------------
# Build ordered sequences from campaign JSONs
# ---------------------------------------------------------------------------

def load_campaign_sequence(camp_path: Path,
                            tid2tac: dict[str, str],
                            shuffle_within_tactic: bool = False) -> list[str]:
    """Return a kill-chain-ordered TID sequence for a single campaign."""
    with open(camp_path, encoding="utf-8") as f:
        data = json.load(f)
    techs = data.get("techniques") or []
    tids: list[str] = []
    for t in techs:
        tid = (t.get("techniqueID") or "").strip().upper()
        if tid:
            tids.append(tid)
    # Group by tactic, then sort
    grouped: dict[str, list[str]] = {}
    for tid in tids:
        tac = tid2tac.get(tid) or tid2tac.get(tid.split(".")[0]) or "Unknown"
        grouped.setdefault(tac, []).append(tid)
    # within-group order
    out: list[str] = []
    rng = random.Random()
    for tac in sorted(grouped.keys(), key=lambda t: _KC_RANK.get(t, 99)):
        members = grouped[tac]
        if shuffle_within_tactic:
            rng.shuffle(members)
        else:
            members.sort()
        out.extend(members)
    return out


def all_campaign_sequences(tid2tac: dict[str, str],
                            shuffle: bool = False) -> list[list[str]]:
    seqs: list[list[str]] = []
    for p in sorted(CAMPAIGN_DIR.glob("*.json")):
        seq = load_campaign_sequence(p, tid2tac, shuffle_within_tactic=shuffle)
        if seq:
            seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Build training examples
# ---------------------------------------------------------------------------

def build_vocab(seqs: list[list[str]]) -> TIDVocab:
    v = TIDVocab()
    for s in seqs:
        for t in s:
            v.add(t)
            # also add parent technique to vocab (helps generalization for
            # sub-techniques the test stream may emit at parent granularity)
            if "." in t:
                v.add(t.split(".")[0])
    return v


def make_windows(seq: list[int], window: int,
                 pad_idx: int, sos_idx: int, eos_idx: int):
    """Yield (input, target) pairs for next-token prediction.

    Each window is `window` tokens; target is the same window shifted by 1.
    Sequences shorter than `window` are pre-padded with <SOS>; the (window-1)
    boundary token after the sequence is <EOS>."""
    if not seq:
        return []
    full = [sos_idx] + seq + [eos_idx]
    pairs = []
    for start in range(0, len(full) - 1):
        end = min(start + window, len(full) - 1)
        window_in = full[start:end]
        window_out = full[start + 1:end + 1]
        # left-pad if shorter than window
        pad = window - len(window_in)
        if pad > 0:
            window_in = [pad_idx] * pad + window_in
            window_out = [pad_idx] * pad + window_out
        pairs.append((window_in, window_out))
    return pairs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(epochs: int = 60,
          batch_size: int = 32,
          lr: float = 3e-3,
          shuffle_per_epoch: bool = True,
          seed: int = 0) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    print(f"[deepag] loading TID→tactic map from {MITRE_CSV}")
    tid2tac = load_tid_tactic_map(MITRE_CSV)
    print(f"  {len(tid2tac)} TIDs with tactic")

    base_seqs = all_campaign_sequences(tid2tac, shuffle=False)
    print(f"[deepag] {len(base_seqs)} campaigns; "
          f"avg len={sum(len(s) for s in base_seqs) / max(len(base_seqs),1):.1f}, "
          f"max={max(len(s) for s in base_seqs)}")

    vocab = build_vocab(base_seqs)
    print(f"[deepag] vocab size: {len(vocab)}")

    cfg = DeepAGConfig(vocab_size=len(vocab))
    model = DeepAG(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pad_idx, sos, eos = vocab.PAD, vocab.SOS, vocab.EOS

    print(f"[deepag] training for {epochs} epochs ...")
    for epoch in range(1, epochs + 1):
        # re-derive shuffled sequences each epoch (within-tactic randomness)
        if shuffle_per_epoch:
            seqs = all_campaign_sequences(tid2tac, shuffle=True)
        else:
            seqs = base_seqs
        # encode
        encoded = [[vocab.encode(t) for t in s] for s in seqs]

        # build training pool: all sliding windows from forward + backward
        pool = []
        for s in encoded:
            pool.extend(make_windows(s, cfg.window, pad_idx, sos, eos))
        random.shuffle(pool)

        model.train()
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()
        for i in range(0, len(pool), batch_size):
            batch = pool[i:i + batch_size]
            x = torch.tensor([b[0] for b in batch], dtype=torch.long)
            y = torch.tensor([b[1] for b in batch], dtype=torch.long)
            x_rev = x.flip(dims=[1])
            y_rev = y.flip(dims=[1])
            mask = (x == pad_idx)

            optim.zero_grad()
            fwd = model.forward_logits(x, key_padding_mask=mask)
            bwd = model.backward_logits(x_rev,
                                          key_padding_mask=mask.flip(dims=[1]))
            loss_fwd = F.cross_entropy(
                fwd.reshape(-1, fwd.size(-1)),
                y.reshape(-1),
                ignore_index=pad_idx,
            )
            loss_bwd = F.cross_entropy(
                bwd.reshape(-1, bwd.size(-1)),
                y_rev.reshape(-1),
                ignore_index=pad_idx,
            )
            loss = loss_fwd + loss_bwd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            n_tok = (y != pad_idx).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

        avg = total_loss / max(total_tokens, 1)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  epoch {epoch:>3}: loss={avg:.4f}  "
                  f"({len(pool)} windows, {time.time()-t0:.1f}s)")

    # save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, MODEL_DIR / "model.pt")
    vocab.save(MODEL_DIR / "vocab.txt")
    print(f"[deepag] saved → {MODEL_DIR}")


if __name__ == "__main__":
    train()
