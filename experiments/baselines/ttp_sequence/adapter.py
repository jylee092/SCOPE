"""
DeepAG (TID-adapted) baseline adapter.

Pipeline at inference time:
  1. Read the corresponding Sigma result.json — Sigma's per-alert top-1
     TID gives us the chronological log-vector-equivalent input stream.
  2. For each alert position t, slide a window of (window-1) past TIDs
     into the trained Transformer + bi-LSTM and read off the forward-
     LSTM Top-K next-TID logits.
  3. Combine per-position:
        merged_topk = priority union of (DeepAG predictions, Sigma top-5)
     Sigma's coverage stays intact; DeepAG injects sequence-aware
     refinements that outrank low-probability Sigma ranks.
  4. The chain output picks the DeepAG top-1 when its softmax probability
     exceeds a threshold (η=0.1 per Li et al. §5.1.2); otherwise it falls
     back to the Sigma top-1.

The adapter therefore exercises exactly DeepAG's sequence-prediction
capability (Transformer + bi-LSTM next-token prediction over its trained
vocabulary) while remaining comparable to the other TID-output baselines.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

import config
from experiments.baselines.common.adapter import BaselineAdapter, BaselinePrediction
from experiments.baselines.ttp_sequence.model import (
    DeepAG, TIDVocab, load_model,
)


SIGMA_DIR = config.OUTPUT_BASE_DIR / "baselines" / "sigma"
MODEL_DIR = config.OUTPUT_BASE_DIR / "baselines" / "ttp_sequence" / "_model"
THRESHOLD = 0.5            # raised from 0.1 (DeepAG §5.1.2 used vocab≈50; ours is 375)
TOP_K = 5
WINDOW = 10                 # h in DeepAG §5.1.2

# Merge strategy:
#   "deepag-priority"  → DeepAG top-K first, Sigma fills remainder, chain top-1
#                          replaced by DeepAG when prob ≥ THRESHOLD (Li et al. §5.1.2
#                          spirit; the standalone DeepAG behaviour).
#   "sigma-priority"   → Sigma top-K first, DeepAG augments empty slots; chain top-1
#                          stays Sigma — measures DeepAG's *additive* refinement.
MERGE_STRATEGY = "deepag-priority"


_TACTIC_DISPLAY = {
    "reconnaissance": "Reconnaissance",
    "resource-development": "Resource Development",
    "initial-access": "Initial Access",
    "execution": "Execution",
    "persistence": "Persistence",
    "privilege-escalation": "Privilege Escalation",
    "defense-evasion": "Defense Evasion",
    "credential-access": "Credential Access",
    "discovery": "Discovery",
    "lateral-movement": "Lateral Movement",
    "collection": "Collection",
    "command-and-control": "Command and Control",
    "exfiltration": "Exfiltration",
    "impact": "Impact",
}


def _norm_tactic(s: str) -> str:
    return _TACTIC_DISPLAY.get((s or "").strip().lower(), (s or "").strip())


def _load_tid_tactic_map() -> dict[str, str]:
    """Reuse the train-time TID→tactic map for tactic display in output."""
    from experiments.baselines.ttp_sequence.train import load_tid_tactic_map
    return load_tid_tactic_map(config.MITRE_CSV_PATH)


def _find_sigma_result(scenario_json_path: Path) -> Path | None:
    rel = scenario_json_path.relative_to(config.DATASET_FOLDER).with_suffix("")
    p = SIGMA_DIR / rel / "result.json"
    return p if p.exists() else None


# ---------------------------------------------------------------------------

class DeepAGAdapter(BaselineAdapter):
    name = "ttp_sequence"

    def __init__(self, merge_strategy: str = MERGE_STRATEGY,
                  threshold: float = THRESHOLD):
        if not (MODEL_DIR / "model.pt").exists():
            raise RuntimeError(
                f"Trained DeepAG model not found at {MODEL_DIR}. "
                f"Run experiments/baselines/ttp_sequence/train.py first."
            )
        self.model: DeepAG = load_model(MODEL_DIR / "model.pt")
        self.vocab: TIDVocab = TIDVocab.load(MODEL_DIR / "vocab.txt")
        self.tid2tac = _load_tid_tactic_map()
        self.threshold = threshold
        self.merge_strategy = merge_strategy
        self.window = WINDOW
        self.k = TOP_K
        print(f"  [deepag] loaded model (vocab={len(self.vocab)}) from {MODEL_DIR}; "
              f"merge={merge_strategy}, threshold={threshold}")

    # ----- forward LSTM next-TID prediction -----
    @torch.no_grad()
    def _predict_next(self, context_tids: list[str]) -> list[tuple[str, float]]:
        """Return top-K (tid, prob) for the next TID given a chronological
        list of recent TIDs, using the forward LSTM head."""
        if not context_tids:
            ids = [self.vocab.SOS]
        else:
            ids = [self.vocab.SOS] + [self.vocab.encode(t) for t in context_tids]
        # truncate / pad to window
        ids = ids[-self.window:]
        pad = self.window - len(ids)
        if pad > 0:
            ids = [self.vocab.PAD] * pad + ids
        x = torch.tensor([ids], dtype=torch.long)
        mask = (x == self.vocab.PAD)
        logits = self.model.forward_logits(x, key_padding_mask=mask)[0, -1]
        probs = F.softmax(logits, dim=-1)
        top = torch.topk(probs, k=min(self.k * 4, probs.size(0)))
        out: list[tuple[str, float]] = []
        for p, i in zip(top.values.tolist(), top.indices.tolist()):
            tid = self.vocab.decode(i)
            if tid in ("<PAD>", "<UNK>", "<SOS>", "<EOS>"):
                continue
            out.append((tid, float(p)))
            if len(out) >= self.k:
                break
        return out

    # ------------------------------------------------------------
    def predict(self, scenario_json_path: Path) -> BaselinePrediction:
        sigma_path = _find_sigma_result(scenario_json_path)
        if not sigma_path:
            print(f"  [deepag] no sigma result for {scenario_json_path.stem} — empty")
            return BaselinePrediction(scenario=scenario_json_path.stem,
                                        notes={"error": "no sigma input"})
        with open(sigma_path, encoding="utf-8") as f:
            sigma = json.load(f)

        sigma_alerts: list[dict] = sigma.get("notes", {}).get("alerts", [])
        if not sigma_alerts:
            return BaselinePrediction(scenario=sigma["scenario"],
                                        notes={"n_alerts": 0,
                                                "n_sigma_alerts": 0,
                                                "alerts": []})

        # ---- per-position DeepAG refinement ----
        prev_top1: list[str] = []
        per_alert: list[dict] = []
        tactic_seq: list[str] = []
        technique_seq: list[str] = []
        last_tac, last_tid = None, None
        n_deepag_used = 0

        for a in sigma_alerts:
            sigma_topk: list[str] = list(a.get("topk_tids", []))
            if not sigma_topk:
                continue
            sigma_top1 = sigma_topk[0]

            # DeepAG forward prediction conditioned on prev_top1 history
            deepag_topk = self._predict_next(prev_top1)

            # Decide whether DeepAG is confident enough to take over.
            deepag_confident = (
                self.merge_strategy == "deepag-priority"
                and deepag_topk and deepag_topk[0][1] >= self.threshold
            )

            seen: set[str] = set()
            merged: list[tuple[str, float, str]] = []
            if deepag_confident:
                # DeepAG drives both top-K ordering and chain top-1
                primary = [(t, p, "deepag") for t, p in deepag_topk]
                secondary = [(t, 0.0, "sigma") for t in sigma_topk]
                top1, top1_src = deepag_topk[0][0], "deepag"
                n_deepag_used += 1
            else:
                # Sigma drives top-K and chain top-1; DeepAG fills empty slots
                primary = [(t, 0.0, "sigma") for t in sigma_topk]
                secondary = [(t, p, "deepag") for t, p in deepag_topk]
                top1, top1_src = sigma_top1, "sigma"

            for tid, prob, src in primary + secondary:
                if tid in seen:
                    continue
                seen.add(tid)
                merged.append((tid, prob, src))
                if len(merged) >= self.k:
                    break
            merged_tids = [m[0] for m in merged]

            # Track context for next position (use the chosen top-1)
            prev_top1.append(top1)
            if len(prev_top1) > self.window * 2:
                prev_top1 = prev_top1[-self.window * 2:]

            # Tactic via TID→tactic map
            tac = self.tid2tac.get(top1) or self.tid2tac.get(top1.split(".")[0]) or ""

            per_alert.append({
                "ts": a.get("ts", ""),
                "event_index": a.get("event_index"),
                "event_id":     a.get("event_id"),
                "topk_tids":    merged_tids,
                "topk_tactics": [
                    self.tid2tac.get(t) or self.tid2tac.get(t.split(".")[0]) or ""
                    for t in merged_tids
                ],
                "deepag_top1":  deepag_topk[0][0] if deepag_topk else None,
                "deepag_prob":  deepag_topk[0][1] if deepag_topk else 0.0,
                "chosen_top1":  top1,
                "chosen_src":   top1_src,
            })

            if top1 != last_tid:
                technique_seq.append(top1); last_tid = top1
            if tac and tac != last_tac:
                tactic_seq.append(tac); last_tac = tac

        return BaselinePrediction(
            scenario=sigma["scenario"],
            tactic_sequence=tactic_seq,
            technique_sequence=technique_seq,
            per_group_topk=[a["topk_tids"] for a in per_alert],
            notes={
                "n_alerts":          len(per_alert),
                "n_sigma_alerts":    len(sigma_alerts),
                "n_deepag_top1_used": n_deepag_used,
                "alerts":            per_alert,
            },
        )
