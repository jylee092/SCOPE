"""
Native-task metrics for DeepAG and MAGIC -- what each model is *natively*
designed to do, on its own terms (i.e., without Sigma in the loop).

DeepAG: next-token top-K accuracy on a held-out 20% slice of the 53 CTI
campaigns it was trained on. Mirrors the prediction setup of Li et al. §5.3.

MAGIC: entity-level anomaly AUC on the 35 Mordor scenarios. Positive class
= process nodes whose events overlap with any TP behavior group in the
SCOPE annotation; negative class = all other process nodes. Mirrors the
entity-level evaluation of Jia et al. §5.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import config
from pipeline.data_loader import load_and_normalize
from experiments.baselines.shield.graph import build_provenance_graph
from experiments.baselines.magic.graph_io import nx_to_pyg
from experiments.baselines.magic.model import load_model as load_magic_model
from experiments.baselines.ttp_sequence.model import (
    load_model as load_deepag_model, TIDVocab,
)
from experiments.baselines.ttp_sequence.train import (
    load_tid_tactic_map, all_campaign_sequences,
)


# ---------------------------------------------------------------------------
# DeepAG: next-token top-K on held-out CTI campaigns
# ---------------------------------------------------------------------------

def deepag_native(holdout_frac: float = 0.2, seed: int = 0,
                  window: int = 10) -> dict:
    rng = random.Random(seed)
    print("[deepag native] loading model + vocab ...")
    model_dir = config.OUTPUT_BASE_DIR / "baselines" / "ttp_sequence" / "_model"
    model = load_deepag_model(model_dir / "model.pt")
    vocab = TIDVocab.load(model_dir / "vocab.txt")

    print("[deepag native] re-deriving CTI sequences ...")
    tid2tac = load_tid_tactic_map(config.MITRE_CSV_PATH)
    seqs = all_campaign_sequences(tid2tac, shuffle=False)
    rng.shuffle(seqs)
    n_test = max(1, int(len(seqs) * holdout_frac))
    test_seqs = seqs[:n_test]
    print(f"[deepag native] held-out test set: {n_test}/{len(seqs)} campaigns")

    # encode and slide
    pad, sos, eos = vocab.PAD, vocab.SOS, vocab.EOS
    correct_top1, correct_top5, total = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for s in test_seqs:
            ids = [sos] + [vocab.encode(t) for t in s] + [eos]
            for pos in range(1, len(ids)):
                ctx = ids[max(0, pos - window):pos]
                ctx = [pad] * (window - len(ctx)) + ctx
                target = ids[pos]
                if target in (pad, sos):
                    continue
                x = torch.tensor([ctx], dtype=torch.long)
                logits = model.forward_logits(x,
                                                key_padding_mask=(x == pad))[0, -1]
                top5 = torch.topk(logits, k=5).indices.tolist()
                total += 1
                if top5[0] == target:
                    correct_top1 += 1
                if target in top5:
                    correct_top5 += 1

    return {
        "n_test_campaigns": n_test,
        "n_predictions":    total,
        "top1_acc":         correct_top1 / max(total, 1),
        "top5_acc":         correct_top5 / max(total, 1),
    }


# ---------------------------------------------------------------------------
# MAGIC: per-process anomaly AUC over 35 Mordor scenarios
# ---------------------------------------------------------------------------

def _attack_process_guids(scope_dir: Path, scenario_stem: str) -> set[str]:
    """Return ProcessGuids whose events fall in any TP behavior group of the
    SCOPE annotation for this scenario. Used as positive class in AUC."""
    feat_path = scope_dir / f"{scenario_stem}_feature_result.json"
    ann_path  = scope_dir / f"{scenario_stem}_annotation.json"
    if not feat_path.exists() or not ann_path.exists():
        return set()
    with open(feat_path, encoding="utf-8") as f:
        feat = json.load(f)
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)
    tp_idxs: set[int] = set()
    idxs_by_group = {g["group_id"]: set(g.get("all_idxs", []) or [])
                     for g in feat}
    for g in ann.get("groups", []):
        if g.get("gt_is_true_positive"):
            tp_idxs |= idxs_by_group.get(g.get("group_id"), set())
    return tp_idxs


def _roc_auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUC from two lists of scores."""
    if not pos or not neg:
        return float("nan")
    p = np.array(pos)
    n = np.array(neg)
    wins = 0.0
    ties = 0.0
    for s in p:
        wins += float((n < s).sum())
        ties += float((n == s).sum())
    return (wins + 0.5 * ties) / (len(p) * len(n))


def magic_native() -> dict:
    print("[magic native] loading model + benign reference ...")
    model_dir = config.OUTPUT_BASE_DIR / "baselines" / "magic" / "_model"
    model = load_magic_model(model_dir / "model.pt")
    benign_emb = torch.load(model_dir / "benign_emb.pt", weights_only=False)

    aucs: list[float] = []
    n_scenarios_eval = 0
    n_skipped = 0
    for scen in sorted(config.DATASET_FOLDER.rglob("*.json")):
        scenario_stem = scen.stem
        rel = scen.relative_to(config.DATASET_FOLDER).with_suffix("")
        scope_dir = config.OUTPUT_BASE_DIR / rel
        try:
            df = load_and_normalize(str(scen))
            df = df.reset_index().rename(columns={"index": "_orig_idx"})
        except Exception:
            n_skipped += 1; continue

        tp_event_idxs = _attack_process_guids(scope_dir, scenario_stem)
        if not tp_event_idxs:
            n_skipped += 1; continue

        # positive process GUIDs = those with at least one event in TP group
        if "ProcessGuid" not in df.columns:
            n_skipped += 1; continue
        pos_guids = set(
            df.loc[df["_orig_idx"].isin(tp_event_idxs),
                   "ProcessGuid"].dropna().astype(str)
        )
        pos_guids.discard(""); pos_guids.discard("nan")
        if not pos_guids:
            n_skipped += 1; continue

        g = build_provenance_graph(df)
        if g.number_of_nodes() == 0:
            n_skipped += 1; continue
        data, node_ids = nx_to_pyg(g)
        emb = model.embed(data)

        # KNN distance
        ref = benign_emb
        chunk = 256
        scores: list[float] = [0.0] * emb.size(0)
        for i in range(0, emb.size(0), chunk):
            block = emb[i:i + chunk]
            d = torch.cdist(block.float(), ref.float())
            kn, _ = torch.topk(d, k=min(10, ref.size(0)),
                                  dim=1, largest=False)
            kn = kn.mean(dim=1).cpu().numpy()
            for j, v in enumerate(kn):
                scores[i + j] = float(v)

        pos_scores: list[float] = []
        neg_scores: list[float] = []
        for i, n in enumerate(node_ids):
            if not n.startswith("P:"):
                continue
            guid = n[2:]
            (pos_scores if guid in pos_guids else neg_scores).append(scores[i])
        auc = _roc_auc(pos_scores, neg_scores)
        if auc != auc:                                 # NaN
            n_skipped += 1; continue
        aucs.append(auc)
        n_scenarios_eval += 1
        print(f"  {scenario_stem[:55]:<58}  AUC={auc:.3f}  "
              f"(pos={len(pos_scores)}, neg={len(neg_scores)})")

    return {
        "n_scenarios_evaluated": n_scenarios_eval,
        "n_scenarios_skipped":   n_skipped,
        "macro_AUC":             float(mean(aucs)) if aucs else 0.0,
    }


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("DeepAG native -- next-TID top-K accuracy on held-out CTI campaigns")
    print("=" * 70)
    deep = deepag_native()
    print(json.dumps(deep, indent=2))

    print()
    print("=" * 70)
    print("MAGIC native -- entity-level anomaly AUC on 35 Mordor scenarios")
    print("=" * 70)
    mg = magic_native()
    print(json.dumps(mg, indent=2))

    out = config.OUTPUT_BASE_DIR / "baselines" / "_native_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"deepag": deep, "magic": mg}, f, indent=2)
    print(f"\nSaved: {out}")
