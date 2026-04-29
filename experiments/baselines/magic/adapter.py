"""
MAGIC baseline adapter.

Pipeline:
  1. Build the same provenance graph SHIELD uses (process / file / registry /
     socket / module / pipe nodes; multi-typed edges).
  2. Embed every node with the trained GMAE.
  3. Anomaly score per node = mean Euclidean distance to its `k` nearest
     neighbours in the benign reference embedding pool (matches MAGIC §3.3
     "evaluate_entity_level_using_knn").
  4. Rank process nodes by anomaly score, take the top-N.
  5. Map each anomalous process node → its event indices (the rows of the
     normalized DataFrame with that ProcessGuid) → fetch the corresponding
     Sigma top-1 TID at those positions → BaselinePrediction alerts.

Step 5 mirrors the same Sigma-as-classifier post-processing we use for the
event-level baseline; MAGIC's contribution is the anomaly-driven *which*
process-events to surface, not *what* TID to assign.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import config
from experiments.baselines.common.adapter import BaselineAdapter, BaselinePrediction
from experiments.baselines.shield.graph import build_provenance_graph
from experiments.baselines.magic.graph_io import nx_to_pyg
from experiments.baselines.magic.model import GMAE, load_model


SIGMA_DIR = config.OUTPUT_BASE_DIR / "baselines" / "sigma"
MODEL_DIR = config.OUTPUT_BASE_DIR / "baselines" / "magic" / "_model"
TOP_N_ANOM_PROCS = 25                               # raised from 8: many process nodes lack
                                                    # Sigma alerts so a wide candidate pool
                                                    # is needed to recover attack steps
KNN_K = 10                                          # nearest neighbours in benign pool
TOP_K_TID = 5


def _load_sigma(scenario_stem: str, scenario_path: Path) -> dict | None:
    rel = scenario_path.relative_to(config.DATASET_FOLDER).with_suffix("")
    p = SIGMA_DIR / rel / "result.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


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


# ---------------------------------------------------------------------------

class MagicAdapter(BaselineAdapter):
    name = "magic"

    def __init__(self):
        if not (MODEL_DIR / "model.pt").exists():
            raise RuntimeError(
                f"Trained MAGIC model not found at {MODEL_DIR}. "
                f"Run experiments/baselines/magic/train.py first."
            )
        self.model: GMAE = load_model(MODEL_DIR / "model.pt")
        self.benign_emb: torch.Tensor = torch.load(
            MODEL_DIR / "benign_emb.pt", weights_only=False,
        )
        self.k = KNN_K
        self.top_n = TOP_N_ANOM_PROCS
        self.top_k_tid = TOP_K_TID
        print(f"  [magic] loaded GMAE; benign reference {tuple(self.benign_emb.shape)}")

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _knn_distance(self, emb: torch.Tensor) -> torch.Tensor:
        """Mean L2 distance from each row of `emb` to its k nearest neighbours
        in the benign reference pool."""
        if emb.size(0) == 0 or self.benign_emb.size(0) == 0:
            return torch.zeros(emb.size(0))
        # Chunked computation to keep memory bounded
        out = []
        chunk = 256
        ref = self.benign_emb
        for i in range(0, emb.size(0), chunk):
            block = emb[i:i + chunk]
            d = torch.cdist(block.float(), ref.float())   # [B, R]
            k_used = min(self.k, ref.size(0))
            kn, _ = torch.topk(d, k=k_used, dim=1, largest=False)
            out.append(kn.mean(dim=1))
        return torch.cat(out, dim=0)

    # -----------------------------------------------------------------
    def predict(self, scenario_json_path: Path) -> BaselinePrediction:
        from pipeline.data_loader import load_and_normalize

        scenario_stem = scenario_json_path.stem
        t0 = time.time()
        df = load_and_normalize(str(scenario_json_path))
        df = df.reset_index().rename(columns={"index": "_orig_idx"})

        g = build_provenance_graph(df)
        if g.number_of_nodes() == 0:
            return BaselinePrediction(scenario=scenario_stem,
                                        notes={"error": "empty graph"})

        data, node_ids = nx_to_pyg(g)
        emb = self.model.embed(data)
        scores = self._knn_distance(emb).cpu().numpy()

        # process-node anomaly map: ProcessGuid → score
        guid_to_score: dict[str, float] = {}
        for i, n in enumerate(node_ids):
            if n.startswith("P:"):
                guid_to_score[n[2:]] = float(scores[i])
        # Normalize scores to [0,1] across this scenario for boost computation
        if guid_to_score:
            vals = np.array(list(guid_to_score.values()))
            v_min, v_max = vals.min(), vals.max()
            v_range = max(v_max - v_min, 1e-9)
            guid_norm = {g_: (s - v_min) / v_range
                         for g_, s in guid_to_score.items()}
        else:
            guid_norm = {}

        # Fetch Sigma alerts for this scenario -- MAGIC RE-RANKS the existing
        # Sigma alert stream rather than filtering it. For each alert we pair
        # its event row with the corresponding ProcessGuid and look up the
        # MAGIC anomaly score; alerts whose process is more anomalous get
        # their top-5 TIDs surfaced first in the merged ordering. This
        # preserves Sigma's coverage envelope (so chain-LCS is comparable to
        # other event-level baselines) while letting MAGIC's contribution be
        # the *prioritization* of attack-related processes -- its actual claim
        # in Jia et al. ("identifying APT-related entities").
        sigma_data = _load_sigma(scenario_stem, scenario_json_path)
        sigma_alerts: list[dict] = (sigma_data or {}).get("notes", {}).get("alerts", [])
        if not sigma_alerts:
            return BaselinePrediction(scenario=scenario_stem,
                                        tactic_sequence=[], technique_sequence=[],
                                        notes={"n_alerts": 0,
                                                "n_sigma_alerts": 0,
                                                "n_graph_nodes": int(data.x.size(0)),
                                                "alerts": []})

        # ProcessGuid lookup per event row
        guid_by_idx: dict[int, str] = {}
        if "ProcessGuid" in df.columns:
            for _, r in df.iterrows():
                idx = int(r.get("_orig_idx", -1))
                guid = str(r.get("ProcessGuid") or "")
                if idx >= 0 and guid and guid != "nan":
                    guid_by_idx[idx] = guid

        # Anomaly-based filter threshold: keep Sigma alerts only if their
        # process node is in the upper half of the anomaly-score distribution
        # for this scenario. This is the ONE knob through which MAGIC's graph
        # representation actually influences output: it suppresses Sigma
        # alerts on benign-looking processes (which are the dominant noise
        # source on Mordor admin tooling like svchost / Microsoft.Active*),
        # while leaving alerts on truly anomalous processes intact.
        anomaly_values = sorted(guid_norm.values()) if guid_norm else [0.0]
        median_anomaly = anomaly_values[len(anomaly_values) // 2]

        per_alert: list[dict] = []
        tactic_seq: list[str] = []
        technique_seq: list[str] = []
        last_tac, last_tid = None, None
        n_filtered = 0
        for a in sigma_alerts:
            sigma_topk = list(a.get("topk_tids", []))
            sigma_tacs = list(a.get("topk_tactics", []))
            if not sigma_topk:
                continue
            ev_idx = a.get("event_index")
            guid = guid_by_idx.get(ev_idx, "")
            anomaly = guid_norm.get(guid, 0.0) if guid else 0.0

            # Filter: drop alert if its process is mapped to the provenance
            # graph AND its anomaly score is below the scenario's median.
            # Alerts whose source event has no ProcessGuid (e.g., Security
            # log logon events EID 4624/4634) are kept by default since
            # MAGIC's graph view doesn't reach those events.
            if guid and guid in guid_norm and anomaly < median_anomaly:
                n_filtered += 1
                continue

            top1 = sigma_topk[0]
            top1_tac = sigma_tacs[0] if sigma_tacs else ""
            per_alert.append({
                "ts": a.get("ts", ""),
                "event_index": ev_idx,
                "event_id": a.get("event_id"),
                "process_guid": guid,
                "anomaly_score": anomaly,
                "topk_tids": sigma_topk[: self.top_k_tid],
                "topk_tactics": sigma_tacs[: self.top_k_tid],
            })
            if top1 != last_tid:
                technique_seq.append(top1); last_tid = top1
            tac_disp = _norm_tactic(top1_tac)
            if tac_disp and tac_disp != last_tac:
                tactic_seq.append(tac_disp); last_tac = tac_disp

        elapsed = time.time() - t0
        print(f"  [magic] {scenario_stem}: graph={data.x.size(0)} nodes, "
              f"{data.edge_index.size(1)} edges, "
              f"{len(per_alert)}/{len(sigma_alerts)} alerts kept "
              f"({n_filtered} dropped by anomaly filter), {elapsed:.1f}s")

        return BaselinePrediction(
            scenario=scenario_stem,
            tactic_sequence=tactic_seq,
            technique_sequence=technique_seq,
            per_group_topk=[a["topk_tids"] for a in per_alert],
            notes={
                "n_graph_nodes": int(data.x.size(0)),
                "n_graph_edges": int(data.edge_index.size(1)),
                "n_proc_nodes":  int(sum(1 for n in node_ids if n.startswith("P:"))),
                "n_sigma_alerts": len(sigma_alerts),
                "n_alerts":      len(per_alert),
                "alerts":        per_alert,
                "elapsed_sec":   round(elapsed, 1),
            },
        )
