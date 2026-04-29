"""
Quality-gated novelty (Table 4 supplement).

Reviewer concern: chain-novel = (n(S*) >= 0.30) is a *distance* metric
that does not require the chain to be correct. A garbage chain with
zero overlap with the reference can still be "novel". To address this,
we additionally report

    coherent-novel = chain_novel AND technique_lcs_norm >= 0.50

per method. Per-scenario technique_lcs_norm is read from the existing
per-method scoring outputs; per-scenario n(S*) is recomputed here from
the same predicted tactic sequences using the 53-campaign library.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from pipeline.attack_chain import (
    compute_novelty, load_campaign_library, load_tactic_map,
)


TAU2     = 0.30
LCS_GATE = 0.50


# -- helpers -----------------------------------------------------------------

def _load_eval_v2():
    """Load SCOPE per-scenario chain.technique_lcs_norm and predicted tactics."""
    d = json.load(open(ROOT / "output" / "eval_v2_results.json"))
    out = {}
    for r in d:
        scen = r["scenario"]
        ch = r.get("chain", {})
        out[scen] = {
            "tech_lcs": ch.get("technique_lcs_norm", 0.0),
            "pred_tactics": [m.get("ref_tactic")
                             for m in ch.get("matched_pairs", [])
                             if m.get("pred_idx") is not None],
        }
    return out


def _scope_chain_per_scenario():
    """Iterate Dataset/ via config.configure_dataset and read the
    canonical SCOPE viterbi.json (output/<rel>/<scenario>_viterbi.json).
    Returns {scenario_name: [tactics...]}."""
    out = {}
    for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
        config.configure_dataset(ds)
        vit = config.VITERBI_JSON_PATH
        if not vit.exists():
            continue
        breakdown = json.load(open(vit, encoding="utf-8"))
        if not isinstance(breakdown, list) or not breakdown:
            continue
        out[config.DATASET_NAME] = [s.get("tactic")
                                    for s in breakdown if s.get("tactic")]
    return out


def _load_baseline_scores(method: str):
    """Per-scenario tech_lcs_norm for a baseline."""
    p = ROOT / "output" / "baselines" / method / "_scores.json"
    if not p.exists():
        return {}
    d = json.load(open(p, encoding="utf-8"))
    return {r["scenario"]: r.get("technique_lcs_norm", 0.0)
            for r in d.get("rows", [])}


def _load_baseline_chain(method: str):
    """Per-scenario predicted tactic sequence for a baseline.
    For Sigma/MAGIC/DeepAG/SHIELD we read their result.json files."""
    base = ROOT / "output" / "baselines" / method
    out = {}
    for p in base.rglob("result.json"):
        d = json.load(open(p, encoding="utf-8"))
        scen = d.get("scenario") or d.get("scenario_name")
        if not scen:
            continue
        seq = d.get("tactic_sequence") or []
        if not seq:
            # Some baselines store per-alert kill_chain_stages
            alerts = d.get("notes", {}).get("alerts", [])
            seq = []
            seen = None
            for a in alerts:
                tac = (a.get("topk_tactics") or [None])[0]
                if tac and tac != seen:
                    seq.append(tac)
                    seen = tac
        out[scen] = seq
    return out


# -- main --------------------------------------------------------------------

def main():
    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    campaigns  = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)
    print(f"loaded {len(campaigns)} campaigns")

    methods = {
        "SCOPE":  None,   # filled below
        "Sigma":  ("sigma",  _load_baseline_scores("sigma"),  _load_baseline_chain("sigma")),
        "MAGIC":  ("magic",  _load_baseline_scores("magic"),  _load_baseline_chain("magic")),
        "SHIELD": ("shield", _load_baseline_scores("shield"), _load_baseline_chain("shield")),
        "DeepAG": ("deepag", _load_baseline_scores("deepag"), _load_baseline_chain("deepag")),
    }

    # SCOPE: tech_lcs from eval_v2; chain from viterbi.json per scenario
    scope_lcs = _load_eval_v2()
    methods["SCOPE"] = ("scope", {s: v["tech_lcs"] for s, v in scope_lcs.items()}, None)

    print(f"\n{'method':<10} {'scenarios':>10} "
          f"{'chain-novel':>13} {'tech_lcs>=0.5':>15} {'coherent-novel':>15}")
    print('-' * 70)

    scope_chains = _scope_chain_per_scenario()

    for name, info in methods.items():
        if name == "SCOPE":
            tech_lcs_by_scen = info[1]
            scenarios = list(tech_lcs_by_scen.keys())
            chain_novel = []
            coherent_novel = []
            for scen in scenarios:
                pred_tac = scope_chains.get(scen, [])
                if not pred_tac:
                    n_val = 0.0
                else:
                    n_val, _ = compute_novelty(pred_tac, campaigns)
                cn = n_val >= TAU2
                ql = tech_lcs_by_scen[scen] >= LCS_GATE
                chain_novel.append(cn)
                coherent_novel.append(cn and ql)
        else:
            _, lcs_map, chain_map = info
            scenarios = list(lcs_map.keys())
            chain_novel = []
            coherent_novel = []
            for scen in scenarios:
                pred_tac = chain_map.get(scen, []) or []
                if not pred_tac:
                    n_val = 0.0
                else:
                    n_val, _ = compute_novelty(pred_tac, campaigns)
                cn = n_val >= TAU2
                ql = lcs_map[scen] >= LCS_GATE
                chain_novel.append(cn)
                coherent_novel.append(cn and ql)

        n_total  = len(scenarios)
        n_cn     = sum(chain_novel)
        n_ql     = sum(1 for s in scenarios if (info[1].get(s, 0) >= LCS_GATE))
        n_coh    = sum(coherent_novel)
        print(f"{name:<10} {n_total:>10} "
              f"{n_cn:>5}/{n_total:<5} ({n_cn/n_total:.2f})  "
              f"{n_ql:>5}/{n_total:<5} ({n_ql/n_total:.2f})  "
              f"{n_coh:>5}/{n_total:<5} ({n_coh/n_total:.2f})")


if __name__ == "__main__":
    main()
