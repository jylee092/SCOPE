"""
Q3 novelty measurement.

Two metrics per the paper §5.3 / §5.4 / §7.4:

(a) TTP-level novel mapping (per behavior group):
    top-1 candidate's TID family differs from the rule prior
    AND confidence_margin ≥ μ.
    Reported as fraction across all groups in 35 scenarios.

(b) Chain-level novelty coverage (per scenario):
    n(S*) = 1 - max_C LCS(S*_tactics, C_tactics) / |S*_tactics|
    Reported as fraction of scenarios with n(S*) ≥ τ_2.

For SCOPE we compute (a) over the per-group ttp_mapping.json and (b) over
the per-scenario viterbi.json + the existing 53-campaign library.

For baselines (Sigma, MAGIC, SHIELD, DeepAG) we compute:
  - (a) approximation: fraction of alerts whose chosen top-1 lies outside
    the union of TIDs proposed by the per-scenario Sigma rule set
    (Sigma's natural "rule prior" envelope); for Sigma itself this is 0
    by construction.
  - (b): same chain-novelty measure as SCOPE, using each baseline's
    predicted tactic_sequence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from pipeline.attack_chain import (
    compute_novelty, load_campaign_library, load_tactic_map,
)


MARGIN_MU = 0.005                         # see config.VITERBI_MARGIN_HIGH=0.015 / p90 ≈ 0.015
NOVEL_TAU2 = 0.30                          # paper §7.1 threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rule_tid_from_group_id(gid: str) -> str:
    """SCOPE group ids: `<TID>[_<sub>]_<index>` (sub is 3-digit; trailing
    index is variable length). E.g., `T1003_001_7` → `T1003.001`,
    `T1003_53` → `T1003`."""
    if not gid:
        return ""
    parts = gid.split("_")
    if not parts or not parts[0].startswith("T"):
        return ""
    out = [parts[0]]
    for p in parts[1:]:
        # Sub-technique segments are always exactly 3 digits
        if p.isdigit() and len(p) == 3:
            out.append(p)
        else:
            break
    if len(out) > 1:
        return out[0] + "." + ".".join(out[1:])
    return out[0]


def _family_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    return a.split(".")[0] == b.split(".")[0]


def _load_json(p: Path):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# SCOPE -- measured directly from ttp_mapping + viterbi outputs
# ---------------------------------------------------------------------------

def scope_novelty(margin_mu: float = MARGIN_MU,
                   tau2: float = NOVEL_TAU2) -> dict:
    """
    TTP-novel uses the raw-similarity top-1 (re-ranking the candidate list
    by `similarity` rather than by the post-boost `p_ttp`). This isolates
    the bare-model "what would the embedding alone choose?" from SCOPE's
    rule-TID prior multiplicative boost (RULE_TID_PRIOR=1.15), which by
    design pushes the rule-anticipated TID to the top in p_ttp space -- and
    therefore makes the post-boost top-1 trivially equal to the rule prior
    in nearly every group. The raw-similarity top-1 captures the meaningful
    case: a behavior whose embedding *most-resembles* a different technique
    even before any rule guidance is applied.
    """
    n_total_groups = 0
    n_novel_groups = 0
    n_total_scenarios = 0
    n_chain_novel = 0
    chain_n_values = []

    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    campaigns = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)

    for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
        config.configure_dataset(ds)
        ttp_path = config.TTP_MAPPING_JSON_PATH
        vit_path = config.VITERBI_JSON_PATH
        if not (ttp_path.exists() and vit_path.exists()):
            continue
        n_total_scenarios += 1

        # ---- (a) TTP-level novel ----
        with open(ttp_path, encoding="utf-8") as f:
            mapping = json.load(f)
        for g in mapping:
            n_total_groups += 1
            sims = g.get("similar_techniques") or []
            if len(sims) < 2:
                continue
            sims_by_raw = sorted(sims, key=lambda x: -float(x.get("similarity") or 0.0))
            raw_top1 = (sims_by_raw[0].get("technique_id") or "")
            raw_top1_sim = float(sims_by_raw[0].get("similarity") or 0.0)
            raw_top2_sim = float(sims_by_raw[1].get("similarity") or 0.0)
            raw_margin = raw_top1_sim - raw_top2_sim
            rule_tid = _rule_tid_from_group_id(g.get("group_id") or "")
            if raw_margin < margin_mu:
                continue
            if not _family_match(raw_top1, rule_tid):
                n_novel_groups += 1

        # ---- (b) Chain-level novel ----
        with open(vit_path, encoding="utf-8") as f:
            breakdown = json.load(f)
        if not isinstance(breakdown, list) or not breakdown:
            continue
        pred_tactics = [s["tactic"] for s in breakdown if s.get("tactic")]
        if not pred_tactics:
            continue
        n_value, _ = compute_novelty(pred_tactics, campaigns)
        chain_n_values.append(n_value)
        if n_value >= tau2:
            n_chain_novel += 1

    return {
        "method": "SCOPE",
        "ttp_total_groups": n_total_groups,
        "ttp_novel_groups": n_novel_groups,
        "ttp_novel_frac": n_novel_groups / max(n_total_groups, 1),
        "chain_total_scenarios": n_total_scenarios,
        "chain_novel_scenarios": n_chain_novel,
        "chain_novel_frac": n_chain_novel / max(n_total_scenarios, 1),
        "chain_n_mean": mean(chain_n_values) if chain_n_values else 0.0,
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _sigma_rule_envelope(scenario_path: Path) -> set[str]:
    """Union of all TIDs that Sigma's per-event top-K proposed for this
    scenario. Used as the "rule prior" envelope for TTP-novel proxy."""
    sigma_dir = config.OUTPUT_BASE_DIR / "baselines" / "sigma"
    rel = scenario_path.relative_to(config.DATASET_FOLDER).with_suffix("")
    p = sigma_dir / rel / "result.json"
    data = _load_json(p)
    if not data:
        return set()
    tids: set[str] = set()
    for a in data.get("notes", {}).get("alerts", []):
        for t in a.get("topk_tids", []):
            tids.add(t)
    return tids


def baseline_novelty(name: str, results_dir_name: str,
                      tau2: float = NOVEL_TAU2,
                      common_denominator: int = 779) -> dict:
    """name: human-readable; results_dir_name: subdirectory of output/baselines/."""
    base = config.OUTPUT_BASE_DIR / "baselines" / results_dir_name
    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    campaigns = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)

    n_total_alerts = 0
    n_novel_alerts = 0
    n_total_scenarios = 0
    n_chain_novel = 0

    for vit_path in sorted(base.rglob("result.json")):
        rel = vit_path.relative_to(base).parent
        scen_data = _load_json(vit_path)
        if not scen_data:
            continue
        n_total_scenarios += 1

        # ---- (a) TTP-novel proxy: chosen top-1 outside Sigma rule envelope ----
        # Locate the matching scenario file to compute the Sigma envelope.
        scen_dir = config.DATASET_FOLDER / rel
        # The scenario JSON sits in the deepest subdir; find it.
        candidates = list((config.DATASET_FOLDER / rel).rglob("*.json"))
        sigma_env: set[str] = set()
        if candidates:
            sigma_env = _sigma_rule_envelope(candidates[0])
        elif scen_dir.with_suffix(".json").exists():
            sigma_env = _sigma_rule_envelope(scen_dir.with_suffix(".json"))

        for a in scen_data.get("notes", {}).get("alerts", []):
            top1 = (a.get("topk_tids") or [None])[0]
            if not top1:
                continue
            n_total_alerts += 1
            # Sigma itself: "novel" relative to itself = 0 by definition; the
            # union envelope is exactly the set of Sigma-proposed TIDs.
            if results_dir_name == "sigma":
                continue
            if top1 not in sigma_env and top1.split(".")[0] not in (
                t.split(".")[0] for t in sigma_env
            ):
                n_novel_alerts += 1

        # ---- (b) Chain-level novel ----
        pred_tactics = scen_data.get("tactic_sequence") or []
        if not pred_tactics:
            continue
        n_value, _ = compute_novelty(pred_tactics, campaigns)
        if n_value >= tau2:
            n_chain_novel += 1

    # Use a common denominator (SCOPE's total behavior-group count, 779)
    # so that the "TTP-novel rate" is comparable across methods regardless
    # of how many alerts each emits. Otherwise a method that emits very
    # few alerts can show a misleadingly high alert-relative novelty rate
    # (SHIELD: 9/17=53%) that masks its low detection coverage.
    novel_count = 0 if results_dir_name == "sigma" else n_novel_alerts
    return {
        "method": name,
        "ttp_total_alerts": n_total_alerts,
        "ttp_novel_alerts": novel_count,
        "ttp_novel_alert_rate": (novel_count / max(n_total_alerts, 1)
                                  if results_dir_name != "sigma" else 0.0),
        "ttp_novel_frac":     novel_count / max(common_denominator, 1),
        "common_denominator": common_denominator,
        "chain_total_scenarios": n_total_scenarios,
        "chain_novel_scenarios": n_chain_novel,
        "chain_novel_frac": n_chain_novel / max(n_total_scenarios, 1),
    }


# ---------------------------------------------------------------------------

def main(margin_mus: list[float] | None = None,
          tau2s: list[float] | None = None) -> None:
    # Raw-similarity margin is in cosine-distance space, so use a larger
    # threshold sweep than the p_ttp-margin we used previously.
    margin_mus = margin_mus or [0.02, 0.05]
    tau2s = tau2s or [0.30, 0.50]

    print("=" * 70)
    print(f"SCOPE -- sweep over (μ, τ_2)")
    print("=" * 70)
    for mu in margin_mus:
        for t2 in tau2s:
            s = scope_novelty(margin_mu=mu, tau2=t2)
            print(f"  μ={mu:.2f} τ_2={t2:.2f}  "
                  f"TTP-novel={s['ttp_novel_frac']:.4f} "
                  f"({s['ttp_novel_groups']}/{s['ttp_total_groups']})  "
                  f"chain-novel={s['chain_novel_frac']:.4f} "
                  f"({s['chain_novel_scenarios']}/{s['chain_total_scenarios']})  "
                  f"⟨n(S*)⟩={s['chain_n_mean']:.3f}")

    # use μ=0.02 + τ_2=0.30 as canonical for baseline comparison
    canonical_mu, canonical_tau2 = 0.02, 0.30
    print()
    print("=" * 70)
    print(f"Baselines (chain-novel τ_2={canonical_tau2})")
    print("=" * 70)
    rows = []
    for name, dirn in [
        ("Sigma",  "sigma"),
        ("MAGIC",  "magic"),
        ("SHIELD", "shield"),
        ("DeepAG", "deepag"),
    ]:
        r = baseline_novelty(name, dirn, tau2=canonical_tau2)
        rows.append(r)
        print(f"  {r['method']:<10} TTP-novel={r['ttp_novel_frac']:.4f} "
              f"({r['ttp_novel_alerts']}/{r['common_denominator']} groups) "
              f"[alert-rate={r['ttp_novel_alert_rate']:.4f} "
              f"of {r['ttp_total_alerts']}] "
              f"chain-novel={r['chain_novel_frac']:.4f} "
              f"({r['chain_novel_scenarios']}/{r['chain_total_scenarios']})")

    out = config.OUTPUT_BASE_DIR / "_novelty_scores.json"
    canonical_scope = scope_novelty(margin_mu=canonical_mu, tau2=canonical_tau2)
    canonical_scope["mu"] = canonical_mu
    canonical_scope["tau2"] = canonical_tau2
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "canonical_thresholds": {
                "mu":   canonical_mu,
                "tau2": canonical_tau2,
            },
            "scope":      canonical_scope,
            "baselines":  rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
