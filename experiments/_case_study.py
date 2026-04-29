"""
Case Study (§7.6) numbers for the msf_record_mic running example:

  normalized δ̃(S*)  — the Viterbi joint score normalized to [0,1]
                     by chain length (geometric-mean per-step score).
  n(S*)              — chain-level novelty against the 53-campaign library.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from pipeline.attack_chain import (
    compute_novelty, load_campaign_library, load_tactic_map,
)


SCEN_VITERBI = (
    config.OUTPUT_BASE_DIR
    / "atomic" / "credential_access" / "empire_mimikatz_logonpasswords"
    / "empire_mimikatz_logonpasswords_2020-08-07103224"
    / "empire_mimikatz_logonpasswords_2020-08-07103224_viterbi.json"
)


def main() -> None:
    with open(SCEN_VITERBI, encoding="utf-8") as f:
        breakdown = json.load(f)
    chain_len = len(breakdown)
    print(f"Chain length: {chain_len}")

    # δ̃(S*) ∈ [0, 1] — geometric mean of (per-step similarity × per-step
    # transition compatibility). Both factors live in [0, 1] natively
    # (cosine similarity in the embedding space, geometric-mean fused
    # transition compatibility), so their geometric mean is interpretable as
    # a per-step quality score in the same range.
    sims  = [s.get("similarity")        for s in breakdown
             if s.get("similarity") is not None]
    trans = [s.get("transition_weight") for s in breakdown
             if s.get("transition_weight") is not None]
    mean_sim   = sum(sims) / len(sims) if sims else 0.0
    mean_trans = sum(trans) / len(trans) if trans else 1.0
    delta_tilde = (mean_sim ** 0.5) * (mean_trans ** 0.5) if trans \
                  else mean_sim
    print(f"mean similarity     : {mean_sim:.4f}")
    print(f"mean transition wt. : {mean_trans:.4f}")
    print(f"normalized δ̃(S*)    : {delta_tilde:.4f}")

    # Novelty
    pred_tactics = [s["tactic"] for s in breakdown if s.get("tactic")]
    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    campaigns = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)
    novelty, closest = compute_novelty(pred_tactics, campaigns)
    print(f"\npred_tactics ({len(pred_tactics)}): {pred_tactics}")
    print(f"closest campaign  : {closest}")
    print(f"n(S*)             : {novelty:.4f}")


if __name__ == "__main__":
    main()
