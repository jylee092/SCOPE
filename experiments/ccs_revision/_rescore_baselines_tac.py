"""Re-score baselines (tech-LCS recall, tac-LCS, step-cov, order) at drop 0/10/25/50
using the SAME chain_align scorer as SCOPE. Reads EXISTING saved chains (no LLM).
Fixes R8's tac_lcs=0 (which passed tactic="") by supplying real tactic_sequence.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from experiments.chain_align import evaluate_chain_alignment
from experiments.attack_flows import get_flow

BASE0 = ROOT / "output" / "baselines"          # 0% drop baseline chains
ROB   = ROOT / "output" / "_robustness"        # drop 10/25/50 chains
METHODS = ["sigma", "magic", "deepag", "shield"]


def load_chains(result_dir: Path):
    """scenario -> (tech_seq, tac_seq), lockstep-filtered on non-empty tech."""
    out = {}
    for rp in result_dir.rglob("result.json"):
        try:
            d = json.load(open(rp, encoding="utf-8"))
        except Exception:
            continue
        sc = d.get("scenario")
        if sc is None:
            continue
        techs = d.get("technique_sequence") or []
        tacs  = d.get("tactic_sequence") or []
        pairs = [(t, (tacs[i] if i < len(tacs) else "")) for i, t in enumerate(techs) if t]
        out[sc] = pairs
    return out


def score(chains: dict):
    tech, tac, step, order, n = [], [], [], [], 0
    for scen, pairs in chains.items():
        flow = get_flow(scen)
        if not flow:
            continue
        bd = [{"technique_id": t, "tactic": tc} for t, tc in pairs]
        ca = evaluate_chain_alignment(scen, bd, flow)
        if "error" in ca:
            continue
        tech.append(ca["technique_lcs_norm"])
        tac.append(ca["tactic_lcs_norm"])
        step.append(ca["step_coverage"])
        order.append(ca["order_accuracy"])
        n += 1
    m = lambda x: round(mean(x), 4) if x else 0.0
    return {"n": n, "tech_lcs": m(tech), "tac_lcs": m(tac),
            "step_cov": m(step), "order": m(order)}


def main():
    print(f"{'method':<8}{'drop':>5}{'n':>4}{'tech':>8}{'tac':>8}{'step':>8}{'order':>8}")
    for name in METHODS:
        for drop in [0, 10, 25, 50]:
            if drop == 0:
                d = BASE0 / name
            else:
                d = ROB / f"{name}_drop{drop}_seed0"
            if not d.exists():
                print(f"{name:<8}{drop:>5}  MISSING {d}")
                continue
            r = score(load_chains(d))
            print(f"{name:<8}{drop:>5}{r['n']:>4}{r['tech_lcs']:>8.4f}"
                  f"{r['tac_lcs']:>8.4f}{r['step_cov']:>8.4f}{r['order']:>8.4f}")


if __name__ == "__main__":
    main()
