"""
R3 (CCS reviewer ③/⑨): Strict-metric re-scoring, isolated copy.

Reuses the scoring functions from experiments._strict_metrics but writes all
artifacts under output/_ccs_revision/R3_strict/ so the canonical
output/_strict_metrics.json and per-scenario outputs are never touched.

No new inference: reads existing output/.../<scenario>_viterbi.json on disk.

Run:  python -m experiments.ccs_revision.r3_strict
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
from experiments.attack_flows import get_flow
from experiments._strict_metrics import strict_tid_lcs, step_prf, is_false_chain

OUT_DIR = ROOT / "output" / "_ccs_revision" / "R3_strict"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_rows():
    rows = []
    for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
        config.configure_dataset(ds)
        vit_p = config.VITERBI_JSON_PATH
        if not vit_p.exists():
            continue
        try:
            breakdown = json.load(open(vit_p, encoding="utf-8"))
        except Exception:
            continue
        flow = get_flow(config.DATASET_NAME)
        if not flow:
            continue
        pred_tids = [s.get("technique_id") for s in breakdown if s.get("technique_id")]
        strict = strict_tid_lcs(flow, pred_tids)
        prf = step_prf(flow, pred_tids)
        rows.append({
            "scenario": config.DATASET_NAME,
            "ref_steps": len(flow),
            "pred_steps": len(pred_tids),
            "strict_tid_lcs": round(strict, 4),
            "p_strict": prf["strict"][0], "r_strict": prf["strict"][1], "f1_strict": prf["strict"][2],
            "p_plausible": prf["plausible"][0], "r_plausible": prf["plausible"][1], "f1_plausible": prf["plausible"][2],
            "false_chain": int(is_false_chain(flow, pred_tids)),
        })
    return rows


def main():
    rows = collect_rows()
    n = len(rows)
    if n == 0:
        print("[R3] no scenarios with viterbi output found"); return

    macro = {
        "strict_tid_lcs": round(mean(r["strict_tid_lcs"] for r in rows), 4),
        "p_strict": round(mean(r["p_strict"] for r in rows), 4),
        "r_strict": round(mean(r["r_strict"] for r in rows), 4),
        "f1_strict": round(mean(r["f1_strict"] for r in rows), 4),
        "p_plausible": round(mean(r["p_plausible"] for r in rows), 4),
        "r_plausible": round(mean(r["r_plausible"] for r in rows), 4),
        "f1_plausible": round(mean(r["f1_plausible"] for r in rows), 4),
        "false_chain_rate": round(sum(r["false_chain"] for r in rows) / n, 4),
    }

    # JSON
    with open(OUT_DIR / "strict_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"n_scenarios": n, "macro": macro, "rows": rows}, f, ensure_ascii=False, indent=2)
    # CSV (paper table source)
    with open(OUT_DIR / "strict_metrics_per_scenario.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"[R3] {n} scenarios -> {OUT_DIR}")
    print("  macro:")
    for k, v in macro.items():
        print(f"    {k:<18}: {v}")


if __name__ == "__main__":
    main()
