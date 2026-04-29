"""Collect SHIELD per-scenario timings from existing result.json files."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIELD_DIR = ROOT / "output" / "baselines" / "shield"


def main():
    rows = []
    for p in sorted(SHIELD_DIR.rglob("result.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        n = d.get("notes", {})
        n_ev = int(n.get("n_total_events") or 0)
        sec  = float(n.get("elapsed_sec") or 0.0)
        rows.append({
            "scenario": d.get("scenario", ""),
            "n_events": n_ev,
            "elapsed_sec": sec,
            "n_alerts": int(n.get("n_alerts") or 0),
            "ev_per_sec": (n_ev / sec) if sec > 0 else None,
        })
    print(f"{'scenario':<60} {'events':>8} {'time(s)':>8} {'alerts':>6} {'ev/s':>8}")
    print("-" * 100)
    rows.sort(key=lambda r: r["n_events"])
    for r in rows:
        evs = r["ev_per_sec"]
        evs_s = "  --" if evs is None else f"{evs:.1f}"
        print(f"{r['scenario'][:59]:<60} {r['n_events']:>8} "
              f"{r['elapsed_sec']:>8.1f} {r['n_alerts']:>6} {evs_s:>8}")
    if rows:
        total_ev = sum(r["n_events"] for r in rows)
        total_t  = sum(r["elapsed_sec"] for r in rows)
        print("-" * 100)
        print(f"TOTAL: {total_ev:>8} events  {total_t:>8.1f}s  "
              f"-> aggregate {total_ev/total_t:.1f} ev/s")
        valid = [r["ev_per_sec"] for r in rows if r["ev_per_sec"]]
        if valid:
            print(f"Per-scenario ev/s: min={min(valid):.1f}  "
                  f"median={sorted(valid)[len(valid)//2]:.1f}  "
                  f"max={max(valid):.1f}")
    out = SHIELD_DIR / "_timings.json"
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
