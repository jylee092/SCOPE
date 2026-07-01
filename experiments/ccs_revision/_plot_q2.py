"""Regenerate Fig/q2_robust.pdf with the new D=0 SCOPE curve + re-scored baselines.
SCOPE (D=0, no-Psem): 0%=0.683, 10%=0.5738, 25%=0.5976, 50%=0.5738.
Baselines unchanged (skip/P_sem don't affect them); 0% from tab:main, 10/25/50 re-scored."""
import sys
from pathlib import Path
from statistics import mean
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config
from experiments._robustness_run import _score_alerts_chain_lcs

BASE = config.OUTPUT_BASE_DIR / "_robustness"
DROPS = [10, 25, 50]

def score_baseline(method):
    out = {}
    for p in DROPS:
        d = BASE / f"{method}_drop{p}_seed0"
        vals = []
        for rp in d.rglob("result.json"):
            s = _score_alerts_chain_lcs(rp)
            if s:
                vals.append(s[1])
        out[p] = mean(vals) if vals else 0.0
    return out

# 0% from tab:main (technique-LCS)
zero = {"SCOPE": 0.683, "Sigma": 0.40, "MAGIC": 0.39, "DeepAG": 0.39, "SHIELD": 0.19}
scope = {10: 0.5738, 25: 0.5976, 50: 0.5738}

curves = {"SCOPE": [zero["SCOPE"], scope[10], scope[25], scope[50]]}
for m in ["Sigma", "MAGIC", "DeepAG", "SHIELD"]:
    b = score_baseline(m.lower())
    curves[m] = [zero[m], b[10], b[25], b[50]]
    print(m, "10/25/50 re-scored:", [round(b[p], 3) for p in DROPS])
print("SCOPE curve:", curves["SCOPE"])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
x = [0, 10, 25, 50]
style = {
    "SCOPE":  dict(color="#1f4e79", lw=2.6, marker="o", ms=6, zorder=5),
    "DeepAG": dict(color="#e07b39", lw=1.6, marker="s", ms=4),
    "Sigma":  dict(color="#4a4a4a", lw=1.6, marker="^", ms=4),
    "MAGIC":  dict(color="#6aa84f", lw=1.6, marker="D", ms=4),
    "SHIELD": dict(color="#a64d79", lw=1.6, marker="v", ms=4),
}
fig, ax = plt.subplots(figsize=(5.0, 3.2))
for m in ["SCOPE", "DeepAG", "Sigma", "MAGIC", "SHIELD"]:
    ax.plot(x, curves[m], label=m, **style[m])
ax.axhline(0.40, ls=":", color="gray", lw=1.0)
ax.text(1, 0.415, "best baseline @0%", fontsize=7, color="gray")
ax.set_xlabel("Random log-drop rate (%)")
ax.set_ylabel("Technique-LCS")
ax.set_xticks(x)
ax.set_ylim(0, 0.75)
ax.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.9)
ax.grid(True, ls="--", alpha=0.35)
fig.tight_layout()
out = ROOT / "Fig" / "q2_robust.pdf"
out.parent.mkdir(exist_ok=True)
fig.savefig(out)
# also copy to paper Fig dir if separate
paper_fig = Path(r"D:/Lab/EDR_Agent/Paper/CCS_Project-main/coumputer&security/Fig/q2_robust.pdf")
if paper_fig.parent.exists():
    fig.savefig(paper_fig)
    print("saved paper copy:", paper_fig)
print("saved:", out)
