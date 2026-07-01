"""Regenerate Fig/q4_ablation.pdf WITHOUT the semantic-transition bar
(P_sem removed from the framework). Values from the ablation text."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FULL = 0.68
# (label, tech-LCS) component removals, will sort descending
rows = [
    ("Full SCOPE",              0.68),
    (r"$-$ self-loop/wildcard", 0.67),
    (r"$-$ causal",             0.65),
    (r"$-$ shared-entity",      0.65),
    (r"$-$ tactical",           0.64),
    (r"$-$ distributional map", 0.62),
    (r"$-$ behavior grouping",  0.55),
]
rows_sorted = sorted(rows, key=lambda r: r[1])  # ascending -> plotted bottom..top
labels = [r[0] for r in rows_sorted]
vals = [r[1] for r in rows_sorted]
colors = ["#1f4e79" if l == "Full SCOPE" else "#8fb4d9" for l in labels]

fig, ax = plt.subplots(figsize=(5.0, 3.0))
y = range(len(labels))
ax.barh(list(y), vals, color=colors, edgecolor="#33465c", height=0.62)
ax.axvline(FULL, ls=":", color="#1f4e79", lw=1.2)
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Technique-LCS")
ax.set_xlim(0.5, 0.71)
for i, v in zip(y, vals):
    ax.text(v + 0.002, i, f"{v:.2f}", va="center", fontsize=7.5)
ax.grid(True, axis="x", ls="--", alpha=0.35)
fig.tight_layout()

for out in [Path(__file__).resolve().parents[2] / "Fig" / "q4_ablation.pdf",
            Path(r"D:/Lab/EDR_Agent/Paper/CCS_Project-main/coumputer&security/Fig/q4_ablation.pdf")]:
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print("saved:", out)
