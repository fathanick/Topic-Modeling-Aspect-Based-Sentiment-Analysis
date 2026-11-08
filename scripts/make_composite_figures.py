"""Two composite figures for the 7 figure slots of the revised manuscript.

Slot layout in the DOCX body:
  1 workflow, 2 BERTopic pipeline, 3 prompt, 4 topic sizes,
  5 topic words, 6 topic x destination, 7 aspect mentions + net scores.

Run: ./.venv-figs/bin/python scripts/make_composite_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "output"
FIG = O / "figures"

plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

km = pd.read_csv(O / "sentence_absa/kalimat_aspek_sentimen.csv")
man = json.loads((O / "topic_modeling_final/model_manifest.json").read_text())

# ---------------------------------------------------------------- topic x destination
ts = pd.read_csv(O / "topic_modeling_final/topics_summary.csv").sort_values("topic_id")
xd = pd.read_csv(O / "topic_modeling_final/topic_x_destination_pct.csv")
print("topic_x_destination cols:", list(xd.columns)[:4], "...")

xd = xd.set_index("destination")
rev_by_dest = km.groupby("destination").sentence.nunique().sort_values(ascending=False)
cols = [c for c in rev_by_dest.index if c in xd.index][:16]   # rows, not columns
mat = xd.loc[cols].astype(float).values.T   # topics as rows, destinations as cols
assert mat.size and np.isfinite(mat).all(), "topic x destination matrix is empty"

fig, ax = plt.subplots(figsize=(7.2, 3.4))
im = ax.imshow(mat, aspect="auto", cmap="magma_r",
               vmin=0, vmax=float(np.nanpercentile(mat, 97)))
ax.set_yticks(range(mat.shape[0]))
ax.set_yticklabels([f"T{i}" for i in range(mat.shape[0])], fontsize=7.5)
ax.set_xticks(range(mat.shape[1]))
ax.set_xticklabels(
    [c.replace("Pantai ", "P. ").replace("Candi ", "C. ")
      .replace("Museum ", "Mus. ").replace("Agrowisata ", "Agro. ")
      .replace("Gumuk Pasir ", "Gumuk ") for c in cols],
    rotation=45, ha="right", fontsize=6.5)
ax.set_title("Topic share within each destination (%)", fontsize=8.5, pad=6)
fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015, label="topic share (%)")
fig.subplots_adjust(bottom=0.26)
fig.savefig(FIG / "fig_topic_by_destination.png")
plt.close(fig)
print("wrote fig_topic_by_destination.png")

# ---------------------------------------------------------------- aspects composite
agg = km.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
for c in ["pos", "neu", "neg"]:
    if c not in agg:
        agg[c] = 0
agg["total"] = agg.sum(axis=1)
agg = agg.sort_values("total", ascending=True)
score = (agg.pos - agg.neg) / agg.total

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6),
                         gridspec_kw={"width_ratios": [1.55, 1.0]})
a = axes[0]
y = np.arange(len(agg))
a.barh(y, agg.pos.values, color="#2e7d5b", label="positive")
a.barh(y, agg["neu"].values, left=agg.pos.values, color="#bdbdbd", label="neutral")
a.barh(y, agg.neg.values, left=(agg.pos + agg["neu"]).values,
       color="#b3453a", label="negative")
a.set_yticks(y)
a.set_yticklabels([i.replace("_", " ") for i in agg.index], fontsize=7.2)
for i, r in enumerate(agg.itertuples()):
    a.text(r.total + 14, i, f"{r.total:,}", va="center", fontsize=6.5)
a.set_xlabel("Aspect mentions", fontsize=7.5)
a.set_xlim(0, agg.total.max() * 1.16)
a.legend(frameon=False, fontsize=6.8, loc="lower right")
a.set_title(f"Volume by polarity ({len(km):,} mentions)", fontsize=8)

b = axes[1]
cols_bar = ["#b3453a" if v < 0 else "#2e7d5b" for v in score.values]
b.barh(y, score.values, color=cols_bar)
b.axvline(0, color="#333", lw=0.9)
b.set_yticks(y)
b.set_yticklabels([])
for i, v in enumerate(score.values):
    b.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}", va="center",
           ha="left" if v >= 0 else "right", fontsize=6.5)
b.set_xlim(min(score.min(), 0) - 0.16, max(score.max(), 0) + 0.16)
b.set_xlabel("Net score (P − N) / T", fontsize=7.5)
b.set_title("Net sentiment", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig_aspects_combined.png")
plt.close(fig)
print("wrote fig_aspects_combined.png")
