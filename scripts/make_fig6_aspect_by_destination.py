"""Rebuild the aspect-by-destination heatmap (paper Fig. 6 slot) from raw output.

Pure Matplotlib over output/sentence_absa/kalimat_aspek_sentimen.csv — no
generated imagery, no synthetic values. Axis 0 = aspect, axis 1 = destination;
cells are the share of that destination's aspect mentions taken by the aspect,
which is what the manuscript sentence reports.

Run: ./.venv-figs/bin/python scripts/make_fig6_aspect_by_destination.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output/figures"
km = pd.read_csv(ROOT / "output/sentence_absa/kalimat_aspek_sentimen.csv")

N_ASPECT, N_DEST = 8, 14
combo = km.groupby(["aspect", "destination"]).size().unstack(fill_value=0)
top_aspects = combo.sum(axis=1).sort_values(ascending=False).index[:N_ASPECT]
top_dests = combo.sum(axis=0).sort_values(ascending=False).index[:N_DEST]
share = combo.loc[top_aspects, top_dests] / combo.sum(axis=0)[top_dests]

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                     "figure.dpi": 300, "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(7.2, 4.2))
pct = share.values * 100  # one unit system for cells, annotations and colorbar
im = ax.imshow(pct, aspect="auto", cmap="viridis", vmin=0, vmax=30)


def short(name):
    return (name.replace("Pantai ", "P. ").replace("Candi ", "C. ")
                .replace("Museum ", "Mus. ").replace("Agrowisata ", "Agro. ")
                .replace("Gumuk Pasir ", "Gumuk ").replace("Makam Raja-Raja ", "Makam ")
                .replace("Studio Alam ", "Studio "))


ax.set_xticks(range(share.shape[1]))
ax.set_xticklabels([short(c) for c in share.columns], rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(share.shape[0]))
ax.set_yticklabels([s.replace("_", " ") for s in share.index], fontsize=7.5)
for i in range(share.shape[0]):
    for j in range(share.shape[1]):
        v = pct[i, j]
        if v >= 18:
            # dark text on the bright upper half of viridis, white below
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                    color="#101010" if im.norm(v) > 0.55 else "white")
ax.set_xticks(np.arange(-0.5, share.shape[1]), minor=True)
ax.set_yticks(np.arange(-0.5, share.shape[0]), minor=True)
ax.grid(which="minor", color="white", linewidth=0.4)
ax.tick_params(which="minor", length=0)
ax.set_title("Aspect share within each of the 14 most-reviewed destinations (%)",
             fontsize=8.5, pad=8)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="share of that destination's mentions (%)")
fig.subplots_adjust(bottom=0.26)
fig.savefig(FIG / "fig7_aspect_destination.png")
fig.savefig(FIG / "fig7_aspect_destination.svg")
plt.close(fig)

print("aspects:", list(share.index))
print("destinations:", list(share.columns))
print(f"max share {pct.max():.1f}% at",
      share.stack().idxmax())
print("wrote fig7_aspect_destination.png/.svg")
