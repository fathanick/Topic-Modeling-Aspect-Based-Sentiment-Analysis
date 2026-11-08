"""Aspect x destination heatmap for the manuscript's Fig. 7 slot.

The DOCX Figure 7 caption promises an aspect-*by-destination* view but the
embedded image (fig_aspects_combined.png) is the global aspect profile. This
rebuilds Fig. 7 from output/sentence_absa/kalimat_aspek_sentimen.csv as the
14 most-reviewed destinations x 12 aspects share matrix.

Run: ./.venv-figs/bin/python scripts/make_fig7_aspect_destination.py
"""
from pathlib import Path

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output/figures"
km = pd.read_csv(ROOT / "output/sentence_absa/kalimat_aspek_sentimen.csv")

N_DEST = 14
c = km.groupby(["destination", "aspect"]).size().unstack(fill_value=0)
tot = c.sum(axis=1)
keep = tot.sort_values(ascending=False).index[:N_DEST]
share = (c.T / tot).T.loc[keep] * 100
order = c.sum().sort_values(ascending=False).index
share = share[order]

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                     "figure.dpi": 300, "savefig.dpi": 300})
VMAX = 30.0
fig, ax = plt.subplots(figsize=(7.2, 4.2))
im = ax.imshow(share.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=VMAX)
ax.set_xticks(range(share.shape[1]))
ax.set_xticklabels([a.replace("_", " ") for a in share.columns],
                   rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(share.shape[0]))
ax.set_yticklabels([d.replace("Pantai ", "P. ").replace("Candi ", "C. ")
                    .replace("Museum ", "Mus. ") for d in share.index], fontsize=7)
for i in range(share.shape[0]):
    for j in range(share.shape[1]):
        v = share.values[i, j]
        if v >= 10:
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.6,
                    color="white" if v >= 0.6 * VMAX else "#1a1a1a")
ax.set_title("Aspect share within each of the 14 most-reviewed destinations (%)",
             fontsize=8.5, pad=6)
fig.colorbar(im, ax=ax, fraction=0.026, pad=0.015, label="share of mentions (%)")
fig.tight_layout()
fig.savefig(FIG / "fig7_aspect_destination.png", bbox_inches="tight")
fig.savefig(FIG / "fig7_aspect_destination.svg")
plt.close(fig)

man = FIG / "figures_manifest.json"
m = json.loads(man.read_text())
m["fig7_aspect_destination_sentence"] = str(
    (FIG / "fig7_aspect_destination.png").relative_to(ROOT))
man.write_text(json.dumps(m, indent=2) + "\n")
print("wrote fig7_aspect_destination.png/.svg")
print(share.round(1).to_string())
