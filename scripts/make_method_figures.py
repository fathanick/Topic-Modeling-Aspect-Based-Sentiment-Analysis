"""Two conceptual figures the Methods section needs.

Fig 2 (BERTopic pipeline) and Fig 3 (prompt structure) were previously stock
images that showed settings this study no longer uses (min_cluster_size=10,
n_components=5, Llama 3.1). Redrawn from model_manifest.json and the actual
SYSTEM_PROMPT in src/sentiment_analysis/sentence_absa.py.

Run: ./.venv-figs/bin/python scripts/make_method_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output/figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "font.family": "DejaVu Sans",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

man = json.loads((ROOT / "output/topic_modeling_final/model_manifest.json").read_text())
umap = man["umap"]
vec = man["vectorizer"]


def box(ax, x, y, w, h, title, body, fc="#ffffff"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="#2c3e50",
                           lw=1.0, transform=ax.transAxes, zorder=2))
    ax.text(x + w / 2, y + h * 0.74, title, ha="center", va="center",
            fontsize=7.2, weight="bold", transform=ax.transAxes, zorder=3)
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center",
            fontsize=6.1, transform=ax.transAxes, zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=9,
                                 lw=0.9, color="#2c3e50", zorder=1))


# ---------------------------------------------------------------- BERTopic
fig, ax = plt.subplots(figsize=(7.0, 2.0))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

steps = [
    ("Reviews", "1,530\ncleaned reviews"),
    ("Embeddings", "MiniLM-L12-v2\nmultilingual\n(384-d)"),
    ("UMAP", f"k={umap['n_neighbors']}\n"
             f"d={umap['n_components']}\nmin_dist={umap['min_dist']}"),
    ("HDBSCAN", f"size={man['min_cluster_size']}\n"
                f"{man['cluster_selection_method']}\nEuclidean"),
    ("c-TF-IDF", f"min_df={vec['min_df']}\n"
                 f"ngram={tuple(vec['ngram_range'])}"),
    ("Topics", f"{man['n_topics']} topics\n{man['outlier_pct']}%\noutliers"),
]
n = len(steps)
w, gap = 0.145, 0.024
x = 0.003
for i, (t, b) in enumerate(steps):
    box(ax, x, 0.22, w, 0.56, t, b,
        fc="#eef4f8" if i in (0, 5) else "#ffffff")
    if i < n - 1:
        arrow(ax, x + w + 0.004, 0.50, x + w + gap - 0.004, 0.50)
    x += w + gap
ax.text(0.5, 0.06, "Topic words are produced by c-TF-IDF only; the vectoriser "
        "never feeds the embeddings.", ha="center", fontsize=6.4, style="italic",
        transform=ax.transAxes)
fig.savefig(FIG / "fig_bertopic_pipeline.png")
plt.close(fig)
print("wrote fig_bertopic_pipeline.png")

# ---------------------------------------------------------------- prompt
fig, ax = plt.subplots(figsize=(7.0, 3.5))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.add_patch(Rectangle((0.01, 0.06), 0.98, 0.90, facecolor="#fbfbfb",
                       edgecolor="#2c3e50", lw=1.0))
ax.text(0.03, 0.91, "SYSTEM", fontsize=7.4, weight="bold")
ax.text(0.03, 0.845,
        "You are an Indonesian tourism review analyst.  ·  12 closed aspects  ·  "
        "slang map  ·  polarity criteria",
        fontsize=6.6)
ax.text(0.03, 0.775, "USER", fontsize=7.4, weight="bold")
ax.text(0.03, 0.735, '{"sentence": "<one sentence of the review>"}', fontsize=6.6,
        family="monospace")

ax.add_patch(Rectangle((0.03, 0.30), 0.94, 0.38, facecolor="#ffffff",
                       edgecolor="#9aa8b5", lw=0.8, ls="--"))
ax.text(0.05, 0.635, "CONSTRAINTS", fontsize=7.0, weight="bold")
for k, line in enumerate([
    "1. keyword must be copied verbatim from the sentence (no paraphrase)",
    "2. at most 3 keywords per aspect",
    "3. an aspect appears at most once per sentence",
    "4. no aspect from outside the 12-item list",
    "5. one sentiment per aspect: pos | neu | neg",
]):
    ax.text(0.07, 0.575 - k * 0.052, line, fontsize=6.5)

ax.add_patch(Rectangle((0.03, 0.10), 0.94, 0.16, facecolor="#eef7f1",
                       edgecolor="#2e7d5b", lw=0.9))
ax.text(0.05, 0.215, "OUTPUT (single JSON object, greedy decoding)",
        fontsize=7.0, weight="bold", color="#1d5c40")
ax.text(0.05, 0.145,
        '{"aspects": [{"aspect": "accessibility", "keywords": "jalan", '
        '"sentiment": "neg"}]}',
        fontsize=6.4, family="monospace", color="#1d5c40")

ax.text(0.5, 0.015, "Context window 3,072 tokens; prompts above the limit are "
        "rejected rather than truncated.", ha="center", fontsize=6.3,
        style="italic")
fig.savefig(FIG / "fig_prompt.png")
plt.close(fig)
print("wrote fig_prompt.png")
