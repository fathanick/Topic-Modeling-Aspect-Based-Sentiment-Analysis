"""Regenerate the paper figures from the CURRENT artifacts.

Figures that the old manuscript embeds were produced by a superseded run
(23 topics, 4,926 review-level mentions). Everything here is rebuilt from:

  data/processed/yogyakarta_tourism_reviews_preprocessed.csv   (1,530 reviews)
  output/topic_modeling_final/model_manifest.json              (params)
  output/topic_modeling_final/*.csv                            (topic results)
  output/sentence_absa/kalimat_aspek_sentimen.csv              (8,412 mentions)

Writes PNG + SVG into output/figures/ and a manifest so the manuscript
verifier can assert every embedded figure matches the current numbers.

Run: ./.venv-figs/bin/python scripts/make_paper_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "output"
FIG = O / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

manifest = {}


def save(fig, name):
    p = FIG / f"{name}.png"
    fig.savefig(p)
    fig.savefig(FIG / f"{name}.svg")
    plt.close(fig)
    manifest[name] = str(p.relative_to(ROOT))
    print(f"  wrote {p.relative_to(ROOT)}")


# ---------------------------------------------------------------- data
ts = pd.read_csv(O / "topic_modeling_final/topics_summary.csv").sort_values("topic_id")
man = json.loads((O / "topic_modeling_final/model_manifest.json").read_text())
km = pd.read_csv(O / "sentence_absa/kalimat_aspek_sentimen.csv")
raw = pd.read_csv(O / "sentence_absa/sentence_absa_raw.csv")
xd = pd.read_csv(O / "topic_modeling_final/topic_x_destination_pct.csv")

N_REV = 1530
N_TOPICS = len(ts)
N_OUT = man["n_outliers"]
COVER = man["coverage_pct"]

print("=== figures ===")

# ---------------------------------------------------------------- Fig 1 pipeline
fig, ax = plt.subplots(figsize=(7.0, 2.3))
ax.axis("off")
boxes = [
    ("Data collection\n1,530 reviews\n34 destinations", 0.02),
    ("Sentence split\n6,552 sentences", 0.27),
    ("Aspect extraction\nQwen2.5-7B\n12 aspects, 4-bit", 0.52),
    ("Analysis\n8,412 mentions\nsentence level", 0.77),
]
for txt, x in boxes:
    ax.add_patch(Rectangle((x, 0.30), 0.20, 0.42, fill=False, lw=1.1,
                           edgecolor="#2c3e50", transform=ax.transAxes))
    ax.text(x + 0.10, 0.51, txt, ha="center", va="center", fontsize=7.0,
            transform=ax.transAxes)
for x in (0.225, 0.475, 0.725):
    ax.annotate("", xy=(x + 0.043, 0.51), xytext=(x, 0.51),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#2c3e50"))
ax.text(0.5, 0.08, "BERTopic (8 topics, c_v = 0.591) is applied to the "
        "1,530 preprocessed reviews; ABSA is applied to sentences.",
        ha="center", fontsize=7, style="italic", transform=ax.transAxes)
save(fig, "fig1_workflow")

# ---------------------------------------------------------------- Fig 2 topic barchart
fig, ax = plt.subplots(figsize=(6.4, 3.4))
labels = [f"T{r.topic_id}" for r in ts.itertuples()]
vals = ts.document_count.values
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(vals)))
bars = ax.barh(labels[::-1], vals[::-1], color=colors)
for b, v, pc in zip(bars, vals[::-1], ts.percentage.values[::-1]):
    ax.text(b.get_width() + 8, b.get_y() + b.get_height() / 2,
            f"{int(v)} ({float(str(pc).replace('%', '')):.1f}%)", va="center", fontsize=7.5)
ax.set_xlabel("Reviews assigned")
ax.set_xlim(0, max(vals) * 1.22)
ax.set_title(f"BERTopic topics (n = {N_TOPICS}); {COVER:.1f}% of {N_REV:,} reviews "
             f"assigned, {N_OUT} outliers ({man['outlier_pct']}%)", fontsize=8.5)
save(fig, "fig2_topic_sizes")

# ---------------------------------------------------------------- Fig 3 topic words
fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.axis("off")
ncol = 5
ax.text(0.0, 1.02, "Topic", fontsize=7.5, weight="bold", transform=ax.transAxes)
ax.text(0.11, 1.02, "n", fontsize=7.5, weight="bold", transform=ax.transAxes)
ax.text(0.19, 1.02, "Most representative terms (c-TF-IDF order)",
        fontsize=7.5, weight="bold", transform=ax.transAxes)
row_h = 1.0 / (len(ts) + 0.6)
for i, r in enumerate(ts.itertuples()):
    y = 0.94 - i * row_h
    if i % 2 == 0:
        ax.add_patch(Rectangle((0.0, y - row_h * 0.36), 1.0, row_h * 0.9,
                               facecolor="#f2f5f7", edgecolor="none",
                               transform=ax.transAxes, zorder=0))
    terms = [x.strip() for x in str(r.top_words).split(",")][:ncol]
    ax.text(0.0, y, f"T{r.topic_id}", fontsize=7.5, transform=ax.transAxes, va="center")
    ax.text(0.11, y, f"{r.document_count}", fontsize=7.5,
            transform=ax.transAxes, va="center")
    ax.text(0.19, y, "  ·  ".join(terms), fontsize=7,
            transform=ax.transAxes, va="center", color="#22313f")
ax.set_title(f"Representative terms for the {N_TOPICS} topics", fontsize=8.5, pad=14)
save(fig, "fig3_topic_words")

# ---------------------------------------------------------------- Fig 4 aspect mentions
p = km.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
for c in ["pos", "neu", "neg"]:
    if c not in p:
        p[c] = 0
p = p.sort_values("pos", ascending=True)
p = p.assign(total=p.sum(1)).sort_values("total", ascending=True)
fig, ax = plt.subplots(figsize=(6.6, 3.8))
y = np.arange(len(p))
ax.barh(y, p.pos.values, color="#2e7d5b", label="positive")
ax.barh(y, p["neu"].values, left=p.pos.values, color="#b8b8b8", label="neutral")
ax.barh(y, p.neg.values, left=(p.pos + p["neu"]).values, color="#b3453a", label="negative")
ax.set_yticks(y)
ax.set_yticklabels(p.index, fontsize=8)
for i, r in enumerate(p.itertuples()):
    ax.text(r.total + 12, i, f"{r.total:,}", va="center", fontsize=7)
ax.set_xlabel("Aspect mentions (extracted per sentence)")
ax.set_xlim(0, p.total.max() * 1.14)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.set_title(f"Aspect mentions by polarity, {len(km):,} mentions "
             f"from {len(raw):,} sentences", fontsize=8.5)
ax.set_yticklabels([str(i).replace("_", " ") for i in p.index], fontsize=8)
save(fig, "fig4_aspect_mentions")

# ---------------------------------------------------------------- Fig 5 sentiment score
sc = ((p.pos - p.neg) / p.total).sort_values()
fig, ax = plt.subplots(figsize=(6.2, 3.4))
cols = ["#b3453a" if v < 0 else "#2e7d5b" for v in sc.values]
ax.barh(sc.index, sc.values, color=cols)
ax.axvline(0, color="#333", lw=0.9)
for i, v in enumerate(sc.values):
    ax.text(v + (0.015 if v >= 0 else -0.015), i, f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
ax.set_xlabel("Net sentiment score (P − N) / T")
ax.set_xlim(min(sc.min(), 0) - 0.13, max(sc.max(), 0) + 0.13)
ax.set_title("Net sentiment by aspect", fontsize=8.5)
save(fig, "fig5_aspect_scores")

# ---------------------------------------------------------------- Fig 6 ceiling
per = km.groupby("sentence_id").sentiment.agg(
    n="size",
    has_pos=lambda s: (s == "pos").any(),
    has_neg=lambda s: (s == "neg").any())
n_multi = int((per.n >= 2).sum())
n_mixed = int((per.has_pos & per.has_neg).sum())
n_sent = len(raw)
n_rev = raw.review_id.nunique()

k2 = km.assign(review_id=km.sentence_id.map(raw.set_index("sentence_id").review_id))
rv = k2.groupby("review_id").sentiment.agg(
    has_pos=lambda s: (s == "pos").any(), has_neg=lambda s: (s == "neg").any())
n_comp = int((rv.has_pos & rv.has_neg).sum())
n_rv = len(rv)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
a = axes[0]
vals = [n_sent, int((raw.n_aspects > 0).sum()), n_multi, n_mixed]
labs = ["sentences\nlabelled", "≥1\naspect", "≥2\naspects", "pos +\nneg"]
a.bar(range(len(vals)), vals, color=["#34495e", "#2e7d5b", "#4d8fb0", "#b3453a"])
for i, v in enumerate(vals):
    a.text(i, v + max(vals) * 0.02, f"{v:,}", ha="center", fontsize=7.5)
a.set_xticks(range(len(vals)))
a.set_xticklabels(labs, fontsize=7)
a.set_ylabel("Sentences")
a.set_ylim(0, max(vals) * 1.16)
a.set_title("Nested structure within sentences", fontsize=8.5)
a.text(0.5, -0.34, "categories are subsets of the one to their left",
       transform=a.transAxes, ha="center", fontsize=6.5, style="italic")

b = axes[1]
vals2 = [n_rv, n_comp, n_rv - n_comp]
a2 = b.barh(["one polarity", "competing\npolarities"], [n_rv - n_comp, n_comp],
            color=["#95a5a6", "#b3453a"])
for bb, v in zip(a2, [n_rv - n_comp, n_comp]):
    b.text(v + 12, bb.get_y() + bb.get_height() / 2,
           f"{v:,} ({100*v/n_rv:.1f}%)", va="center", fontsize=7.5)
b.set_xlim(0, n_rv * 1.28)
b.set_xlabel(f"Reviews with mentions (n = {n_rv:,})")
b.set_title("Polarity a review-level label cannot hold", fontsize=8.5)
fig.tight_layout()
save(fig, "fig6_representational_gap")

# ---------------------------------------------------------------- Fig 7 aspect x destination
fig, ax = plt.subplots(figsize=(7.2, 4.2))
combo = (km.groupby(["aspect", "destination"]).size().unstack(fill_value=0))
top = combo.loc[combo.sum(axis=1).sort_values(ascending=False).index[:8]]
top = top[top.sum(axis=0).sort_values(ascending=False).index[:14]]
share = top / combo.sum(axis=0)[top.columns]
im = ax.imshow(share.values, aspect="auto", cmap="viridis",
               vmin=float(np.nanpercentile(share.values, 3)),
               vmax=float(np.nanpercentile(share.values, 99.5)))


def short(name):
    return (name.replace("Pantai ", "P. ")
                 .replace("Candi ", "C. ")
                 .replace("Museum ", "Mus. ")
                 .replace("Agrowisata ", "Agro. ")
                 .replace("Gumuk Pasir ", "Gumuk ")
                 .replace("Makam Raja-Raja ", "Makam ")
                 .replace("Studio Alam ", "Studio "))


ax.set_xticks(range(share.shape[1]))
ax.set_xticklabels([short(c) for c in share.columns], rotation=45,
                   ha="right", fontsize=7)
ax.set_yticks(range(share.shape[0]))
ax.set_yticklabels([s.replace("_", " ") for s in share.index], fontsize=7.5)
for i in range(share.shape[0]):
    for j in range(share.shape[1]):
        v = share.values[i, j]
        if v >= 0.22:
            frac = (v - im.norm.vmin) / (im.norm.vmax - im.norm.vmin)
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if frac > 0.6 else "#101010")
ax.set_title("Aspect mix across the 14 most-reviewed destinations",
             fontsize=8.5, pad=8)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="share of mentions")
fig.subplots_adjust(bottom=0.24)
save(fig, "fig7_aspect_destination")

(FIG / "figures_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
