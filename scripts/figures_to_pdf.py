"""Convert the verified PNG figures to PDF for pdflatex, into the LaTeX build dir."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output/figures"
OUT = ROOT / "overleaf_jitt/figures"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = [
    ("fig1_workflow", "fig1_workflow"),
    ("fig_bertopic_pipeline", "fig2_bertopic"),
    ("fig_prompt", "fig3_prompt"),
    ("fig2_topic_sizes", "fig4_topic_sizes"),
    ("fig3_topic_words", "fig5_topic_words"),
    ("fig_topic_by_destination", "fig6_topic_destination"),
    ("fig_aspects_combined", "fig7_aspects"),
    ("fig7_aspect_destination", "fig8_aspect_destination"),
]

for src, dst in ORDER:
    p = FIG / f"{src}.png"
    assert p.exists(), p
    img = mpimg.imread(str(p))
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 200, h / 200))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    out = OUT / f"{dst}.pdf"
    fig.savefig(out, format="pdf", dpi=300)
    plt.close(fig)
    print(f"{dst}.pdf  {out.stat().st_size:,} bytes  from {w}x{h}px")
