# Reproducibility: figures, audits, manuscript

Every figure in the manuscript is produced by Matplotlib from a result file in
`output/`. No image is hand-drawn or AI-generated; `scripts/audit_figure_provenance.py`
proves it (matplotlib signature + MD5 of embedded bytes == bytes on disk).

## Environments

- `venv/` — analysis pipeline (BERTopic, pandas, ...)
- `.venv-figs/` — figure generation only (matplotlib, numpy, pandas)

```bash
python -m venv .venv-figs
./.venv-figs/bin/pip install matplotlib numpy pandas
```

## Figures

```bash
./.venv-figs/bin/python scripts/make_paper_figures.py        # Fig. 4, 5 (+ workflow)
./.venv-figs/bin/python scripts/make_method_figures.py       # Fig. 2, 3
./.venv-figs/bin/python scripts/make_composite_figures.py    # Fig. 6, 7
./.venv-figs/bin/python scripts/make_fig6_aspect_by_destination.py  # Fig. 8
./.venv-figs/bin/python scripts/figures_to_pdf.py            # PNG -> PDF for LaTeX
```

| Fig. | File | Script | Source artifact |
|---|---|---|---|
| 1 | `fig1_workflow.png` | `make_paper_figures.py` | `model_manifest.json` |
| 2 | `fig_bertopic_pipeline.png` | `make_method_figures.py` | `model_manifest.json` |
| 3 | `fig_prompt.png` | `make_method_figures.py` | `sentence_absa.py` `SYSTEM_PROMPT` |
| 4 | `fig2_topic_sizes.png` | `make_paper_figures.py` | `topics_summary.csv` |
| 5 | `fig3_topic_words.png` | `make_paper_figures.py` | `topics_summary.csv` |
| 6 | `fig_topic_by_destination.png` | `make_composite_figures.py` | `topic_x_destination_pct.csv` |
| 7 | `fig_aspects_combined.png` | `make_composite_figures.py` | `kalimat_aspek_sentimen.csv` |
| 8 | `fig7_aspect_destination.png` | `make_fig6_aspect_by_destination.py` | `kalimat_aspek_sentimen.csv` |

## Audits

```bash
../.venv/bin/python scripts/audit_figure_provenance.py   # Fig. provenance + MD5
../.venv/bin/python scripts/audit_docx_images.py         # placed vs native aspect ratio
../.venv/bin/python scripts/audit_citations.py <pdf>     # refs vs in-text callouts
../.venv/bin/python scripts/validate_manuscript_revision.py
```

## Manuscript numbers

The authoritative scores come from `scripts/review_level_baseline.py`, written to
`output/audit/review_level_baseline.json`. Review-level: tourism value `+0.204`,
culture/heritage `+0.141`, facilities `+0.088`. Facilities receives 900 mentions
(fourth-highest). BERTopic final parameters: `min_cluster_size=15`,
`min_topic_size=15`, `eom`; 8 topics; 94.5% coverage; 84 outliers (5.5%).

## Building the LaTeX mirror

```bash
cd overleaf_jitt
tectonic -X compile main.tex --outdir build --keep-logs
```

`sn-mathphys-num.bst` lives in the project root so the bibliography resolves.
