"""Samples for reviewing the "is no-aspect = neutral?" question.

Neutral in this schema is a VALUE the model assigned to a specific aspect. An
aspect-less sentence has no aspect to assign anything to. Print both side by
side so the difference is visible in the text, not just in the counts.

Run: ./venv/bin/python scripts/sample_neutral_vs_aspectless.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_unlabelled_sentences import classify  # noqa: E402

raw = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw.csv")
al = pd.read_csv(ROOT / "output/audit/aspectless_sentences.csv")

# ---- 1. what `neu` actually looks like
neu_rows = []
for r in raw.itertuples():
    for a in json.loads(r.aspects or "[]"):
        if a["sentiment"] == "neu":
            neu_rows.append({"aspect": a["aspect"], "keywords": a["keywords"],
                             "span": a["span"], "sentence": r.sentence})
neu = pd.DataFrame(neu_rows)

print("=== SENTIMEN NEUTRAL YANG SUNGGUHAN (ada aspek, nilai neu) ===")
print(f"total mention neu: {len(neu)} "
      f"({len(neu)/int(raw.n_aspects.sum())*100:.1f}% dari semua mention)\n")
for r in neu.sample(12, random_state=7).itertuples():
    print(f"  [{r.aspect}] <{r.keywords}>")
    print(f"    {r.sentence[:120]}")
print()

# ---- 2. aspect-less samples, one per class
print("=== KALIMAT TANPA ASPEK (tidak ada yang bisa dinilai) ===")
for c in ["A too short", "B question", "C descriptive",
          "D evaluative cue", "E aspect keyword"]:
    sub = al[al.cls == c]
    print(f"\n-- {c}  (n={len(sub)})")
    for r in sub.sample(min(4, len(sub)), random_state=11).itertuples():
        print(f"    {r.sentence[:120]}")

# ---- 3. the 61 sentences in reviews with no aspect at all
print("\n=== REVIEW YANG TIDAK PUNYA SATU PUN ASPEK (61 kalimat) ===")
solo = al[al.resolution == "review_has_no_aspect_either"]
for r in solo.sample(min(8, len(solo)), random_state=3).itertuples():
    print(f"  [{int(r.stars)}*] {r.sentence[:110]}")
