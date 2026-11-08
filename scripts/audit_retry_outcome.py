"""Did the retry set actually close the gap, or was the gap over-counted?

The 238 came from a heuristic that flagged "has an evaluative cue or an aspect
keyword". A cue is not the same as an attributable aspect: "Udaranya panas
banget" has a cue but names no aspect from the 12-item glossary, and whole-visit
praise ("Saya cukup puas") maps to `overall`, which is deliberately unmapped.

So the retry run is also a test of the heuristic, not just of the model.

Run: ./venv/bin/python scripts/audit_retry_outcome.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sentiment_analysis"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_unlabelled_sentences import classify  # noqa: E402

loc = pd.read_csv(ROOT / "output/sentence_absa_retry_local/sentence_absa_raw.csv")
loc["cls"] = loc.sentence.apply(classify)

print("=== retry run outcome ===")
print(f"retry sentences           {len(loc)}")
got = int((loc.n_aspects > 0).sum())
print(f"now carry an aspect       {got} ({got/len(loc)*100:.1f}%)")
print(f"still zero                {int((loc.n_aspects==0).sum())}")
print(f"mentions recovered        {int(loc.n_aspects.sum())}")
print()
print("by original heuristic class:")
for c in sorted(loc.cls.unique()):
    sub = loc[loc.cls == c]
    print(f"  {c:<22} n={len(sub):4d}  resolved={int((sub.n_aspects>0).sum()):4d} "
          f"({(sub.n_aspects>0).mean()*100:5.1f}%)")

print("\n=== still-zero, sampled ===")
for r in loc[(loc.n_aspects == 0) & (loc.cls == "D evaluative cue")].head(8).itertuples():
    print(f"  D  {r.sentence[:100]}")
for r in loc[(loc.n_aspects == 0) & (loc.cls == "E aspect keyword")].head(8).itertuples():
    print(f"  E  {r.sentence[:100]}")

print("\n=== mentions per aspect, retry only ===")
asps = [a["aspect"] for v in loc.aspects for a in json.loads(v or "[]")]
print(pd.Series(asps).value_counts().to_string())
