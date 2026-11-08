"""Audit every mention saved by the fuzzy matcher.

Fuzzy recovery is the one place where a stored span is NOT the model's own
string, so each recovery has to be inspected: if the similarity is low the
matcher may have attached the wrong part of the sentence to an aspect.

Run: ./venv/bin/python scripts/audit_fuzzy_recovery.py
"""
import difflib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sentiment_analysis"))

from sentence_absa import _first_json, _recover, _recover_fuzzy  # noqa: E402
from remap_aspects import canonical  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Permanent baseline, not /tmp: the OS wipes /tmp and this audit
# then fails silently weeks later.
orig = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw_noremap.csv")
cur = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw.csv")
print(f"mentions {orig.n_aspects.sum()} -> {cur.n_aspects.sum()}")

rows = []
for r in orig.itertuples():
    if not isinstance(r.raw, str) or len(r.raw) >= 600:
        continue
    d = _first_json(r.raw)
    if not isinstance(d, dict):
        continue
    for it in (d.get("aspects") or []):
        if not isinstance(it, dict) or not canonical(str(it.get("aspect", ""))):
            continue
        for f in ("keywords", "span"):
            v = str(it.get(f, "")).strip()
            if not v or _recover(v, r.sentence) is not None:
                continue
            got = _recover_fuzzy(v, r.sentence)
            if got is None:
                continue
            # score with the same normalisation the matcher uses; comparing raw
            # strings reports "mkanan" vs "makanan" as a 0.48 mismatch when the
            # recovery is in fact correct.
            nrm = lambda s: re.sub(r"[^\w]", "", s.lower())
            rows.append({
                "sentence_id": r.sentence_id, "field": f,
                "model_wrote": v, "stored": got,
                "sim": round(difflib.SequenceMatcher(None, nrm(v), nrm(got)).ratio(), 3),
                "sentence": r.sentence,
            })

f = pd.DataFrame(rows)
print(f"fuzzy-recovered strings: {len(f)}")
if not len(f):
    raise SystemExit

print("\nsimilarity distribution:")
bins = [(0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
for lo, hi in bins:
    n = int(((f.sim >= lo) & (f.sim < hi)).sum())
    print(f"  {lo:.2f}-{hi:.2f}: {n:4d}")

print(f"\nby field: {f.field.value_counts().to_dict()}")

print("\n" + "=" * 70)
print("LOWEST 25 — these are the ones that could be wrong")
print("=" * 70)
for _, r in f.nsmallest(25, "sim").iterrows():
    print(f"  [{r.field}] sim={r.sim:.3f}")
    print(f"    model : {r.model_wrote!r}")
    print(f"    stored: {r.stored!r}")
    print(f"    sent  : {r.sentence[:110]}")

# a fuzzy recovery must never be empty, enormous, or longer than the sentence
bad = f[(f.stored.str.len() == 0) | (f.stored.str.len() > f.sentence.str.len())]
print(f"\nstructurally invalid recoveries: {len(bad)}")

# every stored value must still be a literal substring
not_sub = f[~f.apply(lambda r: r.stored in r.sentence, axis=1)]
print(f"stored value not a literal substring: {len(not_sub)}")

f.to_csv(ROOT / "output/audit/fuzzy_recoveries.csv", index=False)
print(f"\nwritten {ROOT/'output/audit/fuzzy_recoveries.csv'}")
