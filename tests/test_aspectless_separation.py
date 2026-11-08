"""Invariants that must hold for the deliverable tables.

The aspect-less sentences are reported separately and carry NO sentiment. That
is only true if the two tables never intersect: a sentence with a mention must
not appear in the aspect-less table, and vice versa. A join in the paper would
silently double-count otherwise.

Run: ./venv/bin/python tests/test_aspectless_separation.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_unlabelled_sentences import classify  # noqa: E402

fails = []


def ck(name, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'} {name}  {detail}")
    if not cond:
        fails.append(name)


al = pd.read_csv(ROOT / "output/audit/aspectless_sentences.csv")
km = pd.read_csv(ROOT / "output/sentence_absa/kalimat_aspek_sentimen.csv")
raw = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw.csv")

# 1. no sentiment, no aspect: the whole point
ck("tidak ada kolom sentiment", "sentiment" not in al.columns, str(list(al.columns)))
ck("tidak ada kolom aspect", "aspect" not in al.columns)
ck("tidak ada kolom keywords", "keywords" not in al.columns)

# 2. the two tables partition the sentence set
ck("union = semua kalimat",
   len(set(al.sentence_id) | set(km.sentence_id)) == len(raw) == 6479,
   str(len(set(al.sentence_id) | set(km.sentence_id))))
ck("irisan kosong", len(set(al.sentence_id) & set(km.sentence_id)) == 0)
ck("jumlah = 6.479", len(al) + km.sentence_id.nunique() == 6479,
   f"{len(al)} + {km.sentence_id.nunique()}")

# 3. every aspect-less sentence really has n_aspects == 0
z = raw[raw.n_aspects == 0]
ck("aspectless = kalimat n_aspects 0", set(al.sentence_id) == set(z.sentence_id))
ck("tabel mention tetap punya aspect+sentiment",
   {"aspect", "sentiment", "keywords"} <= set(km.columns))

# 4. resolution values are exhaustive and disjoint
ck("dua nilai resolution", al.resolution.nunique() == 2,
   str(sorted(al.resolution.unique())))
ck("resolution lengkap", al.resolution.notna().all())
ck("1248 + 61 = 1309",
   int((al.resolution == "no_aspect_no_mention").sum()) == 1248
   and int((al.resolution == "review_has_no_aspect_either").sum()) == 61)

# 5. the classifier column is reproducible
ck("cls bisa direproduksi", (al.sentence.apply(classify) == al.cls).all())

# 6. the mention table stays clean
ck("sentiment hanya pos/neu/neg",
   km.sentiment.isin(["pos", "neu", "neg"]).all())
ck("tidak ada sentiment kosong", km.sentiment.notna().all())
ck("keywords tidak kosong", (km.keywords.astype(str).str.len() > 0).all())

# 7. grounding survives the round trip
bad = 0
for r in km.itertuples():
    if r.keywords not in r.sentence:
        bad += 1
ck("grounding keywords", bad == 0, f"{bad} violations")

print("\n" + ("ALL SEPARATION CHECKS PASSED" if not fails else f"FAILED: {fails}"))
raise SystemExit(1 if fails else 0)
