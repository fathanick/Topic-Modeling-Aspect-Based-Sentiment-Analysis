"""Rescue the polarity the model DID emit for aspect-less sentences.

For 104 of the 1.309 aspect-less sentences the model did not return an empty
list: it returned an aspect outside the 12-item glossary together with a
sentiment, and the parser rejected the pair on the aspect name only. The
polarity for those is recorded in `raw` and can be separated from the aspect
without inventing anything.

For the other 1.205 the model returned `{"aspects": []}` — there is no stored
polarity at all, and labelling it would require an external judgement.

Writes a SEPARATE table. Nothing is merged into the mention table.

Run: ./venv/bin/python scripts/extract_no_aspect_sentiment.py
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw.csv")
al = pd.read_csv(ROOT / "output/audit/aspectless_sentences.csv")


def emitted(s):
    try:
        return json.loads(s).get("aspects", []) or []
    except Exception:
        return []


z = raw[raw.n_aspects == 0].copy()
z["emit"] = z.raw.fillna("").apply(emitted)
z["n_emit"] = z.emit.apply(len)

rows = []
for r in z[z.n_emit > 0].itertuples():
    # one sentence can carry several rejected pairs; keep each polarity, but
    # collapse to a single sentence-level polarity when they agree
    sents = [a.get("sentiment") for a in r.emit if a.get("sentiment") in ("pos", "neu", "neg")]
    if not sents:
        continue
    uniq = sorted(set(sents))
    label = uniq[0] if len(uniq) == 1 else "mixed"
    rows.append({
        "sentence_id": r.sentence_id,
        "review_id": r.review_id,
        "destination": r.destination,
        "stars": int(r.stars),
        "n_pairs": len(r.emit),
        "model_aspect": "; ".join(sorted({str(a.get("aspect")) for a in r.emit})),
        "model_sentiment": "; ".join(sents),
        "sentence_sentiment": label,
        "sentence": r.sentence,
    })

na = pd.DataFrame(rows)
out = ROOT / "output/sentence_absa/no_aspect_sentiment.csv"
na.to_csv(out, index=False)

print("=== kalimat tanpa aspek, dipisah menurut apa yang model keluarkan ===")
print(f"total tanpa aspek                     {len(z)}")
print(f"  model keluarkan daftar kosong       {int((z.n_emit == 0).sum())}")
print(f"  model keluarkan pair lalu ditolak   {int((z.n_emit > 0).sum())}")
print(f"    dari itu, sentimen bisa diselamatkan {len(na)}")
print()
print("=== distribusi sentimen yang diselamatkan ===")
print(na.sentence_sentiment.value_counts().to_string())
print()
print("=== aspek yang model sebut (di luar glosarium) ===")
c = Counter()
for v in na.model_aspect:
    for a in str(v).split("; "):
        c[a] += 1
print(pd.Series(c).sort_values(ascending=False).head(12).to_string())
print()
print("=== contoh ===")
for r in na.head(10).itertuples():
    print(f"  [{r.sentence_sentiment}] ({r.model_aspect}) {r.sentence[:90]}")
print(f"\nwritten {out} ({len(na)} rows)")
