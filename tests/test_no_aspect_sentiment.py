"""Tests for the rescued no-aspect sentiment table.

The 104 sentences carry a polarity the model DID emit. They must not leak into
the aspect-mention table (a mention requires an aspect from the glossary), and
the three negated examples called out during review must still be present.

Run: ./venv/bin/python tests/test_no_aspect_sentiment.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
fails = []


def ck(name, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'} {name}  {detail}")
    if not cond:
        fails.append(name)


na = pd.read_csv(ROOT / "output/sentence_absa/no_aspect_sentiment.csv")
al = pd.read_csv(ROOT / "output/audit/aspectless_sentences.csv")
km = pd.read_csv(ROOT / "output/sentence_absa/kalimat_aspek_sentimen.csv")
raw = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw.csv")

# 1. shape
ck("104 baris", len(na) == 104, str(len(na)))
ck("tidak ada kolom aspect di tabel ini", "aspect" not in na.columns)
ck("punya sentence_sentiment", "sentence_sentiment" in na.columns)

# 2. every row keeps a polarity
ck("semua punya sentimen",
   na.sentence_sentiment.isin(["pos", "neg", "neu", "mixed"]).all())

# 3. the rescued rows come from the reason group, not from thin air
rej = set(al[al.reason == "rejected_aspect_outside_glossary"].sentence_id)
ck("id cocok dengan reason group", set(na.sentence_id) == rej,
   f"{len(set(na.sentence_id))} vs {len(rej)}")

# 4. they are NOT mentions: no overlap with the mention table
ck("tidak masuk tabel mention",
   len(set(na.sentence_id) & set(km.sentence_id)) == 0)
ck("mention tetap 8.412", len(km) == 8412, str(len(km)))

# 5. they all really have n_aspects == 0
z = raw.set_index("sentence_id")
ck("semua n_aspects 0", (z.loc[na.sentence_id].n_aspects == 0).all())

# 6. the model really emitted a pair for each of them
ok = True
for r in na.itertuples():
    emitted = json.loads(z.loc[r.sentence_id].raw or "{}").get("aspects", [])
    if not emitted:
        ok = False
        break
ck("raw memang memuat pair", ok)

# 7. the three negated sentences singled out during review are present
targets = {
    "Parah bgt sih.": "neg",
    "Sepi di hari libur nasional.": "neg",
}
for text, want in targets.items():
    hit = na[na.sentence.str.strip() == text]
    ck(f"ada: {text[:34]!r}", len(hit) == 1 and hit.iloc[0].sentence_sentiment == want,
       hit.iloc[0].sentence_sentiment if len(hit) else "MISSING")

hit = na[na.sentence.str.contains("tidak akan menyarankan sama sekali", na=False)]
ck("ada: 'tidak akan menyarankan sama sekali'",
   len(hit) >= 1 and (hit.sentence_sentiment == "neg").all(),
   str(hit.sentence_sentiment.tolist()))

# 8. the 1.205 must stay unlabelled
ck("1.205 tanpa sentimen", int((al.reason == "model_returned_empty_list").sum()) == 1205)
ck("reason lengkap 1309", int(al.reason.notna().sum()) == 1309)

# 9. neg count matches what was reported
ck("neg = 14", int((na.sentence_sentiment == "neg").sum()) == 14,
   str(int((na.sentence_sentiment == "neg").sum())))

print("\n" + ("ALL NO-ASPECT-SENTIMENT CHECKS PASSED" if not fails else f"FAILED: {fails}"))
raise SystemExit(1 if fails else 0)
