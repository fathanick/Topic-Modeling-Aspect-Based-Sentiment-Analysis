"""End-to-end test of the aggregation step on synthetic model output."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TMP_IN = ROOT / "output/sentence_absa/_test_raw.csv"
TMP_OUT = ROOT / "output/sentence_absa/_test_out"

sents = [
    # 3 aspects, 2 polarities (review 180 case)
    ("0_00", "Hanya saja air pantai kanonya kotor, sisi sebelahnya jernih namun ombaknya keras.", [
        {"aspect": "cleanliness", "keywords": "kotor", "sentiment": "neg", "span": "air pantai kanonya kotor"},
        {"aspect": "scenery", "keywords": "jernih", "sentiment": "pos", "span": "sisi sebelahnya jernih"},
        {"aspect": "safety", "keywords": "ombak", "sentiment": "neg", "span": "ombaknya keras"},
    ]),
    # 2 aspects, 2 polarities
    ("1_00", "Pantainya bersih, tapi parkirnya sempit.", [
        {"aspect": "cleanliness", "keywords": "bersih", "sentiment": "pos", "span": "bersih"},
        {"aspect": "facilities", "keywords": "parkir", "sentiment": "neg", "span": "parkirnya sempit"},
    ]),
    # single aspect
    ("2_00", "Tiket masuk Rp 10.000.", [
        {"aspect": "price", "keywords": "Tiket masuk", "sentiment": "neu", "span": "Tiket masuk Rp 10.000"},
    ]),
    # no aspect
    ("3_00", "Selamat pagi semua.", []),
]

rows = []
for sid, sent, items in sents:
    rows.append({"sentence_id": sid, "review_id": int(sid.split("_")[0]),
                 "sent_index": 0, "n_sentences": 1, "sentence": sent,
                 "destination": "X", "stars": 4, "n_aspects": len(items),
                 "aspects": json.dumps(items, ensure_ascii=False),
                 "status": "ok", "error": ""})
TMP_IN.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(TMP_IN, index=False)

r = subprocess.run([sys.executable, str(ROOT / "src/analysis/sentence_aspect_table.py"),
                    "--input", str(TMP_IN), "--out", str(TMP_OUT)],
                   capture_output=True, text=True)
print(r.stdout[-1800:])
assert r.returncode == 0, r.stderr[-800:]

m = pd.read_csv(TMP_OUT / "sentence_aspect_mentions.csv")
assert len(m) == 6, len(m)
assert (m.sentiment == "neu").sum() == 1

# the 3-aspect sentence must show 2 polarities
s0 = m[m.sentence_id == "0_00"]
assert len(s0) == 3 and s0.sentiment.nunique() == 2, s0

# the deliverable table must carry keywords
t = pd.read_csv(TMP_OUT / "kalimat_aspek_sentimen.csv")
assert list(t.columns) == ["sentence_id", "destination", "aspect", "keywords",
                           "sentiment", "sentence"], t.columns
assert t.keywords.notna().all()

# cleanup
for p in TMP_OUT.glob("*"):
    p.unlink()
TMP_OUT.rmdir()
TMP_IN.unlink()
print("\nSENTENCE TABLE OK")
