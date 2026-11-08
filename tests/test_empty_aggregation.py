"""Tests for the empty-input path of the aggregation step.

The v7 retry kernel died on its second-to-last cell because the aggregation
crashed when there were no mentions, which then surfaced as a confusing
FileNotFoundError one cell later. A run that legitimately produces zero
mentions must still write a well-formed, header-only table.

Run: ./venv/bin/python tests/test_empty_aggregation.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
fails = []


def ck(name, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'} {name}  {detail}")
    if not cond:
        fails.append(name)


COLS = ["sentence_id", "review_id", "sent_index", "n_sentences", "sentence",
        "destination", "stars", "n_aspects", "aspects", "status", "error", "raw"]


def make(path, rows):
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)


with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    # ---- all sentences yield no aspect
    empty_rows = [
        {"sentence_id": f"0_0{i}", "review_id": 0, "sent_index": i, "n_sentences": 3,
         "sentence": "Udaranya panas banget.", "destination": "Prambanan", "stars": 5,
         "n_aspects": 0, "aspects": "[]", "status": "ok", "error": "", "raw": '{"aspects": []}'}
        for i in range(3)
    ]
    make(td / "empty.csv", empty_rows)
    r = subprocess.run([sys.executable, str(ROOT / "src/analysis/sentence_aspect_table.py"),
                        "--input", str(td / "empty.csv"), "--out", str(td / "out")],
                       capture_output=True, text=True)
    ck("empty run exits 0", r.returncode == 0, r.stderr.strip()[:120])
    out = td / "out" / "kalimat_aspek_sentimen.csv"
    ck("empty run writes the table", out.exists())
    if out.exists():
        t = pd.read_csv(out)
        ck("empty table has 0 rows", len(t) == 0, str(len(t)))
        ck("empty table keeps its header",
           list(t.columns) == ["sentence_id", "destination", "aspect",
                               "keywords", "sentiment", "sentence"], str(list(t.columns)))

    # ---- one real mention, to prove the normal path still works
    rows = [dict(empty_rows[0]),
            {"sentence_id": "0_01", "review_id": 0, "sent_index": 1, "n_sentences": 3,
             "sentence": "Toiletnya bersih.", "destination": "Prambanan", "stars": 5,
             "n_aspects": 1,
             "aspects": json.dumps([{"aspect": "cleanliness", "keywords": "Toiletnya",
                                     "sentiment": "pos", "span": "Toiletnya bersih"}]),
             "status": "ok", "error": "", "raw": "{}"}]
    make(td / "one.csv", rows)
    r = subprocess.run([sys.executable, str(ROOT / "src/analysis/sentence_aspect_table.py"),
                        "--input", str(td / "one.csv"), "--out", str(td / "out2")],
                       capture_output=True, text=True)
    ck("normal run exits 0", r.returncode == 0, r.stderr.strip()[:120])
    t = pd.read_csv(td / "out2" / "kalimat_aspek_sentimen.csv")
    ck("normal run has 1 row", len(t) == 1, str(len(t)))
    ck("normal run keeps the aspect", t.iloc[0].aspect == "cleanliness")

print("\n" + ("ALL EMPTY-AGGREGATION CHECKS PASSED" if not fails else f"FAILED: {fails}"))
raise SystemExit(1 if fails else 0)
