"""Write a review sheet for the aspect-less sentences, for manual inspection.

Written as a spreadsheet so Pak Fathan can read the actual text and mark
disagreements. No polarity is assigned anywhere in it: the sheet exists to test
the claim that "no aspect" cannot mean "neutral", not to relabel.

Run: ./venv/bin/python scripts/make_aspectless_review_sheet.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw.csv")
al = pd.read_csv(ROOT / "output/audit/aspectless_sentences.csv")

# --- 40 aspect-less sentences, layered by class so every category is visible
parts = []
for c in ["A too short", "B question", "C descriptive",
          "D evaluative cue", "E aspect keyword"]:
    sub = al[al.cls == c]
    n = {"A too short": 8, "B question": 6, "C descriptive": 10,
         "D evaluative cue": 10, "E aspect keyword": 10}[c]
    parts.append(sub.sample(min(n, len(sub)), random_state=11))
sample = pd.concat(parts).drop_duplicates("sentence_id")

# --- 20 genuine `neu` mentions, to sit next to them for contrast
neu_rows = []
for r in raw.itertuples():
    for a in json.loads(r.aspects or "[]"):
        if a["sentiment"] == "neu":
            neu_rows.append({"kind": "neu_mention", "cls": "",
                             "aspect": a["aspect"], "keywords": a["keywords"],
                             "stars": int(r.stars),
                             "sentence": r.sentence})
neu = pd.DataFrame(neu_rows).sample(20, random_state=7)

a = sample.assign(kind="no_aspect", aspect="", keywords="")
a = a[["kind", "cls", "aspect", "keywords", "stars", "sentence"]]
sheet = pd.concat([a, neu[["kind", "cls", "aspect", "keywords", "stars", "sentence"]]])
sheet = sheet.sort_values(["kind", "cls"], ascending=[False, True]).reset_index(drop=True)
sheet.insert(0, "no", range(1, len(sheet) + 1))
sheet["cek_pak_fathan"] = ""

out = ROOT / "output/audit/aspectless_review_sheet.csv"
sheet.to_csv(out, index=False)

xlsx = ROOT / "output/audit/aspectless_review_sheet.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
    sheet.to_excel(w, index=False, sheet_name="cek")
    ws = w.sheets["cek"]
    ws.column_dimensions["A"].width = 5
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 18
    ws.column_dimensions["E"].width = 22
    ws.column_dimensions["F"].width = 7
    ws.column_dimensions["G"].width = 80
    ws.column_dimensions["H"].width = 18
    ws.freeze_panes = "A2"

print(f"rows: {len(sheet)}  (no_aspect {int((sheet.kind=='no_aspect').sum())}, "
      f"neu_mention {int((sheet.kind=='neu_mention').sum())})")
print(f"written {out}")
print(f"written {xlsx}")
