"""Round-trip: does a filled XLSX actually score correctly?"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data/annotation/adjudication_sheet.xlsx"
KEY = ROOT / "data/annotation/adjudication_key_DO_NOT_SHARE.csv"
TMP_XLSX = ROOT / "data/annotation/_rt.xlsx"
TMP_CSV = ROOT / "data/annotation/_rt.csv"

key = pd.read_csv(KEY).set_index("item_id")

# Fill both annotator tabs programmatically, as a human would.
wb = load_workbook(XLSX)
for i, name in enumerate(("ANOTATOR 1", "ANOTATOR 2")):
    ws = wb[name]
    for r in range(2, 62):
        item = ws.cell(row=r, column=1).value
        # synthetic: "yes" when the LLM said negative (i.e. complaint present)
        val = "yes" if key.loc[item, "_llm_label"] == "negative" else "no"
        if i == 1 and r <= 7:          # annotator 2 differs on 6 items
            val = "unclear"
        ws.cell(row=r, column=4, value=val)
wb.save(TMP_XLSX)

# Flatten the workbook into the CSV shape score_adjudication.py expects.
wb = load_workbook(TMP_XLSX)
out = []
for name, col in (("ANOTATOR 1", "contains_complaint_annotator1"),
                  ("ANOTATOR 2", "contains_complaint_annotator2")):
    ws = wb[name]
    for r in range(2, 62):
        out.append({"item_id": ws.cell(row=r, column=1).value, "col": col,
                    "val": ws.cell(row=r, column=4).value})
flat = pd.DataFrame(out).pivot(index="item_id", columns="col", values="val").reset_index()
flat.to_csv(TMP_CSV, index=False)

r = subprocess.run(
    [sys.executable, str(ROOT / "src/evaluation/score_adjudication.py"),
     "--sheet", str(TMP_CSV), "--key", str(KEY)],
    capture_output=True, text=True)
print(r.stdout[-900:])
assert r.returncode == 0, r.stderr[-600:]
assert "Cohen's kappa: 0.81" in r.stdout, "kappa mismatch"
assert "annotator 1 filled: 60/60" in r.stdout
assert "annotator 2 filled: 60/60" in r.stdout

TMP_XLSX.unlink()
TMP_CSV.unlink()
(ROOT / "data/annotation/adjudication_scored.csv").unlink(missing_ok=True)
print("\nXLSX ROUND-TRIP OK")
