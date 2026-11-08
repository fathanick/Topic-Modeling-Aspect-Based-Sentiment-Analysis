"""Round-trip test for the v2 span-anchored adjudication."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data/annotation/adjudication_v2_sheet.xlsx"
KEY = ROOT / "data/annotation/adjudication_v2_key_DO_NOT_SHARE.csv"
T1 = ROOT / "data/annotation/_v2rt.xlsx"
T2 = ROOT / "data/annotation/_v2rt.csv"

key = pd.read_csv(KEY).set_index("item_id")

wb = load_workbook(XLSX)
for i, name in enumerate(("ANOTATOR 1", "ANOTATOR 2")):
    ws = wb[name]
    for r in range(2, 62):
        item = ws.cell(row=r, column=1).value
        val = key.loc[item, "llm"]          # synthetic: human agrees with B
        if i == 1 and r <= 6:
            val = "cannot_tell"
        ws.cell(row=r, column=6, value=val)
wb.save(T1)

wb = load_workbook(T1)
rows = []
for name, col in (("ANOTATOR 1", "span_polarity_annotator1"),
                  ("ANOTATOR 2", "span_polarity_annotator2")):
    ws = wb[name]
    for r in range(2, 62):
        rows.append({"item_id": ws.cell(row=r, column=1).value, "col": col,
                     "val": ws.cell(row=r, column=6).value})
flat = pd.DataFrame(rows).pivot(index="item_id", columns="col", values="val").reset_index()
flat.to_csv(T2, index=False)

r = subprocess.run([sys.executable, str(ROOT / "src/evaluation/score_adjudication_v2.py"),
                    "--sheet", str(T2), "--key", str(KEY)], capture_output=True, text=True)
print(r.stdout)
assert r.returncode == 0, r.stderr[-800:]
assert "items 60" in r.stdout
assert "exact agreement 100.0%" in r.stdout.replace(",", ".") or "exact agreement 100" in r.stdout
assert "head to head" in r.stdout

T1.unlink(); T2.unlink()
(ROOT / "data/annotation/adjudication_v2_scored.csv").unlink(missing_ok=True)
print("V2 ROUND-TRIP OK")
