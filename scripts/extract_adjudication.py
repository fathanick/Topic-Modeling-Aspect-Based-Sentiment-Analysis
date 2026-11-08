"""Extract filled annotator tabs from the completed workbook into the CSV shape
score_adjudication.py expects, with integrity checks."""
import pandas as pd
from openpyxl import load_workbook

XLSX = "data/annotation/adjudication_completed.xlsx"
OUT = "data/annotation/adjudication_completed.csv"
VALID = {"yes", "no", "absent", "unclear"}

wb = load_workbook(XLSX)
tabs = {}
for name, col in (("ANOTATOR 1", "contains_complaint_annotator1"),
                  ("ANOTATOR 2", "contains_complaint_annotator2")):
    ws = wb[name]
    rows = []
    for r in range(2, ws.max_row + 1):
        rows.append({
            "item_id": ws.cell(row=r, column=1).value,
            "aspect": ws.cell(row=r, column=2).value,
            "val": ws.cell(row=r, column=4).value,
            "notes": ws.cell(row=r, column=5).value,
        })
    df = pd.DataFrame(rows)
    df["val"] = df["val"].astype(str).str.strip().str.lower().replace({"none": "", "nan": ""})
    tabs[name] = df
    filled = (df.val != "").sum()
    print(f"{name}: {filled}/{len(df)} filled")
    bad = set(df.val[df.val != ""].unique()) - VALID
    if bad:
        print(f"  INVALID VALUES: {sorted(bad)}")
    print("  distribution:", df.val.value_counts().to_dict())
    empty = df[df.val == ""].item_id.tolist()
    if empty:
        print(f"  blank items: {empty}")

# build merged csv
a = tabs["ANOTATOR 1"].set_index("item_id")
b = tabs["ANOTATOR 2"].set_index("item_id")
out = pd.DataFrame({
    "item_id": a.index,
    "aspect": a.aspect,
    "contains_complaint_annotator1": a.val.values,
    "contains_complaint_annotator2": b.reindex(a.index).val.values,
})
out.to_csv(OUT, index=False)
print(f"\nwritten {OUT} ({len(out)} rows)")
print("both filled:", ((out.contains_complaint_annotator1 != "") &
                      (out.contains_complaint_annotator2 != "")).sum())
