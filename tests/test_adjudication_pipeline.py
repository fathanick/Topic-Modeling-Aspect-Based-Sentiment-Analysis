"""End-to-end test for the adjudication pipeline.

Builds a synthetic filled sheet from the real key, runs score_adjudication.py,
and asserts the output is coherent. Confirms the scoring path works before any
annotator ever opens the CSV.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "data/annotation/adjudication_sheet.csv"
KEY = ROOT / "data/annotation/adjudication_key_DO_NOT_SHARE.csv"
TMP = ROOT / "data/annotation/_test_synthetic.csv"

sheet = pd.read_csv(SHEET)
key = pd.read_csv(KEY).set_index("item_id")

# Synthetic labeler: says "yes" when the LLM said negative, "no" otherwise.
# Annotator 2 differs on 6 items to exercise kappa.
sheet["contains_complaint_annotator1"] = [
    "yes" if key.loc[i, "_llm_label"] == "negative" else "no" for i in sheet.item_id
]
sheet["contains_complaint_annotator2"] = sheet["contains_complaint_annotator1"].copy()
sheet.loc[sheet.index[:6], "contains_complaint_annotator2"] = "unclear"
sheet.to_csv(TMP, index=False)

r = subprocess.run(
    [sys.executable, str(ROOT / "src/evaluation/score_adjudication.py"),
     "--sheet", str(TMP), "--key", str(KEY)],
    capture_output=True, text=True,
)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr)
    raise SystemExit(1)

assert "items: 60" in r.stdout, "did not read 60 items"
assert "Cohen's kappa" in r.stdout, "kappa not reported"
assert "headline sentence" in r.stdout, "headline not produced"
assert (ROOT / "data/annotation/adjudication_scored.csv").exists(), "no scored file"

scored = pd.read_csv(ROOT / "data/annotation/adjudication_scored.csv")
assert len(scored) == 60
assert "supports" in scored.columns
TMP.unlink()
print("\nADJUDICATION PIPELINE OK (synthetic)")
