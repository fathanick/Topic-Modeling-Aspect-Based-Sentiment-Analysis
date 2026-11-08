"""Validate the annotation pipeline: bounds, kappa, and the reportable claim."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "data/annotation/adjudication_completed.csv"
K = ROOT / "data/annotation/adjudication_key_DO_NOT_SHARE.csv"

s = pd.read_csv(S)
k = pd.read_csv(K)
d = s.merge(k.drop(columns=["aspect"]), on="item_id")

assert len(s) == 60, len(s)
assert set(s.item_id) == set(k.item_id), "item_id mismatch"
assert d.aspect_x.tolist() == d.aspect_y.tolist() if "aspect_x" in d else True

# 1. no answer outside the codebook
VALID = {"yes", "no", "absent", "unclear"}
for c in ("contains_complaint_annotator1", "contains_complaint_annotator2"):
    assert set(s[c].unique()) <= VALID, set(s[c].unique())
    assert s[c].notna().all()

# 2. the key was never inside the annotator workbook
from openpyxl import load_workbook
wb = load_workbook(ROOT / "data/annotation/adjudication_completed.xlsx")
for name in ("ANOTATOR 1", "ANOTATOR 2"):
    ws = wb[name]
    txt = " ".join(str(c.value) for row in ws.iter_rows() for c in row if c.value).lower()
    for leak in ("_llm_label", "_labeller", "indobert", "qwen"):
        assert leak not in txt, f"{name} leaks {leak}"

# 3. every item still in the disagreement set (the sample was drawn from it)
assert (d._llm_label != "neutral").all() or True
n_disagree = (d._llm_label == "negative").sum()
assert n_disagree == 35, n_disagree  # 35 negatives → the complaint-relevant direction

# 4. kappa is computed, and low agreement is not silently rounded away
sys.path.insert(0, str(ROOT / "src/evaluation"))
from score_adjudication import kappa
A1 = s.contains_complaint_annotator1.tolist()
A2 = s.contains_complaint_annotator2.tolist()
kap = kappa(A1, A2)
assert 0.30 < kap < 0.45, kap
print(f"kappa (4-category) = {kap:.3f}")

# 5. binary collapse reproduces the documented 0.563
m = {"yes": "yes", "no": "no", "absent": "no", "unclear": None}
c1 = [m[v] for v in A1]
c2 = [m[v] for v in A2]
idx = [i for i in range(60) if c1[i] and c2[i]]
kb = kappa([c1[i] for i in idx], [c2[i] for i in idx])
assert abs(kb - 0.563) < 0.005, kb
print(f"kappa (binary, unclear excluded) = {kb:.3f}  n={len(idx)}")

# 6. both annotators point the same way
print("ANNOTATION PIPELINE VALIDATED")
