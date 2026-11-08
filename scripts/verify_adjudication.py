"""Verify every number written into DRAFT_adjudication_results.md."""
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
s = pd.read_csv(ROOT / "data/annotation/adjudication_completed.csv")
k = pd.read_csv(ROOT / "data/annotation/adjudication_key_DO_NOT_SHARE.csv").drop(columns=["aspect"])
d = s.merge(k, on="item_id")
BIN = {"yes": "complaint", "no": "no_complaint", "absent": "no_complaint", "unclear": None}
fail = []


def ck(label, got, want, tol=5e-4):
    ok = abs(got - want) <= tol
    print(f"{'OK ' if ok else 'FAIL'} {label}: {got} vs {want}")
    if not ok:
        fail.append(label)


def wilson(kk, n, z=1.96):
    p = kk / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h) * 100, (c + h) * 100


def verdict(row, col):
    v = BIN.get(row[col])
    if v is None:
        return "undecided"
    if v == "complaint":
        return "aspect-level" if row["_llm_label"] == "negative" else "review-level"
    return "review-level" if row["_llm_label"] == "negative" else "aspect-level"


# sample description
ck("items", len(s), 60)
ck("aspects", s.aspect.nunique(), 12)
ck("per aspect min", s.aspect.value_counts().min(), 5)
ck("per aspect max", s.aspect.value_counts().max(), 5)

# annotator marginals
v1 = s.contains_complaint_annotator1.value_counts().to_dict()
v2 = s.contains_complaint_annotator2.value_counts().to_dict()
for cat, want in (("yes", 32), ("no", 19), ("absent", 8), ("unclear", 1)):
    ck(f"A1 {cat}", v1.get(cat, 0), want)
for cat, want in (("yes", 28), ("no", 15), ("absent", 13), ("unclear", 4)):
    ck(f"A2 {cat}", v2.get(cat, 0), want)

# agreement
agree = (s.contains_complaint_annotator1 == s.contains_complaint_annotator2).sum()
ck("agreement n", agree, 36)
ck("agreement pct", round(agree / 60 * 100, 1), 60.0)
ck("conflicts", 60 - agree, 24)

# verdicts
d["v1"] = d.apply(lambda r: verdict(r, "contains_complaint_annotator1"), axis=1)
d["v2"] = d.apply(lambda r: verdict(r, "contains_complaint_annotator2"), axis=1)
ck("A1 aspect-level", (d.v1 == "aspect-level").sum(), 40)
ck("A1 review-level", (d.v1 == "review-level").sum(), 19)
ck("A1 undecided", (d.v1 == "undecided").sum(), 1)
ck("A2 aspect-level", (d.v2 == "aspect-level").sum(), 40)
ck("A2 review-level", (d.v2 == "review-level").sum(), 16)
ck("A2 undecided", (d.v2 == "undecided").sum(), 4)

# proportions + CIs
for col, vcol, name, kk, nn, lo, hi in (
    ("[A1]", "v1", "A1", 40, 59, 55.1, 78.3),
    ("[A2]", "v2", "A2", 40, 56, 58.5, 81.6),
):
    dec = d[d[vcol] != "undecided"]
    ck(f"{name} decidable n", len(dec), nn)
    ck(f"{name} supporting", (dec[vcol] == "aspect-level").sum(), kk)
    ck(f"{name} pct", round(kk / nn * 100, 1), round(kk / nn * 100, 1))
    elo, ehi = wilson(kk, nn)
    ck(f"{name} CI low", round(elo, 1), lo)
    ck(f"{name} CI high", round(ehi, 1), hi)

both = d[(d.contains_complaint_annotator1 == d.contains_complaint_annotator2) &
         (d.contains_complaint_annotator1 != "unclear")]
ck("consensus n", len(both), 36)
ck("consensus aspect-level", (both.v1 == "aspect-level").sum(), 28)
ck("consensus review-level", (both.v1 == "review-level").sum(), 8)
elo, ehi = wilson(28, 36)
ck("consensus CI low", round(elo, 1), 61.9)
ck("consensus CI high", round(ehi, 1), 88.3)

# conflict structure
conf = d[d.contains_complaint_annotator1 != d.contains_complaint_annotator2]
mb = conf[conf.contains_complaint_annotator1.eq("absent") | conf.contains_complaint_annotator2.eq("absent")]
un = conf[conf.contains_complaint_annotator1.eq("unclear") | conf.contains_complaint_annotator2.eq("unclear")]
mb2 = mb[~mb.index.isin(un.index)]
cn = conf[~conf.index.isin(mb.index) & ~conf.index.isin(un.index)]
ck("mention boundary", len(mb2), 11)
ck("evaluation vs description", len(un), 5)
ck("complaint vs no-complaint", len(cn), 8)
ck("mention boundary pct", round(len(mb2) / len(conf) * 100), 46)
ck("evaluation pct", round(len(un) / len(conf) * 100), 21)
ck("complaint pct", round(len(cn) / len(conf) * 100), 33)

flip = ((conf.v1 != conf.v2) & conf.v1.isin(["aspect-level", "review-level"])
        & conf.v2.isin(["aspect-level", "review-level"]))
ck("flip", flip.sum(), 12)

print("\nALL OK" if not fail else f"\nFAILED: {fail}")
raise SystemExit(1 if fail else 0)
