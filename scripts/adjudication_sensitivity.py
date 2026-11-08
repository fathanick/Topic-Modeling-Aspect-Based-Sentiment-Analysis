"""Sensitivity analysis: does the conclusion survive annotator choice?

Reports the adjudication verdict three ways:
  - annotator 1 alone
  - annotator 2 alone
  - consensus only (items where both agreed; conflicts excluded)
and on the binary scheme the paper will actually use.
"""
import pandas as pd

s = pd.read_csv("data/annotation/adjudication_completed.csv")
k = pd.read_csv("data/annotation/adjudication_key_DO_NOT_SHARE.csv").drop(columns=["aspect"])
d = s.merge(k, on="item_id")

# The paper's substantive question is binary:
#   did the review contain a complaint about this aspect?
BIN = {"yes": "complaint", "no": "no_complaint", "absent": "no_complaint", "unclear": None}


def verdict(row, col):
    v = BIN.get(row[col])
    if v is None:
        return "undecided"
    llm, lab = row["_llm_label"], row["_labellerA_label"]
    # In the disagreement set the two labellers said different things.
    said_negative = llm == "negative"
    if v == "complaint":
        return "aspect-level" if said_negative else "review-level"
    return "review-level" if said_negative else "aspect-level"


for col, label in (("contains_complaint_annotator1", "annotator 1"),
                   ("contains_complaint_annotator2", "annotator 2")):
    d["v"] = d.apply(lambda r: verdict(r, col), axis=1)
    print(f"\n=== {label} alone ===")
    print(d.v.value_counts().to_string())

# consensus: only items where both gave the same non-unclear answer
both = d[(d.contains_complaint_annotator1 == d.contains_complaint_annotator2) &
         (d.contains_complaint_annotator1 != "unclear")]
print(f"\n=== consensus only (n={len(both)}) ===")
print(f"agreement rate on the whole sheet: "
      f"{(d.contains_complaint_annotator1 == d.contains_complaint_annotator2).mean()*100:.1f}%")
both = both.copy()
both["v"] = both.apply(lambda r: verdict(r, "contains_complaint_annotator1"), axis=1)
print(both.v.value_counts().to_string())
print("\nby aspect:")
print(pd.crosstab(both.aspect, both.v).to_string())

# how many conflict items would flip the headline
print("\n=== conflict items: what each says ===")
conf = d[d.contains_complaint_annotator1 != d.contains_complaint_annotator2].copy()
conf["v1"] = conf.apply(lambda r: verdict(r, "contains_complaint_annotator1"), axis=1)
conf["v2"] = conf.apply(lambda r: verdict(r, "contains_complaint_annotator2"), axis=1)
print(pd.crosstab(conf.v1, conf.v2).to_string())
print(f"\nconflicts where the two annotators point to DIFFERENT labellers: "
      f"{((conf.v1 != conf.v2) & (conf.v1.isin(['aspect-level','review-level'])) & (conf.v2.isin(['aspect-level','review-level']))).sum()}")
