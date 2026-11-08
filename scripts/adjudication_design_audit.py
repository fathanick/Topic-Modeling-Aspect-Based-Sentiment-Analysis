"""Quantify the two design confounds in the 60-item adjudication.

C1 MULTI-ASPECT PRIMING: the review contains complaints about OTHER aspects.
   An annotator who sees "tidak ada air" may encode it as a complaint about the
   queried aspect.

C2 NEUTRAL ITEMS: 12 items disagree on whether the label is neutral vs positive/
   negative. The complaint question cannot discriminate positive from neutral, so
   for those items the human answer carries no information about the disagreement.

Also: does the direction survive once both confounds are removed?
"""
import pandas as pd

ab = pd.read_csv("output/absa_llm_gpu/absa_llm_results.csv").set_index("review_id")
k = pd.read_csv("data/annotation/adjudication_key_DO_NOT_SHARE.csv")
s = pd.read_csv("data/annotation/adjudication_completed.csv")
ASP = ["cleanliness", "facilities", "price", "scenery", "atmosphere", "accessibility",
       "crowd", "safety", "service", "food", "historical_value", "photo_spot"]
COMPLAINT_MARKERS = ["tidak", "kurang", "sayang", "mahal", "kotor", "rusak", "susah",
                     "sulit", "bau", "jelek", "panas", "jauh", "penuh", "ramai sekali",
                     "kecewa", "tidak ada", "belum", "minim", "sepi", "mati", "lambat"]

d = k.merge(s.drop(columns=["aspect"]), on="item_id")

rows = []
for _, r in d.iterrows():
    rid, asp = r["_review_id"], r["aspect"]
    row = ab.loc[rid]
    present = [a for a in ASP
               if isinstance(row[f"{a}_sentiment"], str) and row[f"{a}_sentiment"].strip()]
    other_negative = [a for a in present
                      if a != asp and row[f"{a}_sentiment"] == "negative"]
    ev = row[f"{asp}_evidence"] if f"{asp}_evidence" in row else ""
    ev = ev if isinstance(ev, str) else ""
    own_has_complaint = any(m in ev.lower() for m in COMPLAINT_MARKERS)
    rows.append({
        "item_id": r["item_id"], "aspect": asp, "n_aspects": len(present),
        "other_negative": len(other_negative), "other_neg_names": ",".join(other_negative),
        "own_evidence_has_complaint": own_has_complaint,
        "llm": r["_llm_label"], "labA": r["_labellerA_label"],
        "a1": r["contains_complaint_annotator1"], "a2": r["contains_complaint_annotator2"],
    })
t = pd.DataFrame(rows)

print("=== C1: multi-aspect priming ===")
print(f"parent review discusses >1 aspect: {(t.n_aspects>1).sum()}/60")
print(f"parent review has >=1 NEGATIVE aspect other than the queried one: "
      f"{(t.other_negative>0).sum()}/60 = {(t.other_negative>0).mean()*100:.0f}%")
print(f"median other negative aspects: {t.other_negative.median():.0f}")

print("\n=== C2: neutral items cannot be adjudicated by this question ===")
neutral = t.llm.eq("neutral") | t.labA.eq("neutral")
print(f"items involving a neutral label: {neutral.sum()}/60")
print(t[neutral][["item_id","aspect","llm","labA"]].to_string(index=False))

print("\n=== how often do annotators say 'yes' when the review has OTHER complaints? ===")
for col in ("a1", "a2"):
    yy = t[t[col] == "yes"]
    print(f"  {col}: yes {len(yy)} | of those, {((yy.other_negative>0)).sum()} "
          f"({(yy.other_negative>0).mean()*100:.0f}%) have complaints on other aspects")

print("\n=== clean subset: no neutral, and complaint question is meaningful ===")
clean = t[~neutral]
print(f"n = {len(clean)}")
for col in ("a1", "a2"):
    sub = clean[clean[col].isin(["yes", "no"])]
    print(f"  {col}: yes {int((sub[col]=='yes').sum())} / no {int((sub[col]=='no').sum())} "
          f"= {(sub[col]=='yes').mean()*100:.0f}% complaint")

from pathlib import Path
OUT = Path("output/audit"); OUT.mkdir(parents=True, exist_ok=True)
t.to_csv(OUT / "adjudication_design_audit.csv", index=False)
print(f"\nwritten {OUT}/adjudication_design_audit.csv")
