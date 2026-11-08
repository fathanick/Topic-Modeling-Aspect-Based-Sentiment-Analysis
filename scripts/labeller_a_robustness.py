"""Robustness checks for the Labeller A audit before anything goes in a paper."""
import pandas as pd

ASPECTS = ["scenery", "facilities", "price", "accessibility", "crowd", "atmosphere",
           "service", "food", "historical_value", "cleanliness", "safety", "photo_spot"]

ab = pd.read_csv("output/absa_llm_gpu/absa_llm_results.csv")
ib = pd.read_csv("output/absa_indobertweet/absa_indobertweet_results_v2.csv")

rows = []
for i in range(len(ab)):
    for a in ASPECTS:
        llm, ind = ab.at[i, f"{a}_sentiment"], ib.at[i, f"{a}_sentiment"]
        if isinstance(llm, str) and llm in ("positive", "neutral", "negative") \
                and isinstance(ind, str) and ind in ("positive", "neutral", "negative"):
            rows.append({"ridex": i, "aspect": a, "llm": llm, "ind": ind})
d = pd.DataFrame(rows)
n = len(d)
print(f"n = {n}\n")

# ---- tie diagnostics
g = d.groupby("ridex")["llm"]
def n_ties(s):
    vc = s.value_counts()
    return (vc == vc.max()).sum() > 1
ties = g.apply(n_ties)
print(f"reviews with a tied majority aspect label: {ties.sum()}/{len(ties)} "
      f"({ties.sum()/len(ties)*100:.1f}%)  -> tie-breaking is arbitrary there")
print(f"mentions inside tied reviews: {d.ridex.isin(ties[ties].index).sum()}\n")

# ---- binary scheme: positive vs not-positive
binmap = {"positive": "pos", "neutral": "notpos", "negative": "notpos"}
d["b_llm"] = d.llm.map(binmap)
d["b_ind"] = d.ind.map(binmap)
print("=== BINARY (positive vs not) ===")
print(f"constant 'pos'         {(d.b_llm=='pos').mean()*100:.1f}%")
print(f"A as-is                {(d.b_llm==d.b_ind).mean()*100:.1f}%")
gA = d.groupby("ridex")["b_ind"]
d["B2"] = d.ridex.map(gA.agg(lambda s: s.value_counts().idxmax()))
print(f"A collapsed            {(d.b_llm==d['B2']).mean()*100:.1f}%")
print(f"windowing adds         {((d.b_llm==d.b_ind).mean()-(d.b_llm==d['B2']).mean())*100:+.1f} pp\n")

# ---- restrict to reviews where B itself varies (the informative subset)
vary = d.groupby("ridex")["llm"].nunique()
mixed = vary[vary > 1].index
dm = d[d.ridex.isin(mixed)]
print(f"=== reviews where Labeller B varies within the review (n={len(dm)} mentions) ===")
print(f"constant 'positive'    {(dm.llm=='positive').mean()*100:.1f}%")
print(f"A as-is                {(dm.llm==dm.ind).mean()*100:.1f}%")
gA2 = dm.groupby("ridex")["ind"]
dm = dm.copy()
dm["B2"] = dm.ridex.map(gA2.agg(lambda s: s.value_counts().idxmax()))
print(f"A collapsed            {(dm.llm==dm['B2']).mean()*100:.1f}%")
print(f"windowing adds         {((dm.llm==dm.ind).mean()-(dm.llm==dm['B2']).mean())*100:+.1f} pp")

# ---- how much does A vary within a review?
spread = gA.nunique()
print(f"\nreviews where A gives identical labels to all its aspects: "
      f"{(spread==1).sum()}/{len(spread)} = {(spread==1).mean()*100:.1f}%")

# ---- how much does A vary by review at all
print(f"\nA's label distribution: {d.ind.value_counts().to_dict()}")
print(f"B's label distribution: {d.llm.value_counts().to_dict()}")

# ---- what does A do on reviews B labels mixed-polarity?
mixed_rev = d.groupby("ridex")["llm"].agg(lambda s: "positive" in set(s) and "negative" in set(s))
mr = d[d.ridex.isin(mixed_rev[mixed_rev].index)]
print(f"\n=== reviews containing BOTH B-positive and B-negative (n={len(mr)} mentions) ===")
print(f"A calls them positive: {(mr.ind=='positive').mean()*100:.1f}%")
print(f"A agrees with B here:  {(mr.llm==mr.ind).mean()*100:.1f}%")
print(f"constant 'positive':   {(mr.llm=='positive').mean()*100:.1f}%")
