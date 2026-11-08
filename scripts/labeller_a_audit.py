"""What is Labeller A, really, and does its aspect windowing add anything?

Code inspection (src/sentiment_analysis/absa_indobertweet.py:327-366) shows
Labeller A is NOT a review-level classifier. It:
  1. detects aspects with a keyword lexicon,
  2. cuts a +/-100 character context window around the aspect keywords,
  3. runs IndoBERTweet on THAT context.

So it emits up to 12 aspect labels per review. Any framing that calls it a
"review-level labeller" is wrong.

This script asks the question that matters instead: does the windowing add
information, or is the output just a review-level judgement redistributed across
aspect slots?
"""
import pandas as pd

ASPECTS = ["scenery", "facilities", "price", "accessibility", "crowd", "atmosphere",
           "service", "food", "historical_value", "cleanliness", "safety", "photo_spot"]

ab = pd.read_csv("output/absa_llm_gpu/absa_llm_results.csv")
ib = pd.read_csv("output/absa_indobertweet/absa_indobertweet_results_v2.csv")

rows = []
for i in range(len(ab)):
    for a in ASPECTS:
        llm = ab.at[i, f"{a}_sentiment"]
        ind = ib.at[i, f"{a}_sentiment"]
        if isinstance(llm, str) and llm in ("positive", "negative", "neutral") \
                and isinstance(ind, str) and ind in ("positive", "negative", "neutral"):
            rows.append({"ridex": i, "aspect": a, "llm": llm, "ind": ind})
d = pd.DataFrame(rows)
n = len(d)
hit = (d.llm == d.ind).sum()
print(f"comparable mentions: {n}")
print(f"Labeller A as-is            {hit/n*100:.1f}%")

# B0: constant corpus majority
maj = d.llm.value_counts().idxmax()
print(f"B0 always '{maj}'            {(d.llm == maj).mean()*100:.1f}%")

# B1: per-review majority of the TRUE aspect labels (upper bound for any
#     one-label-per-review representation)
g = d.groupby("ridex")["llm"]
per_rev = g.agg(lambda s: s.value_counts().idxmax())
d["B1"] = d.ridex.map(per_rev)
print(f"B1 per-review majority (B)  {(d.llm == d['B1']).mean()*100:.1f}%  <- representational ceiling")

# B2: per-review majority of LABELLER A's own labels, spread across its aspects.
#     This is Labeller A with its aspect windowing removed.
gA = d.groupby("ridex")["ind"]
per_revA = gA.agg(lambda s: s.value_counts().idxmax())
d["B2"] = d.ridex.map(per_revA)
print(f"B2 Labeller A collapsed     {(d.llm == d['B2']).mean()*100:.1f}%  <- windowing removed")

# B3: per-review majority of A, but only over aspects A actually labelled
print(f"\naspect windowing adds: {hit/n*100 - (d.llm == d['B2']).mean()*100:+.1f} pp "
      f"over its own review-level collapse")

# how often does A disagree with its own review-level collapse?
print(f"A deviates from its own review majority on {(d.ind != d['B2']).mean()*100:.1f}% of mentions")

print("\n=== by aspect ===")
t = d.groupby("aspect").apply(lambda x: pd.Series({
    "n": len(x),
    "A_agree": (x.llm == x.ind).mean() * 100,
    "B2_agree": (x.llm == x["B2"]).mean() * 100,
}), include_groups=False)
t["A_minus_B2"] = (t.A_agree - t.B2_agree).round(1)
print(t.round(1).to_string())
