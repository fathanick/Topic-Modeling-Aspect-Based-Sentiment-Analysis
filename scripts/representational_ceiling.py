"""The representational ceiling: what is the BEST agreement any review-level
labeller could possibly achieve against the aspect labels?

If each review must carry exactly one label, the maximum agreement with its own
aspect mentions is achieved by labelling each review with its majority aspect
polarity. That number is a hard upper bound for the whole representational class,
independent of model quality.

Compared against the observed agreement, this separates "Labeller A is a bad
model" from "no review-level labeller can do better".
"""
import pandas as pd

ASP = ["cleanliness", "facilities", "price", "scenery", "atmosphere", "accessibility",
       "crowd", "safety", "service", "food", "historical_value", "photo_spot"]

div = pd.read_csv("output/divergence/aspect_disagreement.csv")
print("comparable labels:", len(div))
print("observed disagreement:", f"{div.disagree.mean()*100:.1f}%")
print("observed agreement:   ", f"{(~div.disagree).mean()*100:.1f}%")

# For each review, among its comparable aspects, what is the majority polarity?
g = div.groupby("review_id")["llm"]
maj = g.agg(lambda s: s.value_counts().idxmax())
share = g.agg(lambda s: s.value_counts().max() / len(s))
n_aspects = g.size()

ceiling_hits = (share * n_aspects).sum()
total = n_aspects.sum()
print(f"\nmentions in the comparable set: {total}")
print(f"max hits for ANY one-label-per-review labeller: {ceiling_hits:.0f}")
print(f"CEILING = {ceiling_hits/total*100:.1f}%")
print(f"observed = {(~div.disagree).mean()*100:.1f}%")
print(f"gap to ceiling = {(ceiling_hits/total - (~div.disagree).mean())*100:.1f} pp")

# how many reviews are internally unreachable: no single label gets them all
multi = n_aspects[n_aspects >= 2]
mixed = div.groupby("review_id")["llm"].agg(lambda s: {"positive","negative"} <= set(s))
print(f"\nreviews with >=2 comparable aspects: {len(multi)}")
print(f"reviews containing both positive and negative aspects: {mixed.sum()} "
      f"({mixed.sum()/len(multi)*100:.1f}%)")
print(f"of all reviews: {mixed.sum()}/{len(n_aspects)} = {mixed.sum()/len(n_aspects)*100:.1f}%")

# perfect-model thought experiment: if B were 100% correct, what would a
# majority-labelled review-level model score?
print("\n=== interpretation ===")
print(f"Any review-level labeller, even a perfect one, tops out at "
      f"{ceiling_hits/total*100:.1f}% agreement with the aspect labels.")
print(f"Labeller A achieves {(~div.disagree).mean()*100:.1f}%.")
print(f"It is therefore within {abs(ceiling_hits/total - (~div.disagree).mean())*100:.1f} pp "
      f"of the theoretical maximum for its representational class.")
