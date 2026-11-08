import re

import pandas as pd

km = pd.read_csv("output/sentence_absa/kalimat_aspek_sentimen.csv")
raw = pd.read_csv("output/sentence_absa/sentence_absa_raw.csv")
s = pd.read_csv("data/processed/sentences.csv")


def norm(t):
    return re.sub(r"\s+", " ", str(t).lower()).strip()


km["ok"] = [norm(k) in norm(t) for k, t in zip(km.keywords, km.sentence)]
print(f"mention {len(km)} | keywords substring persis {km.ok.sum()} ({100*km.ok.mean():.1f}%)")

print("reviews in raw:", raw.review_id.nunique(), "| rows:", len(raw))
print("sentences.csv reviews:", s.review_id.nunique(), "rows:", len(s))
a, b = set(raw.review_id), set(s.review_id)
print("raw not in sentences:", len(a - b))
print("sentences not in raw:", len(b - a), sorted(b - a)[:5])
print("sentence_id covered:", km.sentence_id.isin(raw.sentence_id).all())
