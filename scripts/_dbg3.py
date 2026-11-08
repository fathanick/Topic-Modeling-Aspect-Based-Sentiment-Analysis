import re
import sys
sys.path.insert(0, "src/preprocessing")
import pandas as pd
from sentence_split import split_sentences, _DOT_RUN, _is_boundary

df = pd.read_csv("data/processed/yogyakarta_tourism_reviews_preprocessed.csv")
texts = df.original_text.astype(str).tolist()
longest = max((s for t in texts for s in split_sentences(t)), key=lambda s: len(s.split()))
print(f"longest sentence: {len(longest.split())} words\n")

# find it in its review and show each punctuation run + decision
for t in texts:
    if longest[:60] in t:
        print("punct runs in the review and the decision taken:")
        for m in _DOT_RUN.finditer(t):
            d = _is_boundary(t, m.start(), m.end())
            ctx = t[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
            print(f"  {m.group()!r:8} boundary={d!s:5} | ...{ctx}...")
        break
