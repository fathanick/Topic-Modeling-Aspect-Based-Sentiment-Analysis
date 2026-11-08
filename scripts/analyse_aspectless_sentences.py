"""Label the 1.309 aspect-less sentences, and test whether they need a label.

The question: a sentence with no aspect from the 12-item glossary — is its
sentiment "attached to the other aspects of the same review", or is there
nothing to attach?

The answer depends on the UNIT. This study's unit is the sentence-aspect
mention, so a sentence carrying no aspect contributes no mention and its
polarity has nowhere to live in the ABSA table. The review-level table is a
different, coarser unit and already handles it. Checked here against the data
rather than asserted.

Run: ./venv/bin/python scripts/analyse_aspectless_sentences.py
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_unlabelled_sentences import classify  # noqa: E402

raw = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw.csv")
z = raw[raw.n_aspects == 0].copy()
z["cls"] = z.sentence.apply(classify)

per_rev = raw.groupby("review_id").n_aspects.sum()
rev = raw.groupby("review_id").agg(nsent=("sentence", "size"),
                                   tot=("n_aspects", "sum"),
                                   stars=("stars", "first"))
z["rev_total"] = z.review_id.map(per_rev)
z["rev_nsent"] = z.review_id.map(rev.nsent)

# ---- 1. where do these sentences live?
in_asp_review = int((z.rev_total > 0).sum())
solo = z[z.rev_total == 0]
print("=== 1. konteks kalimat tanpa aspek ===")
print(f"total kalimat tanpa aspek          {len(z)}")
print(f"  di dalam review yang punya aspek {in_asp_review} "
      f"({in_asp_review/len(z)*100:.1f}%)")
print(f"  di review tanpa aspek sama sekali{solo.shape[0]:>4} "
      f"({solo.shape[0]/len(z)*100:.1f}%)")
print(f"review beraspek, kalimatnya 1 saja{sorted(rev.nsent[rev.tot>0].unique())[:1]}"
      " -> every sentence in an aspect-bearing review is judged independently")

# ---- 2. the 41 aspect-less reviews
zero_rev = rev[rev.tot == 0]
print("\n=== 2. review tanpa satu pun aspek ===")
print(f"review                {len(zero_rev)} / {len(rev)} "
      f"({len(zero_rev)/len(rev)*100:.1f}%)")
print(f"kalimat di dalamnya   {solo.shape[0]}")
print("sebaran bintang:")
for s, n in zero_rev.stars.value_counts().sort_index().items():
    print(f"  {s} bintang  {n}")

# ---- 3. polarity the model could express for these sentences
# The prompt only allows sentiment inside an aspect object; there is no
# top-level sentiment field. Confirm that structurally, not by memory.
sys.path.insert(0, str(ROOT / "src" / "sentiment_analysis"))
from sentence_absa import SYSTEM_PROMPT  # noqa: E402

has_toplevel = bool(re.search(r'"sentiment"\s*:\s*"<', SYSTEM_PROMPT)) and \
    SYSTEM_PROMPT.count('"sentiment"') == 1
print("\n=== 3. skema prompt ===")
print(f'"sentiment" hanya di dalam objek aspect: {not has_toplevel}')
print(f'jumlah kemunculan "sentiment" di prompt: {SYSTEM_PROMPT.count(chr(34)+"sentiment"+chr(34))}')

# ---- 4. do these sentences ever sit alone in a review?
print("\n=== 4. bisakah dipisah jadi entri tersendiri? ===")
print(f"kalimat tanpa aspek yang reviewnya HANYA kalimat itu: "
      f"{int((z.rev_nsent == 1).sum())}")

# ---- 5. what the star rating already carries
print("\n=== 5. informasi yang sudah tersedia di level review ===")
print(f"review dengan rating bintang: {rev.stars.notna().sum()} / {len(rev)}")

# ---- deliverable
z["resolution"] = "no_aspect_no_mention"
z.loc[z.rev_total == 0, "resolution"] = "review_has_no_aspect_either"

# Why the sentence carries no aspect decides whether its polarity survives.
# When the model DID emit a pair and only the aspect key fell outside the
# glossary, the sentiment is recorded in `raw` and is kept in a separate table.
# When the model returned an empty list there is no stored polarity at all.
def _emit(s):
    try:
        return json.loads(s).get("aspects", []) or []
    except Exception:
        return []


ems = z.raw.fillna("").apply(_emit)
z["reason"] = ["rejected_aspect_outside_glossary" if n else "model_returned_empty_list"
               for n in ems.apply(len)]

out = z[["sentence_id", "review_id", "destination", "stars", "rev_total",
         "rev_nsent", "cls", "resolution", "reason", "sentence"]]
dst = ROOT / "output/audit/aspectless_sentences.csv"
out.to_csv(dst, index=False)
print(f"\nwritten {dst} ({len(out)} rows)")
print(out.resolution.value_counts().to_string())
print()
print(out.reason.value_counts().to_string())
