"""Sentence-level co-occurrence: can one sentence carry two aspects with
different polarities? This is what a span-level unit would need to represent.
"""
import re

import pandas as pd

ASP = ["cleanliness", "facilities", "price", "scenery", "atmosphere", "accessibility",
       "crowd", "safety", "service", "food", "historical_value", "photo_spot"]

ab = pd.read_csv("output/absa_llm_gpu/absa_llm_results.csv")


def sentences(text):
    """Return (start, end) offsets per sentence, keeping it simple and verifiable."""
    out = []
    for m in re.finditer(r"[^.!?\n]+[.!?]*", text):
        s, e = m.start(), m.end()
        if text[s:e].strip():
            out.append((s, e))
    return out


def locate(text, ev):
    i = text.find(ev)
    return (i, i + len(ev)) if i >= 0 else None


stats = {"reviews": 0, "sent_with_2plus_aspect": 0, "sent_with_opposite": 0,
         "sent_total": 0, "spans": 0, "spans_unlocatable": 0}
examples = []

for _, r in ab.iterrows():
    text = str(r.original_text)
    sents = sentences(text)
    if not sents:
        continue
    stats["reviews"] += 1
    per_sent = {i: [] for i in range(len(sents))}
    for a in ASP:
        lab = r.get(f"{a}_sentiment")
        ev = r.get(f"{a}_evidence")
        if not isinstance(lab, str) or not lab.strip():
            continue
        if not isinstance(ev, str) or not ev.strip():
            continue
        stats["spans"] += 1
        loc = locate(text, ev.strip())
        if loc is None:
            stats["spans_unlocatable"] += 1
            continue
        for i, (s, e) in enumerate(sents):
            if loc[0] >= s and loc[1] <= e + 1:
                per_sent[i].append((a, lab.strip(), ev.strip()))
                break

    for i, items in per_sent.items():
        if len({a for a, _, _ in items}) < 2:
            continue
        stats["sent_with_2plus_aspect"] += 1
        labs = {l for _, l, _ in items}
        if "positive" in labs and "negative" in labs:
            stats["sent_with_opposite"] += 1
            if len(examples) < 8:
                examples.append((r.review_id, sents[i], items))

for i in range(len(ab)):
    pass

print("=== sentence level ===")
print(f"reviews analysed                 {stats['reviews']}")
print(f"spans located in a sentence      {stats['spans'] - stats['spans_unlocatable']}/{stats['spans']}")
print(f"sentences with 2+ aspects        {stats['sent_with_2plus_aspect']}")
print(f"sentences with OPPOSITE polarity {stats['sent_with_opposite']}")
if stats["sent_with_2plus_aspect"]:
    print(f"  share opposite among multi-aspect sentences: "
          f"{stats['sent_with_opposite']/stats['sent_with_2plus_aspect']*100:.1f}%")

print("\n=== examples: one sentence, two aspects, opposite polarity ===")
for rid, (s, e), items in examples:
    txt = ab.loc[ab.review_id == rid, "original_text"].iloc[0]
    print(f"\nreview {rid}: {txt[s:e].strip()[:200]}")
    for a, l, ev in items:
        print(f"   {a:18} {l:9} | {ev[:70]}")
