"""Sentence segmentation quality check before building the sentence-level dataset.

Indonesian reviews are informal. A naive split on . ! ? fails on:
  - decimal prices          "Rp 10.000"      -> splits into "Rp 10" + "000"
  - abbreviations           "dll.", "No.", "Jl."
  - ellipsis                "...makan...enak"
  - missing final period
  - emoji / newline boundaries

This measures how often each failure occurs, so the splitter can be fixed on
evidence rather than guesswork.
"""
import re
from collections import Counter

import pandas as pd

df = pd.read_csv("data/processed/yogyakarta_tourism_reviews_preprocessed.csv")
texts = df.original_text.astype(str)
print(f"reviews: {len(df)}")

# --- how often do prices / decimals appear at all
thousand = texts.str.count(r"\b\d{1,3}\.\d{3}\b")
print(f"reviews containing a thousands-separated number: {(thousand>0).sum()} "
      f"({(thousand>0).mean()*100:.1f}%)  total occurrences {thousand.sum()}")

# --- abbreviation periods
ABBR = ["dll", "dsb", "dst", "jl", "no", "bln", "thn", "sdr", "a.n", "yth"]
abbr_hits = Counter()
for t in texts:
    for a in ABBR:
        for m in re.finditer(rf"\b{a}\.", t, re.IGNORECASE):
            abbr_hits[a] += 1
print("abbreviation-final periods:", dict(abbr_hits) if abbr_hits else "none")

# --- ellipsis
ellipsis_reviews = texts.str.contains(r"\.\.\.|…", regex=True).sum()
print(f"reviews with '...' or '…': {ellipsis_reviews}")

# --- newlines as de facto sentence breaks
nl = chr(10)
print(f"reviews with newlines: {texts.str.contains(nl).sum()}")

# --- naive splitter vs proposed splitter
def naive(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]

def fixed(t):
    t = re.sub(r"\b(\d{1,3})\.(\d{3})\b", r"\1<DOT>\2", t)       # protect 10.000
    t = re.sub(r"\b(dll|dsb|dst|jl|no|bln|thn|sdr|yth)\.", r"\1<DOT>", t, flags=re.I)
    t = re.sub(r"\.{2,}|…", "<ELL>", t)                           # protect ellipsis
    parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    out = []
    for p in parts:
        p = p.replace("<DOT>", ".").replace("<ELL>", "...").strip()
        if p:
            out.append(p)
    return out

n_naive = sum(len(naive(t)) for t in texts)
n_fixed = sum(len(fixed(t)) for t in texts)
print(f"\nnaive splitter  sentences: {n_naive}")
print(f"fixed splitter  sentences: {n_fixed}")
print(f"reduction from protecting prices/abbreviations/ellipsis: {n_naive - n_fixed}")

# --- show the difference on a few reviews containing prices
shown = 0
for t in texts:
    if re.search(r"\b\d{1,3}\.\d{3}\b", t) and len(t) > 120:
        a, b = naive(t), fixed(t)
        if len(a) != len(b):
            print(f"\nNAIVE ({len(a)}): {a}")
            print(f"FIXED ({len(b)}): {b}")
            shown += 1
            if shown >= 3:
                break

# --- sentence length distribution under the fixed splitter
lens = [len(s.split()) for t in texts for s in fixed(t)]
s = pd.Series(lens)
print(f"\nwords per sentence: mean {s.mean():.1f} median {s.median():.0f} "
      f"p95 {s.quantile(.95):.0f} max {s.max()}")
print(f"sentences under 3 words: {(s<3).sum()} ({(s<3).mean()*100:.1f}%)")
