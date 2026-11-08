"""Rigorous validation of the sentence splitter on the real corpus.

Four independent checks, because hand-written unit cases cannot prove a splitter
works on 1,530 informal reviews:

  A  round-trip      joining the sentences reproduces the original text
  B  fragment scan   no sentence begins with a continuation fragment
                     ('000', ',-', '000/orang', '.', ',') left over from a split
  C  dangling scan   no sentence ends with a token that implies truncation
                     ('Rp', 'Rp.', 'dll', 'dsb', 'e.g')
  D  span coverage   every sentence is a contiguous substring of the review
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
from sentence_split import split_sentences  # noqa: E402

df = pd.read_csv(ROOT / "data" / "processed" / "yogyakarta_tourism_reviews_preprocessed.csv")
texts = df.original_text.astype(str).tolist()
print(f"reviews: {len(texts)}")

all_sents = [s for t in texts for s in split_sentences(t)]
print(f"sentences: {len(all_sents)}\n")

# ---------- A round-trip ----------
# Bullet glyphs are separators, not content, so they are removed from both sides
# before comparing. Everything else must survive the split byte for byte.
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[•▪]", " ", s)).strip()
bad_rt = []
for t in texts:
    if norm(" ".join(split_sentences(t))) != norm(t):
        bad_rt.append(t)
print(f"A  round-trip failures: {len(bad_rt)}/{len(texts)}")
for t in bad_rt[:3]:
    print(f"   orig  {norm(t)[:110]!r}")
    print(f"   join  {norm(' '.join(split_sentences(t)))[:110]!r}")

# ---------- B continuity scan (pairwise) ----------
# A fragment is not "a sentence starting with a digit" (dates and list numbers
# legitimately do). It is a pair where sentence i ends mid-token and sentence
# i+1 continues it. That is the only signature that proves a bad split.
END_OPEN = re.compile(r"(\bRp\.?|\bRp|\d{1,3},?|[,(\-/]|\bdll\.?)$", re.IGNORECASE)
START_CONT = re.compile(r"^(\d{1,3}\b|,-|\)|/|\.|,)")
pairs = []
for t in texts:
    ss = split_sentences(t)
    for a, b in zip(ss, ss[1:]):
        if END_OPEN.search(a.strip()) and START_CONT.match(b.strip()):
            pairs.append((a, b))
print(f"\nB  broken pairs (sentence i left open, i+1 continues it): {len(pairs)}")
for a, b in pairs[:10]:
    print(f"   ...{a[-45:]!r}  +  {b[:45]!r}")

# ---------- C dangling scan ----------
DANGLE_END = re.compile(
    r"\b(Rp|Rp\.|dll|dsb|dst|jl|jln|No|Thn|Blm)$", re.IGNORECASE)
dang = [s for s in all_sents
        if DANGLE_END.search(s.strip()) and not s.strip().endswith(".")]
print(f"\nC  sentences ending in a truncation token: {len(dang)}")
for s in dang[:10]:
    print(f"   {s[-80:]!r}")

# ---------- D span coverage ----------
not_sub = []
for t in texts:
    for s in split_sentences(t):
        if s not in t:
            not_sub.append((t, s))
print(f"\nD  sentences that are not a substring of their review: {len(not_sub)}")
for t, s in not_sub[:5]:
    print(f"   {s[:90]!r}")
    print(f"     in {t[:90]!r}")

# ---------- price integrity ----------
print("\n--- price integrity ---")
PRICE = re.compile(r"\d{1,3}\.\d{3}")
prices_ok = prices_broken = 0
for t in texts:
    want = len(PRICE.findall(t))
    got = sum(len(PRICE.findall(s)) for s in split_sentences(t))
    prices_ok += min(want, got)
    prices_broken += max(0, want - got)
print(f"thousands-separated prices in source: {prices_ok + prices_broken}")
print(f"preserved whole: {prices_ok}   broken: {prices_broken}")

# ---------- sentence length extremes ----------
lens = pd.Series([len(s.split()) for s in all_sents])
print(f"\n--- length ---")
print(f"mean {lens.mean():.1f}  median {lens.median():.0f}  "
      f"p95 {lens.quantile(.95):.0f}  max {lens.max()}")
print(f"under 3 words: {(lens<3).sum()} ({(lens<3).mean()*100:.1f}%)")
print(f"over 80 words: {(lens>80).sum()}")
print("\nlongest sentences:")
for s in sorted(all_sents, key=lambda x: -len(x.split()))[:3]:
    print(f"   [{len(s.split())}w] {s[:130]!r}")

fail = len(bad_rt) + len(pairs) + len(dang) + len(not_sub) + prices_broken
print(f"\n{'SPLITTER OK' if fail == 0 else f'PROBLEMS: {fail}'}")
