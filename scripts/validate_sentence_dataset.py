"""Full validity audit of data/processed/sentences.csv before any GPU run.

Checks that the dataset is internally consistent, faithful to the source corpus,
free of the artefacts that waste model calls, and safe to label.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
from sentence_split import split_sentences  # noqa: E402

SRC = ROOT / "data" / "processed" / "yogyakarta_tourism_reviews_preprocessed.csv"
SEN = ROOT / "data" / "processed" / "sentences.csv"

src = pd.read_csv(SRC)
sen = pd.read_csv(SEN)
fails = []


def ck(label, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'} {label}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(label)


print("=" * 70)
print("1. SHAPE AND RECONCILIATION")
print("=" * 70)
ck("source reviews", len(src) == 1530, str(len(src)))
ck("sentences", len(sen) == 6552, str(len(sen)))
ck("every review present", sen.review_id.nunique() == len(src),
   f"{sen.review_id.nunique()} vs {len(src)}")
ck("sentence_id unique", sen.sentence_id.is_unique)
ck("no nulls anywhere", sen.notna().all().all())
ck("columns", list(sen.columns) == [
    "sentence_id", "review_id", "sent_index", "n_sentences", "is_multi_sentence",
    "sentence", "destination", "stars"], str(list(sen.columns)))

print("\n" + "=" * 70)
print("2. FIDELITY TO SOURCE (the splitter contract)")
print("=" * 70)
# re-split every review and compare to the stored rows exactly
mismatch = []
for rid, group in sen.groupby("review_id"):
    expected = split_sentences(str(src.at[rid, "original_text"]))
    stored = group.sort_values("sent_index").sentence.tolist()
    if expected != stored:
        mismatch.append(rid)
ck("stored sentences == fresh split", len(mismatch) == 0, f"{len(mismatch)} mismatched")

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[•▪]", " ", s)).strip()
bad_join = [rid for rid, g in sen.groupby("review_id")
            if norm(" ".join(g.sort_values("sent_index").sentence)) !=
            norm(str(src.at[rid, "original_text"]))]
ck("rejoin reproduces review text", len(bad_join) == 0, f"{len(bad_join)} failures")

sub = sum(1 for _, r in sen.iterrows()
          if r.sentence not in str(src.at[r.review_id, "original_text"]))
ck("every sentence is a substring of its review", sub == 0, f"{sub} violations")

print("\n" + "=" * 70)
print("3. ORDERING AND METADATA")
print("=" * 70)
order_ok = all(list(g.sort_values("sent_index").sent_index) == list(range(len(g)))
               for _, g in sen.groupby("review_id"))
ck("sent_index is 0..n-1 per review", order_ok)
n_ok = all((g.n_sentences == len(g)).all() and (g.is_multi_sentence == (len(g) > 1)).all()
           for _, g in sen.groupby("review_id"))
ck("n_sentences and is_multi_sentence correct", n_ok)

dest_ok = all((g.destination == src.at[rid, "destination"]).all()
              for rid, g in sen.groupby("review_id"))
star_ok = all((g.stars == src.at[rid, "stars"]).all()
              for rid, g in sen.groupby("review_id"))
ck("destination matches source", dest_ok)
ck("stars matches source", star_ok)

print("\n" + "=" * 70)
print("4. WHAT THE MODEL WILL BE ASKED TO LABEL")
print("=" * 70)
lengths = sen.sentence.str.split().str.len()
print(f"     words/sentence: mean {lengths.mean():.1f}  median {lengths.median():.0f}  "
      f"p95 {lengths.quantile(.95):.0f}  max {lengths.max()}")
print(f"     chars/sentence: mean {sen.sentence.str.len().mean():.0f}  "
      f"max {sen.sentence.str.len().max()}")

# sentences with no letters at all cannot carry an aspect; the runner skips them
no_letters = sen[~sen.sentence.str.contains(r"[A-Za-zÀ-ÿ]", regex=True)]
print(f"     sentences with no letters: {len(no_letters)} "
      f"({len(no_letters)/len(sen)*100:.1f}%) -> marked unlabelable, skipped by runner")
print("     composition: " + ", ".join(
    f"{k!r}x{v}" for k, v in no_letters.sentence.value_counts().head(5).items()))
ck("skipped share is small", len(no_letters) / len(sen) < 0.02,
   f"{len(no_letters)/len(sen)*100:.1f}%")

very_short = sen[lengths <= 2]
print(f"     sentences of <=2 words: {len(very_short)} ({len(very_short)/len(sen)*100:.1f}%)")
print("     examples: " + " | ".join(repr(s) for s in very_short.sentence.head(6)))

very_long = sen[lengths > 120]
print(f"     sentences >120 words: {len(very_long)}")
for s in very_long.sentence.head(2):
    print(f"        [{len(s.split())}w] {s[:110]}...")

print("\n" + "=" * 70)
print("5. ENCODING AND TEXT HYGIENE")
print("=" * 70)
moji = sen[sen.sentence.str.contains(r"[ÃÂ�]", regex=True)]
ck("no mojibake markers", len(moji) == 0, f"{len(moji)}")
nl = sen[sen.sentence.str.contains("\n", regex=False)]
ck("no embedded newlines (CSV integrity)", len(nl) == 0, f"{len(nl)}")
lead = sen[sen.sentence.str.contains(r"^\s|\s$", regex=True)]
ck("no leading/trailing whitespace", len(lead) == 0, f"{len(lead)}")

emoji = sen.sentence.str.contains(r"[\U0001F300-\U0001FAFF]", regex=True)
print(f"     sentences containing emoji: {emoji.sum()} (preserved, not stripped)")

print("\n" + "=" * 70)
print("6. DUPLICATES")
print("=" * 70)
dup_text = sen.sentence.duplicated().sum()
print(f"     exact duplicate sentence strings: {dup_text}")
ck("sentence_id unique despite text duplicates", sen.sentence_id.is_unique)
dup_src = src.original_text.duplicated().sum()
print(f"     duplicate review texts in source: {dup_src}")

print("\n" + "=" * 70)
print("7. COVERAGE OF KNOWN ASPECT EVIDENCE")
print("=" * 70)
# Every evidence span the previous LLM run produced should now sit inside
# exactly one sentence. If not, the sentence unit misses where aspects live.
# The previous run fed the model text truncated to 500 chars, so evidence taken
# from the tail of a long review is absent from that field but present in the
# source. Those are counted separately, not treated as splitter failures.
ab = pd.read_csv(ROOT / "output" / "absa_llm_gpu" / "absa_llm_results.csv")
ASPECTS = ["cleanliness", "facilities", "price", "scenery", "atmosphere", "accessibility",
           "crowd", "safety", "service", "food", "historical_value", "photo_spot"]
by_rev = {rid: g.sort_values("sent_index").sentence.tolist()
          for rid, g in sen.groupby("review_id")}
inside = outside = truncated = 0
for _, r in ab.iterrows():
    for a in ASPECTS:
        ev = r.get(f"{a}_evidence")
        if not isinstance(ev, str) or not ev.strip():
            continue
        ev = ev.strip()
        full = str(src.at[r.review_id, "original_text"])
        if ev not in full:
            truncated += 1
            continue
        if any(ev in s for s in by_rev.get(r.review_id, [])):
            inside += 1
        else:
            outside += 1
print(f"     evidence spans found in a single sentence:   {inside}")
print(f"     evidence spans crossing a sentence boundary: {outside}")
print(f"     evidence not in full source text at all:     {truncated}")
tot = inside + outside
ck("most evidence sits inside one sentence", outside / max(tot, 1) < 0.05,
   f"{outside}/{tot} cross a boundary ({outside/max(tot,1)*100:.2f}%)")

print("\n" + "=" * 70)
print("8. DECISION-COST SUMMARY")
print("=" * 70)
print(f"     model calls required: {len(sen)}")
print(f"     of which <=2 words:   {len(very_short)}  ({len(very_short)/len(sen)*100:.1f}%)")
print(f"     sentences per review: mean {len(sen)/len(src):.1f}")

print("\n" + "=" * 70)
if fails:
    print(f"FAILED: {fails}")
    raise SystemExit(1)
print("DATASET VALID")
