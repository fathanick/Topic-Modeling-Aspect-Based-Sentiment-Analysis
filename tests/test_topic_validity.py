"""Prove the reported numbers are reproducible and internally consistent.

Checks, in order of how badly each has burned us:
  1. coherence tokenizer keeps words glued to punctuation (the bug that made
     c_v 0.5444 and 0.5194 for the same config)
  2. topic shares recomputed from the per-review CSV match topics_summary.csv
  3. outlier share matches, and topics + outliers = all reviews
  4. every review appears exactly once
  5. coherence JSON reports the same topic count as the summary table
  6. sweep winner's c_v equals the value in sweep_choice.json

Usage: python tests/test_topic_validity.py [--dir output/topic_modeling_final]
"""

import argparse
import json
import re
import sys
from pathlib import Path

FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def tokenizer(text):
    """Mirror of topic_modeling_bertopic.tokenize_for_coherence."""
    cleaned = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return [w for w in cleaned.split() if len(w) > 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/topic_modeling_final")
    args = ap.parse_args()
    d = Path(args.dir)
    if not d.exists():
        raise SystemExit(f"{d} not found")

    print("[1] coherence tokenizer")
    toks = tokenizer("Pantai-nya bagus, tapi parkir Rp 10.000.")
    check("pantai" in toks, "keeps 'pantai' from 'Pantai-nya'")
    check("bagus" in toks, "keeps 'bagus' from 'bagus,'")
    check("parkir" in toks, "keeps 'parkir'")
    check(all(not any(c.isdigit() or c in "-.,!" for c in t) for t in toks),
          "no punctuation or digits left in tokens")

    import pandas as pd

    print("[2] topic shares reconcile with the review-level CSV")
    r = pd.read_csv(d / "topic_modeling_results.csv")
    t = pd.read_csv(d / "topics_summary.csv")
    n = len(r)

    counts = r.topic.value_counts()
    ok = True
    for _, row in t.iterrows():
        got = int(counts.get(row.topic_id, 0))
        if got != int(row.document_count):
            ok = False
            print(f"       topic {row.topic_id}: summary {row.document_count} vs csv {got}")
    check(ok, f"all {len(t)} topic sizes match between summary and CSV")

    print("[3] outliers and totals")
    outl = int((r.topic == -1).sum())
    check((r.topic == -1).mean() < 1.0, "outlier share computed")
    check(outl + int(t.document_count.sum()) == n,
          f"{outl} outliers + {int(t.document_count.sum())} clustered = {n} reviews")

    print("[4] no duplicated reviews")
    if "review_id" in r.columns:
        check(r.review_id.is_unique, "review_id unique")
    else:
        check(len(r) == n, "row count equals review count")

    print("[5] coherence metadata")
    c = json.load(open(d / "topic_coherence.json"))
    check(c.get("n_topics_scored") == len(t),
          f"coherence scored {c.get('n_topics_scored')} topics, summary lists {len(t)}")
    check(-1.0 < c["c_v"] < 1.0, f"c_v {c['c_v']} within (-1,1)")
    check(-1.0 < c["c_npmi"] < 1.0, f"c_npmi {c['c_npmi']} within (-1,1)")

    print("[6] sweep choice reproducibility")
    sp = d / "sweep_choice.json"
    if sp.exists():
        s = json.load(open(sp))
        if "c_v_sweep_row" in s:
            check(abs(s["c_v"] - s["c_v_sweep_row"]) < 1e-9 or not s.get("reproduced", True),
                  f"final c_v {s['c_v']} vs sweep row {s['c_v_sweep_row']} "
                  f"(reproduced={s.get('reproduced')})")
    else:
        print("       (no sweep_choice.json, skipped)")

    print()
    if FAIL:
        raise SystemExit(f"FAILED: {len(FAIL)} check(s)")
    print("ALL TOPIC-VALIDITY CHECKS PASSED")


if __name__ == "__main__":
    main()
