"""Rerun a sample of failed reviews and print the RAW model output.

The main runner only stores parsed fields, so a parse failure leaves no trace of
what the model actually emitted. This prints it, so the fix is chosen from
evidence rather than guesswork.

Usage (Kaggle GPU):
    !python scripts/diagnose_failed.py --n 24
"""
import argparse
import json

import pandas as pd

from absa_llm_local_gpu import (INPUT_FILE, build_messages, generate, load_model,
                                parse_output)

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="absa_gpu/absa_checkpoint.jsonl")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--max-new-tokens", type=int, default=256)
args = ap.parse_args()

recs = [json.loads(l) for l in open(args.checkpoint) if l.strip()]
failed = [r for r in recs if r["analysis_status"] == "failed"]
print(f"failed total: {len(failed)} / {len(recs)}")

df = pd.read_csv(INPUT_FILE)
print(f"max_new_tokens: {args.max_new_tokens}\n")

model, tok = load_model(args.model, four_bit=True)

for r in failed[: args.n]:
    i = r["review_id"]
    row = df.iloc[i]
    msgs = [build_messages(row["original_text"], row.get("stars", "N/A"))]
    raw = generate(model, tok, msgs, args.max_new_tokens)[0]
    parsed = parse_output(raw, row["original_text"])

    print("=" * 70)
    print(f"id={i} len={len(str(row['original_text']))} parsed={'OK' if parsed else 'FAIL'}")
    print("--- RAW ---")
    print(repr(raw[-700:]))
    print()
