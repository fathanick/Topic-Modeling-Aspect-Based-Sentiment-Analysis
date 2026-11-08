"""Chunked, resumable ABSA runner for Kaggle (or any long-lived box).

Why this exists: Groq's free tier caps at 200k tokens/day, but a full run needs
~790k. Kaggle gives 12h of wall clock, which is enough for the *time* but not for
the *quota*. So: run as much as the quota allows, checkpoint every review, stop
cleanly, and resume on the next session.

Usage on Kaggle:
    !python scripts/kaggle_runner.py --max-hours 10
then re-run the same cell tomorrow; finished reviews are never redone.

Usage locally:
    GROQ_API_KEY=... python scripts/kaggle_runner.py --max-hours 10
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODULE = ROOT / "src" / "sentiment_analysis" / "absa_llm_groq.py"


def load_absa_module():
    """Import the ABSA module without executing its main()."""
    spec = importlib.util.spec_from_file_location("absa", MODULE)
    module = importlib.util.module_from_spec(spec)
    src = MODULE.read_text().split('if __name__ == "__main__":')[0]
    exec(compile(src, str(MODULE), "exec"), module.__dict__)
    return module


def get_api_key(cli_key=None):
    """Key resolution order: CLI, Kaggle secret, environment."""
    if cli_key:
        return cli_key
    try:
        from kaggle_secrets import UserSecretClient
        return UserSecretClient().get_secret("GROQ_API_KEY")
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=10.0,
                    help="stop before the Kaggle session is killed (default 10)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N reviews (for a smoke test)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--reasoning", default=None, choices=["low", "medium", "high", "none"])
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="output dir; default output/absa_llm_groq. Use a timestamped "
                         "dir (e.g. /kaggle/working/absa_run1) to avoid clobbering")
    args = ap.parse_args()

    key = get_api_key(args.api_key)
    if not key:
        sys.exit("No API key. Set Kaggle secret GROQ_API_KEY or the env var.")

    absa = load_absa_module()
    if args.model:
        absa.MODEL_NAME = args.model
    if args.reasoning:
        absa.REASONING_EFFORT = None if args.reasoning == "none" else args.reasoning

    from groq import Groq
    client = Groq(api_key=key)

    df = absa.load_data(str(ROOT / absa.INPUT_FILE))
    if df is None:
        sys.exit("Input CSV not found - upload data/processed/ first.")
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)

    tracker = absa.ABSAResults()
    tracker.model_name = absa.MODEL_NAME
    tracker.total_reviews = len(df)
    tracker.budget = absa.TokenBudget(absa.TPM_BUDGET)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / absa.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / absa.CHECKPOINT_FILE

    done_before = len(absa.load_checkpoint(str(ckpt)))
    print(f"model={absa.MODEL_NAME} reasoning={absa.REASONING_EFFORT} "
          f"budget={absa.TPM_BUDGET} tpm")
    print(f"reviews={len(df)} already_done={done_before}")
    deadline = time.time() + args.max_hours * 3600
    start = time.time()
    exhausted = False

    while time.time() < deadline:
        done = absa.load_checkpoint(str(ckpt))
        # review_id is positional and matches run_absa_analysis()
        pending = []
        for i, row in df.iterrows():
            if i not in done:
                rec = row.to_dict()
                rec["review_id"] = i
                pending.append(rec)

        if not pending:
            print("all reviews complete")
            break

        batch = pending[:absa.BATCH_SIZE]
        try:
            results = absa.analyze_batch(client, batch, tracker)
        except absa.DailyQuotaExhausted as e:
            print(f"\nDAILY QUOTA EXHAUSTED: {e}")
            exhausted = True
            break

        for rec in batch:
            analysis = (results or {}).get(rec["review_id"])
            record = {"review_id": rec["review_id"],
                      "destination": rec["destination"],
                      "username": rec.get("username", ""),
                      "stars": rec.get("stars", ""),
                      "original_text": str(rec.get("original_text", ""))[:500],
                      "analysis_status": "success" if analysis else "failed"}
            seen = {}
            for a in (analysis or {}).get("aspects", []) or []:
                key_ = str(a.get("aspect", "")).lower().replace(" ", "_")
                if key_ not in absa.TOURISM_ASPECTS or key_ in seen:
                    continue
                sent = str(a.get("sentiment", "neutral")).lower()
                seen[key_] = sent if sent in ("positive", "negative", "neutral") else "neutral"
                record[f"{key_}_sentiment"] = seen[key_]
                record[f"{key_}_evidence"] = str(a.get("evidence", ""))[:100]
            for key_ in absa.TOURISM_ASPECTS:
                record.setdefault(f"{key_}_sentiment", "")
                record.setdefault(f"{key_}_evidence", "")
            record["num_aspects"] = len(seen)
            record["aspect_sentiments"] = "; ".join(f"{k}:{v}" for k, v in seen.items())
            record["overall_sentiment"] = (analysis or {}).get("overall_sentiment", "neutral")
            absa.append_checkpoint(str(ckpt), record)

        done_now = len(absa.load_checkpoint(str(ckpt)))
        elapsed = (time.time() - start) / 60
        print(f"  {done_now}/{len(df)} reviews | {elapsed:.1f} min | "
              f"tokens {tracker.total_tokens:,} | deadline in "
              f"{(deadline - time.time())/60:.0f} min")
        time.sleep(absa.RATE_LIMIT_DELAY)

    done = absa.load_checkpoint(str(ckpt))
    print(f"\ncheckpoint holds {len(done)}/{len(df)} reviews -> {ckpt}")
    if exhausted:
        print("Re-run this cell tomorrow to continue; nothing is repeated.")
        sys.exit(2)
    if len(done) >= len(df):
        import pandas as pd
        final = pd.DataFrame(list(done.values())).sort_values("review_id")
        final.to_csv(out_dir / "absa_llm_results.csv", index=False)
        print(f"wrote {out_dir/'absa_llm_results.csv'}")


if __name__ == "__main__":
    main()
