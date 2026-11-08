"""Re-parse the stored model responses with the aspect alias resolver.

The GPU run is not repeated: `raw` already holds every model response, so the
recovery is offline and deterministic. Only aspect keys change; grounding,
sentiment validation and dedupe are the same code paths as the Kaggle run.

Usage:
    python scripts/reparse_with_aliases.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sentiment_analysis"))

from sentence_absa import dedupe_aspects, parse_response  # noqa: E402
from remap_aspects import alias_coverage, canonical  # noqa: E402


def main():
    src = ROOT / "output/sentence_absa/sentence_absa_raw.csv"
    baseline = ROOT / "output/sentence_absa/sentence_absa_raw_noremap.csv"
    df = pd.read_csv(src)
    print(f"rows: {len(df)}")

    # Permanent copy of the un-resolved parse. /tmp is wiped by the OS, and an
    # audit that depends on /tmp silently stops working weeks later.
    if not baseline.exists():
        base = df.copy()
        for i, r in df.iterrows():
            cut = isinstance(r.raw, str) and len(r.raw) >= 600
            if cut or not isinstance(r.raw, str) or not r.raw.strip():
                continue
            out = parse_response(r.raw, r.sentence)          # no resolver
            kept = dedupe_aspects(out["aspects"])
            base.at[i, "aspects"] = json.dumps(kept, ensure_ascii=False)
            base.at[i, "n_aspects"] = len(kept)
            base.at[i, "status"] = out["status"]
            base.at[i, "error"] = out["error"]
        base.to_csv(baseline, index=False)
        print(f"baseline (no resolver) written: {baseline.name} "
              f"mentions={int(base.n_aspects.sum())}")


    hit, miss = alias_coverage(df.loc[df.error.notna(), "error"])
    print(f"alias-rescuable mentions: {sum(hit.values())} across {len(hit)} keys")
    print(f"still-unmapped mentions:  {sum(miss.values())} across {len(miss)} keys")

    truncated = int((df.raw.fillna("").str.len() == 600).sum())
    print(f"raw truncated at 600 chars: {truncated} rows (left untouched)")

    before_mentions = int(df.n_aspects.sum())

    aspects, n_aspects, status, error = [], [], [], []
    for _, r in df.iterrows():
        # The runner stores only raw[:600]. Re-parsing a cut-off response drops
        # the aspects that fell past the cut, so those rows keep what the
        # Kaggle run already parsed from the full response.
        cut = isinstance(r.raw, str) and len(r.raw) >= 600
        if cut or not isinstance(r.raw, str) or not r.raw.strip():
            aspects.append(r.aspects)
            n_aspects.append(r.n_aspects)
            status.append(r.status)
            error.append(r.error)
            continue
        out = parse_response(r.raw, r.sentence, resolver=canonical)
        kept = dedupe_aspects(out["aspects"])
        aspects.append(json.dumps(kept, ensure_ascii=False))
        n_aspects.append(len(kept))
        status.append(out["status"])
        error.append(out["error"])

    df["aspects"] = aspects
    df["n_aspects"] = n_aspects
    df["status"] = status
    df["error"] = error
    df.to_csv(src, index=False)

    after_mentions = int(df.n_aspects.sum())
    print(f"\nmentions {before_mentions} -> {after_mentions} "
          f"({after_mentions - before_mentions:+d})")
    ok = int((df.status == "ok").sum())
    print(f"status ok: {ok}/{len(df)} ({ok/len(df)*100:.1f}%)")
    print(df.status.value_counts().to_string())
    print(f"sentences with >=1 aspect: {(df.n_aspects > 0).sum()}")

    bad = 0
    for _, r in df.iterrows():
        for a in json.loads(r.aspects or "[]"):
            if a["keywords"] not in r.sentence or a["span"] not in r.sentence:
                bad += 1
    print(f"grounding violations: {bad}")
    assert bad == 0, "grounding broken by the remap"


if __name__ == "__main__":
    main()
