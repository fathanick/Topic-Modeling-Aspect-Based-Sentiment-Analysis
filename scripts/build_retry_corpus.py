"""Sentences the model passed over despite carrying an evaluative signal.

These 238 are where the model returned an empty aspect list for a sentence that
has an evaluative cue or an aspect keyword. Re-running only these is cheap; a
re-run that also rewrites the 5.170 already-good sentences would throw away
work that has been audited.

Run: ./venv/bin/python scripts/build_retry_corpus.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT / "src" / "sentiment_analysis"


def main():
    import sys
    sys.path.insert(0, str(sys_path))
    from audit_unlabelled_sentences import classify  # noqa: E402

    raw = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw.csv")
    sent = pd.read_csv(ROOT / "data/processed/sentences.csv")

    z = raw[raw.n_aspects == 0].copy()
    z["cls"] = z.sentence.apply(classify)
    retry = z[z.cls.isin(["D evaluative cue", "E aspect keyword"])].copy()
    print(f"zero-aspect sentences : {len(z)}")
    print(f"retry candidates      : {len(retry)}")
    print(retry.cls.value_counts().to_string())
    print("\nstatus:")
    print(retry.status.value_counts().to_string())

    # Only the ones the model itself skipped. The 64 parser rejections already
    # went through the tolerant matcher and are final.
    ok = retry[retry.status == "ok"]
    print(f"\nmodel-skipped (status ok): {len(ok)}")

    out = sent[sent.sentence_id.isin(ok.sentence_id)].copy()
    assert len(out) == len(ok), f"{len(out)} vs {len(ok)}"
    assert out.sentence_id.is_unique
    dst = ROOT / "data/processed/sentences_retry.csv"
    out.to_csv(dst, index=False)
    print(f"\nwritten {dst} ({len(out)} sentences)")

    # keep the destination and review coverage intact for the merge back
    print(f"destinations covered: {out.destination.nunique()}")
    print(f"reviews covered     : {out.review_id.nunique()}")


if __name__ == "__main__":
    main()
