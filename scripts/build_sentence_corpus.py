"""Build the sentence-level corpus: one row per sentence, with provenance.

Output: data/processed/sentences.parquet (or .csv if pyarrow is absent)
Columns: sentence_id, review_id, sent_index, sent_in_review, n_sentences,
         sentence, destination, stars

Note on the n_sentences question: sentences per review does not change between
runs, so it is computed here once rather than asked of the model.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
from sentence_split import split_sentences  # noqa: E402

IN = ROOT / "data" / "processed" / "yogyakarta_tourism_reviews_preprocessed.csv"
OUT = ROOT / "data" / "processed" / "sentences.csv"


def main():
    df = pd.read_csv(IN)
    print(f"reviews: {len(df)}")

    rows = []
    for rid, r in df.iterrows():
        sents = split_sentences(str(r["original_text"]))
        n = len(sents)
        for i, s in enumerate(sents):
            rows.append({
                "sentence_id": f"{rid}_{i:02d}",
                "review_id": rid,
                "sent_index": i,
                "n_sentences": n,
                "is_multi_sentence": n > 1,
                "sentence": s,
                "destination": r["destination"],
                "stars": r["stars"],
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print(f"sentences: {len(out)}")
    print(f"multi-sentence reviews: {df.shape[0] and (out.groupby('review_id').size() > 1).sum()}")
    print(f"single-sentence reviews: {(out.groupby('review_id').size() == 1).sum()}")
    print(f"empty sentences: {(out.sentence.str.len() == 0).sum()}")
    print("\nsentences per review:")
    print(out.groupby("review_id").size().value_counts().sort_index().to_string())
    print(f"\nwritten {OUT}")

    assert out.sentence_id.is_unique
    assert out.sentence.str.strip().ne("").all()
    # every review must survive splitting
    assert out.review_id.nunique() == len(df), (out.review_id.nunique(), len(df))
    print("\nsentence corpus built")


if __name__ == "__main__":
    main()
