"""Are the sentences with zero aspects genuinely non-opinionated?

A skipped sentence is only acceptable if it carries no evaluative content the
12-aspect schema could hold. This script classifies them by evidence rather
than by assertion:

  A  no letters / no words        -> unlabelable by construction (73, excluded)
  B  pure question or greeting    -> no evaluation
  C  purely factual/descriptive   -> no evaluative cue
  D  carries an evaluative cue    -> SUSPECT, the model may have missed it
  E  carries an aspect keyword    -> SUSPECT, strong signal missed

Any D/E sentence is a real gap and is written out for inspection.

Run: ./venv/bin/python scripts/audit_unlabelled_sentences.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Evaluative cues in Indonesian, including the informal spellings this corpus
# uses. Kept deliberately broad: a false SUSPECT costs one manual look, a false
# OK hides a missed opinion.
CUE = re.compile(
    r"\b(bagus|baguss|bagusss|jelek|buruk|indah|keren|mantap|memuaskan|mengecewakan|"
    r"nyaman|tidak nyaman|kurang nyaman|bersih|kotor|mahal|murah|terjangkau|"
    r"ramah|tidak ramah|kasar|cepat|lambat|lama|antri|antre|penuh|sesak|"
    r"sepi|ramai|tenang|berisik|bising|aman|bahaya|berbahaya|rawan|"
    r"sulit|mudah|gampang|susah|repot|jauh|dekat|luas|sempit|"
    r"enak|lezat|tidak enak|hambar|asri|adem|panas|gerah|bau|"
    r"recommended|rekomen|rekomendasi|sayang|kecewa|puas|suka|tidak suka|"
    r"mantul|oke|ok|baguss|parah|payah|mending|lebih baik|worst|best|"
    r"worth|overpriced|nyesel|menyesal|gak worth|sayang sekali)\b",
    re.IGNORECASE)

ASPECT_KW = re.compile(
    r"\b(toilet|wc|kamar mandi|mushola|mushalla|musala|parkir|gazebo|warung|"
    r"penginapan|hotel|harga|tiket|tarif|biaya|parkir|petugas|pemandu|guide|"
    r"pelayanan|jalan|akses|tanjakan|macet|transportasi|pemandangan|view|"
    r"panorama|laut|pasir|sunset|suasana|kuliner|makanan|minuman|seafood|"
    r"ombak|arus|jurang|rambu|pengunjung|antrean|kepadatan|foto|selfie|"
    r"candi|relief|sejarah|budaya|sampah|kebersihan)\b",
    re.IGNORECASE)

QUESTION = re.compile(r"^\s*(apa|apakah|bagaimana|gimana|berapa|kapan|siapa|"
                      r"dimana|di mana|mengapa|kenapa|adakah|bisakah|bolehkah)\b",
                      re.I)


def classify(text: str) -> str:
    t = str(text).strip()
    if not re.search(r"[A-Za-z]", t):
        return "A no letters"
    words = re.findall(r"[A-Za-z]{2,}", t)
    if len(words) < 3:
        return "A too short"
    has_cue = bool(CUE.search(t))
    has_kw = bool(ASPECT_KW.search(t))
    if has_kw:
        return "E aspect keyword"
    if has_cue:
        return "D evaluative cue"
    if t.endswith("?") or QUESTION.match(t):
        return "B question"
    return "C descriptive"


def main():
    raw = pd.read_csv(ROOT / "output/sentence_absa/sentence_absa_raw.csv")
    sent = pd.read_csv(ROOT / "data/processed/sentences.csv")

    z = raw[raw.n_aspects == 0].copy()
    raw["nwords"] = raw.sentence.str.split().str.len()
    z["nwords"] = z.sentence.str.split().str.len()
    print(f"sentences with zero aspects: {len(z)} / {len(raw)} "
          f"({len(z)/len(raw)*100:.1f}%)")

    z["cls"] = z.sentence.apply(classify)
    order = ["A no letters", "A too short", "B question", "C descriptive",
             "D evaluative cue", "E aspect keyword"]
    print("\nclassification:")
    for k in order:
        n = (z.cls == k).sum()
        if n:
            print(f"  {n:5d}  {n/len(z)*100:5.1f}%  {k}")

    suspect = z[z.cls.isin(["D evaluative cue", "E aspect keyword"])]
    clean = len(z) - len(suspect)
    print(f"\ngenuinely non-opinionated (A+B+C): {clean} ({clean/len(z)*100:.1f}%)")
    print(f"suspect, model may have missed an opinion: {len(suspect)} "
          f"({len(suspect)/len(z)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("WHY THE SUSPECTS WERE SKIPPED")
    print("=" * 70)
    st = suspect.status.value_counts()
    print(st.to_string())
    # a 'rejected'/'partial' suspect is a parse failure, not a model judgement
    failed = suspect[suspect.status.isin(["rejected", "badjson", "empty"])]
    print(f"\nof the suspects, dropped by the parser: {len(failed)}")
    print(f"of the suspects, model returned an empty list: "
          f"{(suspect.status == 'ok').sum()}")

    print("\n" + "=" * 70)
    print("SAMPLE — evaluative cue, model still said nothing")
    print("=" * 70)
    for _, r in suspect[suspect.cls == "D evaluative cue"].head(20).iterrows():
        print(f"  [{r.status}] {r.sentence[:130]}")

    print("\n" + "=" * 70)
    print("SAMPLE — aspect keyword present, model still said nothing")
    print("=" * 70)
    for _, r in suspect[suspect.cls == "E aspect keyword"].head(20).iterrows():
        print(f"  [{r.status}] {r.sentence[:130]}")

    out = ROOT / "output/audit/unlabelled_suspects.csv"
    suspect.to_csv(out, index=False)
    print(f"\nwritten {out} ({len(suspect)} rows)")

    # ---- is the skipped set biased? a silent model on long sentences is worse
    print("\n" + "=" * 70)
    print("BIAS CHECK — do skipped sentences differ from labelled ones?")
    print("=" * 70)
    raw["nwords"] = raw.sentence.str.split().str.len()
    lab = raw[raw.n_aspects > 0]
    print(f"  labelled : {len(lab):5d}  mean words {lab.nwords.mean():.1f}")
    print(f"  skipped  : {len(z):5d}  mean words {z.nwords.mean():.1f}")
    print(f"  ratio    : {z.nwords.mean()/lab.nwords.mean():.2f}x")
    print("\n  by star rating (skipped share):")
    for s in sorted(raw.stars.dropna().unique()):
        sub = raw[raw.stars == s]
        print(f"    {s:.0f} star: {len(sub):5d} sentences, "
              f"{(sub.n_aspects == 0).mean()*100:5.1f}% skipped")


if __name__ == "__main__":
    main()
