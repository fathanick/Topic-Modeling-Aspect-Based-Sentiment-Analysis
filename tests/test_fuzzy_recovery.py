"""Tests for the fuzzy recovery path.

The rule under test: a fuzzy recovery must cover the WHOLE opinion, keep word
order, and store a literal substring of the sentence. Any recovery that drops
the evaluative word is worse than no recovery, because the sentiment stored
alongside it then refers to text that is not there.

Run: ./venv/bin/python tests/test_fuzzy_recovery.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sentiment_analysis"))

from sentence_absa import _recover, _recover_fuzzy  # noqa: E402

fails = []


def ck(name, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'} {name}  {detail}")
    if not cond:
        fails.append(name)


S1 = "Suasana pagi hari saat matahari terbit sangat mempesona, dengan kabut tipis."
S2 = "Candi Kalasan merupakan candi Buddha yang lebih tua umurnya dari Cabdi Borobudur."
S3 = "MUNGKIN mkanan yg di warung stlh pintu kluar candi hrg nya lbh ramah di kantong."
S4 = "Akses jalan menuju pantai bisa segera diperbaiki masih rusak seperti yang dulu."
S5 = "Waktu ditanya harga, memang dijawab dan affordable."

# 1. a plain typo is recovered and keeps the whole phrase
got = _recover_fuzzy("Candi Borobudur", S2)
ck("typo in a short phrase is refused (cosmetic-only edits allowed)",
   got is None or "Borobudur" in got, repr(got))
got2 = _recover_fuzzy("Prambananan adalah warisan agung",
                      "Prambanan adalah warisan agung yang harus kita jaga.")
ck("typo in a long phrase recovered", got2 is not None and "warisan agung" in got2,
   repr(got2))

# 2. the exact-match path still wins when the text is present
ck("exact match untouched", _recover("Candi Kalasan", S2) == "Candi Kalasan")

# 3. THE REGRESSION: must not return a short slice that drops the opinion.
#    "suasana pagi hari sangat mempesona" once came back as "Suasana pagi hari
#    saat" — a 0.75 character match with the sentiment word gone.
got = _recover_fuzzy("suasana pagi hari sangat mempesona", S1)
ck("does not drop the evaluative word", got is None or "mempesona" in got, repr(got))

# 4. non-adjacent fragments are refused ("harga affordable" -> "harga,")
got = _recover_fuzzy("harga affordable", S5)
ck("refuses fragmented match", got is None or "affordable" in got, repr(got))

# 5. every recovery is a literal substring
for needle, hay in [
    ("Candi Borobudur", S2),
    ("makanan", S3),
    ("akses jalan menuju pantai masih rusak", S4),
]:
    r = _recover_fuzzy(needle, hay)
    ck(f"substring guarantee {needle[:28]!r}", r is None or r in hay, repr(r))

# 6. a genuine paraphrase is still refused
ck("refuses paraphrase", _recover_fuzzy("pemandangan indah sekali", S1) is None)

# 7. empty / degenerate inputs do not crash
for a, b in [("", S1), (S1, ""), ("   ", S1), ("x", "")]:
    ck(f"degenerate {a[:8]!r}/{b[:8]!r}", _recover_fuzzy(a, b) is None)

# 8. word order is preserved: a reversed phrase is not sewn together
ck("refuses reversed phrase", _recover_fuzzy("mempesona mempesona", S1) is None)

print("\n" + ("ALL FUZZY CHECKS PASSED" if not fails else f"FAILED: {fails}"))
raise SystemExit(1 if fails else 0)
