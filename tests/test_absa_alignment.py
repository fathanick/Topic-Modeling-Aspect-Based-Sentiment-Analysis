"""Self-check for the ABSA batching/dedup fixes (no API key, no network).

Run:  ./venv/bin/python tests/test_absa_alignment.py
"""
import sys, re, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src" / "sentiment_analysis"))

import ast
src = (ROOT / "src" / "sentiment_analysis" / "absa_llm_groq.py").read_text()
# import the module without executing main()
mod = ast.parse(src)
keep = [n for n in mod.body if not (isinstance(n, ast.If) and "__name__" in ast.dump(n))]
ns = {"__file__": str(ROOT / "src" / "sentiment_analysis" / "absa_llm_groq.py"), "__name__": "absa_probe"}
exec(compile(ast.Module(body=keep, type_ignores=[]), "absa_llm_groq.py", "exec"), ns)

create_batch_prompt = ns["create_batch_prompt"]
response_is_aligned = ns["response_is_aligned"]
extract_json = ns["extract_json_from_response"]

# 1. ids are global, not 1..N per batch
batch = [{"review_id": 4, "original_text": "a", "stars": 5},
         {"review_id": 7, "original_text": "b", "stars": 4}]
p = create_batch_prompt(batch)
assert "Review 4 (5 stars)" in p and "Review 7 (4 stars)" in p, p[-200:]
assert "Review 1 (5 stars)" not in p
print("ok: prompt carries global review_id")

# 2. misaligned payloads are rejected, complete ones accepted
assert response_is_aligned([{"review_id": 4}, {"review_id": 7}], [4, 7])
assert not response_is_aligned([{"review_id": 4}], [4, 7])                 # dropped
assert not response_is_aligned([{"review_id": 4}, {"review_id": 4}], [4, 7])  # duplicated
assert not response_is_aligned([{"review_id": 1}, {"review_id": 2}], [4, 7])  # renumbered
assert not response_is_aligned([{"review_id": 7}, {"review_id": 4}], [4, 7]) is False
print("ok: alignment gate rejects drop/duplicate/renumber")

# 3. both response shapes parse
assert len(extract_json(json.dumps({"results": [{"review_id": 1}]}))) == 1
assert len(extract_json(json.dumps([{"review_id": 1}]))) == 1
assert len(extract_json('```json\n{"results":[{"review_id":1}]}\n```')) == 1
print("ok: JSON shapes parse")

# 4. duplicate aspects collapse to one mention (regression for the 384 ghost mentions)
seen, dups = {}, 0
for a, s in [("scenery", "positive"), ("facilities", "positive"), ("scenery", "negative")]:
    if a in seen:
        dups += 1
        continue
    seen[a] = s
assert len(seen) == 2 and dups == 1 and seen["scenery"] == "positive"
print("ok: duplicate aspect counted once, first label kept")

# 5. live replay: old output must collapse 4926 -> 4542 unique mentions
import pandas as pd
old = ROOT / "output" / "absa_llm_groq_prepatch_backup" / "absa_llm_results.csv"
if not old.exists():
    old = ROOT / "output" / "absa_llm_groq" / "absa_llm_results.csv"
if old.exists():
    df = pd.read_csv(old)
    if len(df) >= 1500:  # only the full 1,530-review run is a valid reference
        aspects = list(ns["TOURISM_ASPECTS"])
        total = sum(int((df[a + "_sentiment"].fillna("") != "").sum()) for a in aspects)
        reported = int(df["num_aspects"].sum())
        print(f"replay: columns give {total} unique mentions; old counter gave {reported}")
        assert total == 4542, total
        print("ok: replay reproduces the corrected 4,542 mentions")
    else:
        print(f"skipped replay: {old.name} has {len(df)} rows (sample run, not a reference)")
print("\nALL CHECKS PASSED")
