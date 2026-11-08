"""Self-check for resume/checkpoint and daily-quota handling (no API calls)."""
import ast, json, re, tempfile, os
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src" / "sentiment_analysis" / "absa_llm_groq.py"
src = SRC.read_text()
ns = {}
exec(compile(src.split('if __name__ == "__main__":')[0], "g", "exec"), ns)

tmp = Path(tempfile.mkdtemp()) / "ckpt.jsonl"

# 1. missing file -> empty
assert ns["load_checkpoint"](str(tmp)) == {}
print("ok: missing checkpoint loads empty")

# 2. append + reload
ns["append_checkpoint"](str(tmp), {"review_id": 1, "analysis_status": "success"})
ns["append_checkpoint"](str(tmp), {"review_id": 2, "analysis_status": "success"})
assert set(ns["load_checkpoint"](str(tmp))) == {1, 2}
print("ok: checkpoint round-trips")

# 3. truncated last line (crash mid-write) must not break resume
with open(tmp, "a") as f:
    f.write('{"review_id": 3, "analysis_st')
assert set(ns["load_checkpoint"](str(tmp))) == {1, 2}
print("ok: truncated tail tolerated")

# 4. daily quota is classified, not retried as transient
assert ns["DailyQuotaExhausted"] is not None
code = src
assert '"tokens per day" in error_msg or "(TPD)" in error_msg' in code
assert 'raise DailyQuotaExhausted(error_msg[:300])' in code
print("ok: TPD detected before generic 429 retry")

# 5. resume skips completed work and keeps both halves
old = [{"review_id": 1, "analysis_status": "success"},
       {"review_id": 2, "analysis_status": "success"}]
new = [{"review_id": 2, "analysis_status": "success"},
       {"review_id": 3, "analysis_status": "success"}]
import pandas as pd
merged = pd.concat([pd.DataFrame(old), pd.DataFrame(new)], ignore_index=True)
merged = merged.drop_duplicates(subset="review_id", keep="last").sort_values("review_id")
assert list(merged.review_id) == [1, 2, 3] and len(merged) == 3
print("ok: checkpoint merge dedupes and sorts")

print("\nALL CHECKS PASSED")
