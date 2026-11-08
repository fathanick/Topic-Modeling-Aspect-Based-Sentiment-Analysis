"""Self-check for the local-GPU ABSA runner (no GPU, no model download).

Covers the parts that actually decide correctness: prompt construction, JSON
parsing, the verbatim-evidence rule, aspect dedup, and checkpoint resume.

Run:  ./venv/bin/python tests/test_local_gpu.py
"""
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
MOD = ROOT / "src" / "sentiment_analysis" / "absa_llm_local_gpu.py"

spec = importlib.util.spec_from_file_location("gpu_absa", MOD)
g = importlib.util.module_from_spec(spec)
exec(compile(MOD.read_text(), str(MOD), "exec"), g.__dict__)

# 1. prompt carries the review, the star rating, and the few-shot pairs
msgs = g.build_messages("Pantainya indah tapi parkir jauh", 3)
assert msgs[0]["role"] == "system"
assert msgs[-1]["role"] == "user" and "3 stars" in msgs[-1]["content"]
assert "Pantainya indah tapi parkir jauh" in msgs[-1]["content"]
assert sum(1 for m in msgs if m["role"] == "assistant") == len(g.FEW_SHOT)
print("ok: prompt has system + few-shot + review with stars")

# 2. quotes in review text must not break the JSON-ish prompt framing
msgs = g.build_messages('Dia bilang "bagus" tapi mahal', 4)
assert '"bagus"' not in msgs[-1]["content"]
print("ok: embedded quotes neutralised")

# 3. clean JSON parses, aspect keys normalised
raw = json.dumps({"aspects": [{"aspect": "Photo Spot", "sentiment": "positive",
                               "evidence": "spot foto"}],
                  "overall_sentiment": "positive"})
out = g.parse_output(raw, "spot foto nya bagus")
assert out["aspects"][0]["aspect"] == "photo_spot"
print("ok: aspect key normalised (Photo Spot -> photo_spot)")

# 4. markdown fences and trailing commas survive
fenced = '```json\n{"aspects":[{"aspect":"scenery","sentiment":"positive",\n "evidence":"indah",},],"overall_sentiment":"positive"}\n```'
out = g.parse_output(fenced, "pemandangan indah sekali")
assert out is not None and len(out["aspects"]) == 1
print("ok: fenced JSON with trailing comma parses")

# 5. hallucinated evidence is stripped, not trusted
raw = json.dumps({"aspects": [{"aspect": "scenery", "sentiment": "positive",
                               "evidence": "pemandangan memukau luar biasa"}],
                  "overall_sentiment": "positive"})
out = g.parse_output(raw, "tempatnya biasa aja")
assert out["aspects"][0]["evidence"] == ""
print("ok: non-verbatim evidence dropped")

# 6. duplicate aspect kept once; unknown aspect dropped
raw = json.dumps({"aspects": [
    {"aspect": "scenery", "sentiment": "positive", "evidence": "indah"},
    {"aspect": "scenery", "sentiment": "negative", "evidence": "indah"},
    {"aspect": "vibes", "sentiment": "positive", "evidence": "indah"}]})
out = g.parse_output(raw, "indah banget")
assert len(out["aspects"]) == 1 and out["aspects"][0]["sentiment"] == "positive"
print("ok: duplicate collapsed (first wins), unknown aspect rejected")

# 7. malformed output -> None, which the runner records as 'failed'
assert g.parse_output("I think the review is positive", "bagus") is None
assert g.parse_output("{not json at all", "bagus") is None
print("ok: unparseable output returns None (status=failed)")

# 8. flatten produces all 12 aspect columns whether or not they were mentioned
rec = g.flatten({"aspects": [{"aspect": "scenery", "sentiment": "positive", "evidence": "x"}],
                 "overall_sentiment": "positive"})
aspect_cols = [k for k in rec if k.endswith("_sentiment") and k != "overall_sentiment"]
assert len(aspect_cols) == 12, aspect_cols
assert rec["scenery_sentiment"] == "positive" and rec["safety_sentiment"] == ""
assert rec["num_aspects"] == 1
print("ok: flatten emits 12 aspect columns")

# 9. checkpoint resume: append, reload, tolerate truncated tail
tmp = Path(tempfile.mkdtemp()) / "ckpt.jsonl"
g.append_checkpoint(str(tmp), {"review_id": 0, "analysis_status": "success"})
g.append_checkpoint(str(tmp), {"review_id": 1, "analysis_status": "success"})
with open(tmp, "a") as f:
    f.write('{"review_id": 2, "analysis_st')
assert set(g.load_checkpoint(str(tmp))) == {0, 1}
print("ok: checkpoint resume tolerates truncated tail")

# 10. no-BOS duplication: generation is sliced past the prompt
src = MOD.read_text()
assert 'new_tokens = out[:, enc["input_ids"].shape[1]:]' in src
assert "do_sample=False" in src
print("ok: greedy decoding, prompt tokens sliced out")


# 11. truncated generation: complete objects before the cut are salvaged
truncated = ('{"aspects": [{"aspect": "scenery", "sentiment": "positive", "evidence": "indah"}, '
             '{"aspect": "facilities", "sentiment": "negative", "evidence": "toilet kotor"}, '
             '{"aspect": "price", "sentiment": "posi')
out = g.parse_output(truncated, "indah tapi toilet kotor")
assert out is not None, "truncated output should be salvaged, not dropped"
assert [a["aspect"] for a in out["aspects"]] == ["scenery", "facilities"]
print("ok: truncated JSON salvaged (2 complete aspects kept)")

# 12. salvage still enforces the verbatim rule and valid aspect names
truncated = ('{"aspects": [{"aspect": "scenery", "sentiment": "positive", "evidence": "tidak ada di teks"}, '
             '{"aspect": "vibes", "sentiment": "positive", "evidence": "indah"}, {"aspect": "price"')
out = g.parse_output(truncated, "indah sekali")
assert out["aspects"][0]["evidence"] == "", "non-verbatim evidence must be dropped after salvage"
assert len(out["aspects"]) == 1
print("ok: salvage respects verbatim rule + aspect whitelist")

# 13. input cap is generous enough for long reviews
assert g.MAX_INPUT_CHARS >= 3000, g.MAX_INPUT_CHARS
print("ok: input cap >= 3000 chars")

print("\nALL CHECKS PASSED")
