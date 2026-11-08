"""Self-check for the IndoBERTweet baseline fixes (no model download).

Run:  ./venv/bin/python tests/test_indobert_labels.py
"""
import ast, re
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src" / "sentiment_analysis" / "absa_indobertweet.py"

# Extract only the pure label-mapping function: importing the module would pull
# torch/transformers, which the label logic does not need.
src = SRC.read_text()
block = re.search(r"def normalize_sentiment_label.*?\n(?=def )", src, re.S).group(0)
ns = {"__file__": str(SRC)}
exec(compile(ast.parse(block), SRC.name, "exec"), ns)

norm = ns["normalize_sentiment_label"]

# Model card: LABEL_0=positive, LABEL_1=neutral, LABEL_2=negative
assert norm("LABEL_0", 0.99)[0] == "positive"
assert norm("LABEL_1", 0.62)[0] == "neutral"
assert norm("LABEL_2", 0.98)[0] == "negative"
print("ok: label ids follow the model card (no inversion)")

# The old substring match let 'label_1' hit the positive list via 'pos'... and
# '2' matched 'label_2' -> positive. Exact lookup must reject that.
assert norm("LABEL_2", 0.99)[0] != "positive"
assert norm("2", 0.99)[0] == "negative"
print("ok: exact-id matching, no substring collisions")

# A 5-star-positive review must not be forced negative by label handling
assert norm("positive", 0.9)[0] == "positive"
print("ok: natural-language labels still map")

print("\nALL CHECKS PASSED")
print("next: rerun absa_indobertweet.py, then absa_llm_groq.py, then "
      "src/analysis/topic_aspect_join.py")
