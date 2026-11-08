"""Build the sentence-level Kaggle notebook from the tested sources.

Run from the repo root. Keeps the notebook in sync with
src/sentiment_analysis/sentence_absa.py and src/analysis/sentence_aspect_table.py
rather than letting it drift as a copy.
"""
import ast
import json

CELLS = []

CELLS.append({
    "cell_type": "markdown", "metadata": {},
    "source": [
        "# Sentence-level ABSA — 6,552 sentences\n", "\n",
        "Unit of analysis is the **sentence**, not the review. One sentence may carry\n",
        "several aspects with different sentiments, each justified by a verbatim span.\n", "\n",
        "Input: `sentences.csv` (dataset `fathanick/sentence-absa-tourism-code`).\n",
        "Output: `sentence_absa_raw.csv` + `kalimat_aspek_sentimen.csv`."]})

CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": ["!pip install -q transformers accelerate bitsandbytes pandas"]})

CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": [
                  "import torch, glob, sys, json, subprocess, shutil\n",
                  "from pathlib import Path\n",
                  "print('torch', torch.__version__)\n",
                  "print('gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"]})

CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": [
                  "# locate the dataset (layout varies: flat or nested)\n",
                  "def find(name):\n",
                  "    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True))\n",
                  "    assert hits, f'{name} not found under /kaggle/input'\n",
                  "    return Path(hits[0])\n",
                  "\n",
                  "src_absa = find('sentence_absa.py')\n",
                  "src_tab  = find('sentence_aspect_table.py')\n",
                  "sent     = find('sentences.csv')\n",
                  "retry    = find('sentences_retry.csv')\n",
                  "print('absa  ', src_absa)\n",
                  "print('table ', src_tab)\n",
                  "print('data  ', sent)\n",
                  "print('retry ', retry)\n",
                  "\n",
                  "work = Path('/kaggle/working/repo')\n",
                  "(work / 'src/sentiment_analysis').mkdir(parents=True, exist_ok=True)\n",
                  "(work / 'src/analysis').mkdir(parents=True, exist_ok=True)\n",
                  "shutil.copy(src_absa, work / 'src/sentiment_analysis/sentence_absa.py')\n",
                  "shutil.copy(src_tab, work / 'src/analysis/sentence_aspect_table.py')\n",
                  "print('sentences', sum(1 for _ in open(sent)) - 1)"]})

CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": [
                  "# drop the 73 sentences that carry no letters (emoji, '…', dates, '<3')\n",
                  "import pandas as pd\n",
                  "df = pd.read_csv(sent)\n",
                  "lab = df.sentence.str.contains(r'[A-Za-z\\u00c0-\\u00ff]', regex=True)\n",
                  "print('rows', len(df), 'labelable', int(lab.sum()), 'skipped', int((~lab).sum()))\n",
                  "print(df[~lab].sentence.value_counts().head(8).to_string())"]})

CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": [
                  "# prompt contract check before spending GPU time\n",
                  "sys.path.insert(0, str(work / 'src/sentiment_analysis'))\n",
                  "from sentence_absa import SYSTEM_PROMPT, build_messages, parse_response\n",
                  "print('prompt chars', len(SYSTEM_PROMPT))\n",
                  "demo = 'Pantainya bersih banget, tapi parkirnya sempit dan jauh.'\n",
                  "print(build_messages(demo)[1]['content'])\n",
                  "out = parse_response(json.dumps({'aspects': [\n",
                  "    {'aspect': 'cleanliness', 'keywords': 'bersih', 'sentiment': 'pos', 'span': 'bersih banget'},\n",
                  "    {'aspect': 'facilities', 'keywords': 'parkirnya', 'sentiment': 'neg', 'span': 'parkirnya sempit dan jauh'}]}), demo)\n",
                  "assert out['status'] == 'ok' and len(out['aspects']) == 2, out\n",
                  "print('parser ok')"]})

CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": [
                  "# smoke test: 12 real sentences\n",
                  "cmd = f'python {work}/src/sentiment_analysis/sentence_absa.py --input {sent} --out /kaggle/working/smoke --limit 12 --batch-size 6'\n",
                  "r = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n",
                  "print(r.stdout[-2500:]); print('ERR', r.stderr[-1500:])"]})

CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": [
                  "sm = pd.read_csv('/kaggle/working/smoke/sentence_absa_raw.csv')\n",
                  "print(sm[['sentence_id','n_aspects','status']].to_string(index=False))\n",
                  "for _, r in sm.head(4).iterrows():\n",
                  "    print('\\nSENT:', r.sentence[:160])\n",
                  "    print('STAT:', r.status, '|', str(r.error)[:120])\n",
                  "    print('RAW :', str(r.raw)[:600])\n",
                  "    print('ASP :', r.aspects[:300])"]})

CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": [
                  "# flip to False only after the smoke output has been inspected\n",
                  "SMOKE_ONLY = False\n",
                  "assert True"]})

CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": [
                  "# full run (resumable)\n",
                  "if SMOKE_ONLY:\n",
                  "    print('SMOKE_ONLY=True - skipping full run')\n",
                  "else:\n",
                  "    cmd = f'python {work}/src/sentiment_analysis/sentence_absa.py --input {retry} --out /kaggle/working/retry --resume --batch-size 8'\n",
                  "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n",
                  "    print(r.stdout[-4000:]); print('ERR', r.stderr[-2000:])"]})

CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": [
                  "import os\n",
                  "if not SMOKE_ONLY:\n",
                  "    res = pd.read_csv('/kaggle/working/retry/sentence_absa_raw.csv')\n",
                  "    print('rows', len(res))\n",
                  "    print(res.status.value_counts().to_string())\n",
                  "    print('mean aspects / sentence', round(res.n_aspects.mean(), 2))\n",
                  "    print('sentences with >=2 aspects', int((res.n_aspects >= 2).sum()))\n",
                  "    print('sentences with 0 aspects', int((res.n_aspects == 0).sum()))\n",
                  "    print(res.error.fillna('').str.slice(0, 90).value_counts().head(8).to_string())"]})

CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": [
                  "# aggregate to the deliverable: sentence -> aspect -> keyword -> sentiment\n",
                  "if not SMOKE_ONLY:\n",
                  "    cmd = f'python {work}/src/analysis/sentence_aspect_table.py --input /kaggle/working/retry/sentence_absa_raw.csv --out /kaggle/working/final'\n",
                  "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n",
                  "    print(r.stdout[-4000:]); print('ERR', r.stderr[-1500:])"]})

CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": [
                  "if not SMOKE_ONLY:\n",
                  "    t = pd.read_csv('/kaggle/working/retry/kalimat_aspek_sentimen.csv')\n",
                  "    print('rows', len(t))\n",
                  "    print(t.head(6).to_string(index=False))\n",
                  "    m = pd.read_csv('/kaggle/working/final/sentence_aspect_mentions.csv')\n",
                  "    print('\\nspan is substrings of its sentence:',\n",
                  "          float(m.apply(lambda r: r.span in r.sentence, axis=1).mean()))\n",
                  "    print('keyword is substring of its sentence:',\n",
                  "          float(m.apply(lambda r: r.keywords in r.sentence, axis=1).mean()))"]})

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}

with open("kaggle_sentence_absa.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

# validate: every code cell must parse
bad = 0
for i, c in enumerate(CELLS):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    src = "\n".join(l for l in src.split("\n") if not l.strip().startswith(("!", "%")))
    try:
        ast.parse(src)
    except SyntaxError as e:
        bad += 1
        print(f"cell {i}: {e}")
print(f"wrote kaggle_sentence_absa.ipynb | {len(CELLS)} cells | syntax errors {bad}")
raise SystemExit(1 if bad else 0)
