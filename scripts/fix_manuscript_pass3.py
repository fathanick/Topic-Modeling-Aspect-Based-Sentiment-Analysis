"""Third pass: remove the last superseded claims and settle the bibliography.

Found by validate_manuscript_revision.py:
  - outlier rate still quoted as 43.2% (superseded run, now 5.5%)
  - "81.9%" sentiment share from the old pipeline
  - Llama 3.1 8B / Groq still named as the model
  - +0.059 / +0.043 / +0.101 old review baseline still in one paragraph
  - "three phases" and "few-shot" wording that no longer matches the method
  - bibliography has 34 citation fields but 24 entries

Run after fix_captions_final.py:
  ../.venv/bin/python scripts/fix_manuscript_pass3.py
"""
import re
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT.parent / "Manuscript" / "Manuscript-v3-revised.docx"

doc = docx.Document(DOC)
P = doc.paragraphs
hits = []


def rewrite(startswith, new_text):
    for p in doc.paragraphs:
        if p.text.strip().startswith(startswith):
            if p.runs:
                p.runs[0].text = new_text
                for r in p.runs[1:]:
                    r.text = ""
            hits.append((startswith[:40], p.text[:70]))
            return True
    return False


# ---- 43.2% outlier rate -> 5.5%
rewrite(
    "The outlier rate of 43.2%",
    "The outlier rate of 5.5% is low for BERTopic on user-generated content, "
    "where review brevity and mixed-topic expression frequently prevent clean "
    "cluster assignment and outliers routinely exceed a quarter of the corpus. "
    "The improvement here is attributable to the reduced corpus size after "
    "destination filtering and to tuning min_cluster_size, which was optimised "
    "over ten candidate configurations using coherence rather than left at its "
    "default. Of the 84 outliers, most are very short reviews that name a single "
    "attraction without describing it, and they were retained rather than "
    "forcibly assigned. Topic coverage of 94.5% means the comparison between "
    "topics and aspects in Section 4.4 rests on almost the whole corpus."
)

# ---- 81.9% sentiment share
rewrite(
    "The central finding is not that the LLM detects sentiment more accurately",
    "The central finding is not that the LLM detects sentiment more accurately "
    "than earlier methods; no human-annotated gold standard was constructed, so "
    "no accuracy claim is made. The finding is structural. At review level the "
    "corpus looks overwhelmingly positive: 70.3% of the 5,691 review-level "
    "mentions are positive against 13.4% negative. At sentence level the same "
    "corpus yields 45.8% positive and 37.8% negative over 8,412 mentions. The "
    "negativity was present in the reviews all along; a single label per review "
    "distributed it across whatever the model happened to select. Nothing about "
    "the data changed between the two rows of Table 5. Only the unit of analysis "
    "did, and that is the contribution this paper claims."
)

# ---- old review-level baseline numbers in the practical-consequence paragraph
for p in doc.paragraphs:
    if "a dashboard built on review-level scores" in p.text.lower():
        t = p.text
        t = t.replace("(+0.101 in the earlier formulation)", "(+0.088 at review level)")
        t = t.replace("(+0.059 in the earlier formulation)", "(+0.204 at review level)")
        t = t.replace("(+0.043 in the earlier formulation)", "(+0.141 at review level)")
        p.runs[0].text = t
        for r in p.runs[1:]:
            r.text = ""
        hits.append(("practical consequence", t[:70]))
        break

# ---- model name everywhere
for p in doc.paragraphs:
    t = p.text
    if "Llama 3.1 8B" in t or "llama-3.1" in t.lower():
        t = re.sub(r"few-shot Llama 3\.1 8B approach", "Qwen2.5-7B-Instruct approach", t)
        t = re.sub(r"Llama 3\.1 8B", "Qwen2.5-7B-Instruct", t)
        if p.runs:
            p.runs[0].text = t
            for r in p.runs[1:]:
                r.text = ""
        hits.append(("model name", t[:70]))

# ---- three phases -> four stages, drop few-shot from method description
rewrite(
    "This study employs a computational approach combining BERTopic",
    "This study employs a computational approach combining BERTopic modeling "
    "with LLM-based ABSA. As shown in Fig. 1, the framework consists of four "
    "stages: (1) review collection from Google Maps across 34 destinations in "
    "the Yogyakarta region; (2) sentence-level splitting of each review so that "
    "aspects are attributed to the sentence that carries them rather than to the "
    "review as a whole; (3) aspect and sentiment extraction with a locally "
    "executed Qwen2.5-7B-Instruct model constrained to a 12-aspect glossary; and "
    "(4) aggregation of mentions by aspect, destination and polarity. Stage 2 is "
    "the methodological difference from prior work, which extracts aspects once "
    "per review."
)

# ---- Groq / few-shot in the gap paragraph
for p in doc.paragraphs:
    if p.text.strip().startswith("Despite these contributions, several gaps remain"):
        t = p.text
        t = re.sub(r"few-shot", "prompted", t)
        t = re.sub(r"\bGroq\b", "a hosted inference endpoint", t)
        if p.runs:
            p.runs[0].text = t
            for r in p.runs[1:]:
                r.text = ""
        hits.append(("gap paragraph", t[:70]))

# ---- few-shot in the related-work paragraph
for p in doc.paragraphs:
    if p.text.strip().startswith("More recent studies have explored transformer-based"):
        t = re.sub(r"\bfew-shot\b", "prompted", p.text)
        if p.runs:
            p.runs[0].text = t
            for r in p.runs[1:]:
                r.text = ""
        hits.append(("related work", t[:70]))

doc.save(DOC)
print(f"rewrote {len(hits)} paragraphs")
for h in hits:
    print("  ", h[0], "->", h[1])

# ---------------------------------------------------------------- bibliography
doc = docx.Document(DOC)
paras = [p for p in doc.paragraphs]
# citation fields still present
xml = doc.element.xml
n_sdt = xml.count("ADDIN Mendeley Bibliography CSL_BIBLIOGRAPHY")
# count entries: paragraphs starting with [n]
entries = [p.text.strip() for p in doc.paragraphs
           if re.match(r"^\[\d+\]", p.text.strip())]
print(f"\nbibliography entries: {len(entries)}; citation field: {n_sdt}")
print("last entry:", entries[-1][:90] if entries else None)
