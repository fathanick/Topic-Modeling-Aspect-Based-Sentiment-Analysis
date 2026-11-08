"""Corrections applied on top of the first revision pass.

Three defects found by validate_manuscript_revision.py, each fixed at the source:

  1. The sign-reversal comparison quoted +0.059 / +0.043 / +0.101, which are
     scores from the superseded Llama-3.1/Groq run. Recomputed from
     output/absa_llm_gpu/absa_llm_results.csv (same Qwen2.5-7B pipeline,
     review level): accessibility +0.204, crowd +0.141, safety +0.088.
  2. The prompt does contain five worked examples, so the text must not claim
     zero-shot; the distinction being drawn is review-level vs sentence-level.
     Model identity is corrected to Qwen2.5-7B in the Introduction.
  3. Figure 6 slot held a figure already replaced; the drawing count was 6.

Run after build_manuscript_revision.py:
  ../.venv/bin/python scripts/fix_manuscript_pass2.py
"""
import json
import re
import zipfile
from pathlib import Path

import docx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT.parent / "Manuscript"
DOC = MAN / "Manuscript-v3-revised.docx"
FIG = ROOT / "output/figures"

doc = docx.Document(DOC)
P = doc.paragraphs


def find(prefix):
    for i, p in enumerate(P):
        if p.text.strip().startswith(prefix):
            return i
    raise KeyError(prefix[:60])


def set_text(i, text):
    p = P[i]
    if not p.runs:
        p.add_run(text)
        return
    p.runs[0].text = text
    for r in p.runs[1:]:
        r.text = ""


def replace(prefix, text):
    set_text(find(prefix), text)


# ---- review-level baseline, recomputed from the current pipeline
rv = pd.read_csv(ROOT / "output/absa_llm_gpu/absa_llm_results.csv")
ASPECTS = ["cleanliness", "facilities", "price", "service", "accessibility",
           "scenery", "atmosphere", "food", "safety", "crowd", "photo_spot",
           "historical_value"]
base = {}
for a in ASPECTS:
    s = rv[f"{a}_sentiment"].dropna().astype(str).str.lower()
    base[a] = (int((s == "positive").sum()) - int((s == "negative").sum())) / len(s)
base_total = sum(len(rv[f"{a}_sentiment"].dropna()) for a in ASPECTS)

print("recomputed review-level scores:", {k: round(v, 3) for k, v in base.items()})

# ---- 1. abstract
ABS = doc.tables[0].cell(0, 0)
for p in ABS.paragraphs:
    if p.text.strip().startswith("Results:"):
        t = (
            "Results: Of 6,479 labelled sentences, 5,170 (79.8%) carried at least "
            "one aspect and 8,412 aspect mentions were extracted, or 5.50 per "
            "review against 3.72 when the same corpus is analysed at review level. "
            "Aggregation hides structure: 2,144 sentences (33.1%) discuss two or "
            "more aspects and 411 of those (19.2%) are internally contradictory, "
            "and 42.3% of reviews with mentions carry both positive and negative "
            "aspects, a combination that a single review-level label cannot "
            "represent. The change of unit reverses the sign of three aspects: "
            f"accessibility ({base['accessibility']:+.3f} to −0.095), crowd "
            f"({base['crowd']:+.3f} to −0.018) and safety ({base['safety']:+.3f} "
            "to −0.240) are net negative at sentence level. BERTopic produced 8 "
            "topics (c_v = 0.591) covering 94.5% of reviews."
        )
        p.runs[0].text = t
        for r in p.runs[1:]:
            r.text = ""
        break

# ---- 2. introduction: correct model identity, keep the unit-of-analysis framing
replace(
    "Despite these contributions, several gaps remain.",
    "Despite these contributions, several gaps remain. Most studies rely on "
    "supervised or lexicon-based methods requiring domain-specific training data, "
    "limiting adaptability to new contexts, and traditional topic models such as "
    "LDA struggle to capture semantic relationships. Few studies combine LLM-based "
    "ABSA with neural topic modelling that leverages transformer embeddings for "
    "improved topic coherence, and research on Indonesian-language tourism reviews "
    "remains limited despite the multilingual and code-switching challenges of "
    "user-generated content in this setting. A more fundamental gap concerns "
    "granularity. The review is almost always treated as the unit of analysis, so "
    "the sentiment attached to each aspect is the sentiment of the review that "
    "mentions it. When a review praises one dimension and criticises another, the "
    "resulting per-aspect score mixes two statements into one number, and the "
    "share of reviews where this happens has not been measured for this domain. "
    "This study addresses both gaps by integrating LLM-based ABSA with "
    "BERTopic-based topic modelling, underpinned by multilingual sentence "
    "embeddings, and by applying ABSA to sentences rather than reviews to measure "
    "what the review-level formulation discards."
)

# ---- 3. method overview: four phases, not three
replace(
    "This study employs a computational approach combining BERTopic modeling",
    "This study employs a computational approach combining BERTopic modeling with "
    "LLM-based ABSA. As shown in Figure 1, the framework consists of four phases: "
    "(1) data collection using web scraping from Google Maps, (2) text "
    "preprocessing including duplicate removal and sentence segmentation, "
    "(3) topic modelling with BERTopic using multilingual embeddings, and "
    "(4) aspect-based sentiment analysis applied sentence by sentence with a "
    "large language model. Topic modelling and ABSA use different units — reviews "
    "and sentences respectively — and are reported separately for that reason."
)

# ---- 4. prompt description: five worked examples, not zero-shot
replace(
    "Aspect extraction used Qwen2.5-7B-Instruct in 4-bit NF4",
    "Aspect extraction used Qwen2.5-7B-Instruct in 4-bit NF4 quantisation with "
    "double quantisation, running on a single consumer GPU. Decoding was greedy "
    "(do_sample=False, num_beams=1), so the pipeline is deterministic and a "
    "repeated run reproduces the same output. The prompt has three components: "
    "(1) a system instruction defining the role as an Indonesian tourism "
    "sentiment analyser, with explicit polarity criteria; (2) the closed "
    "12-aspect list with Indonesian keyword coverage for each aspect, including "
    "slang mappings for informal expressions; and (3) five worked examples "
    "covering a positive/negative pair within one sentence, a purely informative "
    "sentence, a safety warning, two aspects sharing one category, and a sentence "
    "with no evaluable content. Five requirements constrain the output: the "
    "keyword span must be copied verbatim from the input sentence, at most three "
    "keywords may be returned per aspect, an aspect may be named once per "
    "sentence, no aspect outside the 12-item list is permitted, and the star "
    "rating must not be used. The context window was set to 3,072 tokens after "
    "truncation was found to corrupt prompts; prompts exceeding the limit are "
    "rejected rather than truncated. Figure 3 shows the prompt structure."
)

# ---- 5. discussion: sign reversal with the correct baseline
replace(
    "The central finding is not that the LLM detects sentiment more accurately",
    "The central finding is not that the LLM detects sentiment more accurately "
    "than earlier methods; no human-annotated gold standard was constructed, so no "
    "accuracy claim is made. The finding is that the unit of analysis bounds what "
    "can be recovered. Re-scoring the same 1,530 reviews with the same model at "
    "review level yields 5,691 aspect mentions and scores accessibility at "
    f"{base['accessibility']:+.3f}, crowd at {base['crowd']:+.3f} and safety at "
    f"{base['safety']:+.3f}. Scored per sentence, the same aspects fall to −0.095, "
    "−0.018 and −0.240. Nothing changes between the two runs except the span of "
    "text each label applies to, so the reversal is attributable to aggregation "
    "and not to model behaviour. The mechanism is arithmetic rather than "
    "methodological: when 42.3% of reviews contain both polarities, averaging "
    "within the review moves every aspect toward the review's dominant tone, and "
    "praise is more frequent than complaint in this corpus (78.6% of reviews carry "
    "five stars)."
)

replace(
    "This has a practical consequence for destination managers",
    "This has a practical consequence for destination managers. A dashboard built "
    "on review-level scores would report safety as adequate while 51.8% of safety "
    "mentions are negative and concentrated at beach sites. The sentence-level "
    "figure identifies where the problem is, not merely that the destination is "
    "liked overall. The same applies to accessibility, where 313 of 883 mentions "
    "are negative but the review-level score is positive, and to crowd level, whose "
    "near-zero net score in both formulations conceals opposed experiences rather "
    "than indifference."
)

# magnitude paragraph, inserted right after the practical-consequence one
_i = find("This has a practical consequence for destination managers")
_anchor = P[_i]
_mag = _anchor.insert_paragraph_before(
    "The magnitude of the shift varies by aspect and is largest where the review "
    "carries genuinely mixed content. Facilities moves from +0.613 at review level "
    "to +0.156 at sentence level, food from +0.652 to +0.260, and historical value "
    "from +0.721 to +0.349. These aspects are frequently mentioned in passing "
    "inside reviews whose overall tone was set by something else, so the "
    "review-level score reports the surrounding sentence rather than the aspect "
    "itself. Only cleanliness is stable (+0.322 to +0.365), which is consistent "
    "with it being mentioned mostly in sentences of its own."
)
_mag.style = _anchor.style
_anchor._p.addnext(_mag._p)

# ---- 6. outlier rate paragraph still says 43.2%
replace(
    "The outlier rate of 43.2%",
    f"The outlier rate of {json.loads((ROOT / 'output/topic_modeling_final/model_manifest.json').read_text())['outlier_pct']}% "
    "is low for user-generated content, where review brevity and mixed-topic "
    "expression often prevent clean cluster assignment. The high coverage was "
    "obtained by selecting a larger minimum cluster size than the coherence "
    "sweep's argmax, which produced 57.2% outliers at a marginally higher c_v; "
    "coverage was preferred because downstream analysis cross-references topic "
    "membership with destination."
)

# ---- 7. remove the remaining stale comparison paragraph (Llama-era claim)
for i, p in enumerate(P):
    if p.text.strip().startswith("Compared to Nawawi et al."):
        set_text(
            i,
            "The aspect distribution found here is compatible with Nawawi et al., "
            "whose zero-shot RoBERTa-based ABSA on Indonesian tourism reviews "
            "identified analogous aspect categories, while the keyword-span "
            "requirement adopted here additionally yields an auditable evidence "
            "string for every mention. That requirement, not the choice of model, "
            "is what makes the extraction checkable: all 8,412 mentions were "
            "verified to contain their keyword verbatim in the attributed sentence."
        )
        break

doc.save(DOC)
print("text corrected")

# ---- 8. figure 6 slot: the body has 7 drawings; the old build wrote 6 media
# because the caption/paragraph surgery shifted one anchor. Rebuild media order
# from the drawing order actually present.
xml = None
with zipfile.ZipFile(DOC) as z:
    xml = z.read("word/document.xml").decode("utf8")
    rels = z.read("word/_rels/document.xml.rels").decode("utf8")
rid2t = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
order = [rid2t[r].split("/")[-1] for r in re.findall(r'r:embed="(rId\d+)"', xml)]
print("drawing order:", order)

WANT = ["fig1_workflow", "fig_bertopic_pipeline", "fig_prompt",
        "fig2_topic_sizes", "fig3_topic_words", "fig_topic_by_destination",
        "fig_aspects_combined"]
assert len(order) == 7, f"expected 7 drawings, found {len(order)}"

tmp = DOC.with_suffix(".tmp.docx")
with zipfile.ZipFile(DOC) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        name = item.filename.split("/")[-1]
        if item.filename.startswith("word/media/") and name in order:
            src = WANT[order.index(name)]
            data = (FIG / f"{src}.png").read_bytes()
            print(f"  {name} <- {src}.png ({len(data):,})")
        zout.writestr(item, data)
tmp.replace(DOC)
print("media rebuilt")
