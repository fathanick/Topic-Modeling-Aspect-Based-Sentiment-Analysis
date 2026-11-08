"""Build the revised manuscript from the verified current artifacts.

Reason for existing: every quantitative claim in the old DOCX refers to a
superseded run (59 destinations, 23 topics, 4,926 review-level mentions,
Llama 3.1 via Groq). This rewrites the text and tables against
`output/`, embeds the regenerated figures, and leaves the reference list
(24 Mendeley SDTs) untouched.

The abstract lives inside a 1x1 table; body prose is paragraph-level.
Footnote: manuscript numbering must stay manual because the template uses
SEQ fields that do not recalculate (word/settings.xml has no updateFields).

Run: ../.venv/bin/python scripts/build_manuscript_revision.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT.parent / "Manuscript"
SRC = MAN / ("Final-Submission-Extracting Actionable Insights from Tourism "
             "Destination Reviews Using BERTopic and LLM-Based Aspect "
             "Sentiment Analysis.docx")
OUT = MAN / "Manuscript-v3-revised.docx"
FIG = ROOT / "output/figures"

import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

TOPIC_MAN = json.loads((ROOT / "output/topic_modeling_final/model_manifest.json").read_text())

shutil.copy(SRC, OUT)
doc = docx.Document(OUT)

PARAS = doc.paragraphs
print(f"loaded {len(PARAS)} paragraphs, {len(doc.tables)} tables")


def find(prefix, start=0):
    """Return index of first paragraph whose stripped text startswith prefix."""
    for i in range(start, len(PARAS)):
        if PARAS[i].text.strip().startswith(prefix):
            return i
    raise KeyError(f"not found: {prefix[:60]}")


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DRAW = f"{W}drawing"


def set_text(i, text):
    """Replace a paragraph's visible text, preserving its style.

    Paragraphs that hold an inline drawing (a figure with its caption in the
    same paragraph) must keep the drawing: rewriting runs would delete it.
    Any drawing found is moved into its own paragraph immediately before this
    one, so the figure and the caption stay adjacent.
    """
    p = PARAS[i]
    drawings = p._p.findall(f".//{DRAW}")
    if drawings:
        holder = p.insert_paragraph_before("")
        holder.style = p.style
        for d in drawings:
            run = holder._p.makeelement(f"{W}r", {})
            holder._p.append(run)
            run.append(d)
    if not p.runs:
        p.add_run(text)
        return p
    p.runs[0].text = text
    for r in p.runs[1:]:
        r.text = ""
    return p


def set_body(texts, anchor_prefix, occurrence=1, style="Normal"):
    """Replace an existing body paragraph identified by a unique prefix.

    `texts` may be a single string or a list: the first replaces the anchor and
    the remaining ones are inserted after it, all copying the anchor's style.
    """
    if isinstance(texts, str):
        texts = [texts]
    seen = 0
    for i, p in enumerate(PARAS):
        if p.text.strip().startswith(anchor_prefix):
            seen += 1
            if seen == occurrence:
                set_text(i, texts[0])
                anchor = PARAS[i]
                for t in texts[1:]:
                    new = anchor.insert_paragraph_before(t)
                    new.style = anchor.style
                    anchor._p.addnext(new._p)
                    anchor = new
                return
    raise KeyError(anchor_prefix[:50])


def insert_after(anchor_prefix, texts):
    """Insert new paragraphs after the paragraph matching anchor_prefix."""
    i = find(anchor_prefix)
    base = PARAS[i]
    new = []
    for t in texts:
        p = base.insert_paragraph_before(t)
        p.style = base.style
        new.append(p)
    # move them after the anchor, preserving order
    for p in new:
        base._p.addnext(p._p)
    return new


print("\n=== abstract ===")
# ---------------------------------------------------------------- abstract
abs_cell = doc.tables[0].cell(0, 0)
abs_paras = [p for p in abs_cell.paragraphs]

ABSTRACT = {
    "Background:": (
        "Background: Online reviews are a primary source of tourism intelligence, "
        "but their volume and informality make manual reading impractical. "
        "Computational pipelines typically assign one sentiment label per review. "
        "This aggregation is convenient, yet a review that praises the scenery while "
        "criticising the road is summarised into a single polarity that matches "
        "neither statement. The size of that loss has not been quantified for "
        "Indonesian tourism reviews."
    ),
    "Objective:": (
        "Objective: This study quantifies how much aspect-level sentiment is lost "
        "when a review is treated as one unit, and develops a sentence-level "
        "framework that recovers it. It combines BERTopic topic modelling with "
        "large language model (LLM) aspect-based sentiment analysis (ABSA) applied "
        "to individual sentences rather than whole reviews."
    ),
    "Methods:": (
        "Methods: A computational study was conducted on 1,530 Google Maps reviews "
        "from 34 tourist destinations in Yogyakarta, Indonesia. Reviews were split "
        "into 6,552 sentences with a round-trip-validated, slice-based splitter. "
        "Aspect extraction was performed with Qwen2.5-7B-Instruct in 4-bit NF4 "
        "quantisation under greedy decoding, using a closed 12-aspect taxonomy and "
        "an evidence-span requirement: every extracted keyword must appear "
        "verbatim in the sentence it is attributed to. BERTopic with multilingual "
        "Sentence-BERT embeddings was applied separately to the reviews."
    ),
    "Results:": (
        "Results: Of 6,479 labelled sentences, 5,170 (79.8%) carried at least one "
        "aspect and 8,412 aspect mentions were extracted, or 5.50 per review "
        "against 3.72 for the same corpus analysed at review level. Aggregation "
        "hides structure: 2,144 sentences (33.1%) discuss two or more aspects and "
        "411 of those (19.2%) are internally contradictory, and 42.3% of reviews "
        "with mentions carry both positive and negative aspects, a combination a "
        "single review-level label cannot represent. Re-extraction at sentence "
        "level reverses the sign of three aspects: accessibility (−0.10), crowd "
        "(−0.02) and safety (−0.24) are net negative, whereas review-level "
        "aggregation scored them as positive (+0.059, +0.043 and +0.101). BERTopic "
        "produced 8 topics (c_v = 0.591) covering 94.5% of reviews."
    ),
    "Conclusion:": (
        "Conclusion: Unit of analysis, not model choice, determines what a review "
        "corpus can reveal. Review-level ABSA systematically understates negative "
        "aspects because praise and complaint co-occur in the same review; "
        "sentence-level extraction separates them. The framework requires no "
        "labelled training data and is transferable to other review domains where "
        "a single text carries mixed evaluations."
    ),
}
for p in abs_paras:
    t = p.text.strip()
    for k, v in ABSTRACT.items():
        if t.startswith(k):
            if p.runs:
                p.runs[0].text = v
                for r in p.runs[1:]:
                    r.text = ""
            break

print("=== introduction ===")
# ---------------------------------------------------------------- intro gap paragraph
set_body(
    "To address these challenges, topic modeling and aspect-based sentiment "
    "analysis (ABSA) offer complementary solutions. Topic modeling automatically "
    "uncovers latent thematic patterns from large collections of unstructured "
    "reviews, while ABSA goes further by identifying specific aspects and "
    "determining sentiment toward each independently, unlike general sentiment "
    "analysis, which produces only a single polarity score. Together, they enable "
    "stakeholders to pinpoint both dominant topics and specific areas for "
    "improvement. A further assumption is usually left implicit: that the review "
    "is the natural unit of analysis. This assumption is questionable for "
    "narrative reviews, in which a single text often praises one dimension while "
    "criticising another. When such a review is aggregated into one label, the "
    "resulting polarity describes neither statement.",
    "To address these challenges")

# replace the objectives paragraph with one that includes the new objective
set_body(
    "The main objectives of this study are as follows: (1) to identify latent "
    "thematic patterns using BERTopic, (2) to implement LLM-based ABSA for "
    "Indonesian tourism reviews, (3) to quantify how much aspect-level information "
    "a review-level formulation discards, and (4) to derive actionable insights "
    "from the recovered sentence-level structure. The contribution is therefore "
    "not a new sentiment model but evidence about the unit of analysis: the same "
    "corpus, read at sentence level, reverses the sign of three of its twelve "
    "aspects.",
    "The main objectives of this study")

print("=== literature: remove unsupported performance claims ===")
# The old text claims few-shot beats zero-shot (not tested here) and that this
# study's output is more detailed than Nawawi et al. (not measured).
set_body(
    "More recent studies have explored transformer-based models with zero-shot "
    "and few-shot learning capabilities. Nawawi et al. integrated aspect-based "
    "sentiment analysis with zero-shot learning using pretrained RoBERTa for "
    "keyword extraction and K-means clustering for aspect construction, "
    "effectively analysing tourist experiences without extensive annotated "
    "datasets. The emergence of large language models has expanded the "
    "possibilities for such analysis. Krugmann and Hartmann benchmarked GPT-3.5, "
    "GPT-4 and Llama 2 against established transfer learning models on sentiment "
    "tasks, reporting that LLMs in a zero-shot setting can approach fine-tuned "
    "classifiers when labelled training data are scarce. Zhang et al. conducted a "
    "comprehensive assessment of LLMs across sentiment analysis sub-tasks, "
    "including ABSA, and found that coarser polarity tasks are handled well while "
    "structured aspect-level extraction remains more demanding. These results "
    "motivate a constrained prompt design in the present study: a closed aspect "
    "list, an explicit output schema, and a requirement that every keyword be "
    "copied from the source sentence.",
    "More recent studies have explored transformer-based models")

print("=== method ===")
# 3.1 data collection
set_body(
    "Data were collected from Google Reviews for tourist destinations in "
    "Yogyakarta, a major cultural and educational tourism hub in Indonesia. The "
    "destination list was compiled from yogyes.com, a local tourism portal, "
    "yielding 34 destinations across eight categories: beaches, temples, cultural "
    "sites, museums, nature, historical sites, religious sites and agrotourism, as "
    "summarised in Table 1. Beaches dominate the corpus, accounting for 18 of the "
    "34 destinations.",
    "Data were collected from Google Reviews")

set_body(
    "Reviews were extracted using Playwright, a browser-based automation tool, "
    "collecting up to 100 reviews per destination and producing an initial dataset "
    "of approximately 3,000 reviews. These reviews, primarily written in Bahasa "
    "Indonesia, frequently incorporate informal expressions, local slang and "
    "English terms, reflecting the multilingual and code-switching tendencies "
    "typical of Indonesian user-generated content. Table 2 presents sample reviews "
    "with their star ratings.",
    "Reviews were extracted using Playwright")

# 3.2 preprocessing
set_body(
    "Text preprocessing proceeded in four stages. First, exact-text duplicate "
    "removal reduced the dataset by 49%, from 3,000 to 1,530 unique reviews. "
    "Second, text cleaning applied lowercase conversion and removed URLs, email "
    "addresses and digits. Third, punctuation and special characters were removed. "
    "Fourth, stop-word removal applied a combined list of 453 words: 255 unique "
    "Indonesian stop words defined in the preprocessing module and the 198 English "
    "stop words supplied by NLTK. Table 3 presents the dataset statistics after "
    "preprocessing.",
    "Text preprocessing was performed")

print("=== 3.4 sentence-level ABSA (new) ===")
# Rewrite the ABSA setup paragraphs
set_body(
    "In this study, we defined 12 tourism-specific aspects based on dimensions "
    "commonly discussed in tourism review literature and recurring in Indonesian "
    "travel reviews: cleanliness, facilities, price/value, service, accessibility, "
    "scenery/view, atmosphere, food and beverage, safety, crowd level, photo spots, "
    "and historical/cultural value. The taxonomy is closed: an aspect absent from "
    "this list cannot be reported, a restriction revisited in Section 5.3.",
    "In this study, we defined 12 tourism-specific aspects")

set_body(
    "Unlike prior work that assigns one label per review, ABSA here operates on "
    "sentences. Each review was split into sentences by a slice-based splitter that "
    "returns character spans of the original text rather than re-joined tokens, so "
    "every sentence is a literal substring of its review. Splitter rules: a period "
    "is not a boundary when preceded by \"Rp\", a bullet character forces a hard "
    "boundary, and no boundary may fall inside a numeric token such as \"5.000\". "
    "The splitter was validated on all 1,530 reviews: 0 round-trip failures, 164 of "
    "164 thousand-separated prices preserved intact, and 0 sentences failing the "
    "substring test, producing 6,552 sentences. Sentences containing no letter "
    "character (for example emoji-only or numeric-only strings) were excluded from "
    "labelling, leaving 6,479.",
    "The LLM-based ABSA was implemented")

set_body(
    "Aspect extraction used Qwen2.5-7B-Instruct in 4-bit NF4 quantisation with "
    "double quantisation, running on a single consumer GPU. Decoding was greedy "
    "(do_sample=False, num_beams=1), so the pipeline is deterministic and a "
    "repeated run reproduces the same output. The prompt has three components: "
    "(1) a system instruction defining the role as an Indonesian tourism sentiment "
    "analyser, with explicit polarity criteria; (2) Indonesian slang mappings for "
    "informal expressions; and (3) the closed 12-aspect list with an output schema "
    "specifying aspect, keyword span and sentiment. Three requirements constrain "
    "the output: the keyword span must be copied verbatim from the input sentence, "
    "at most three keywords may be returned per aspect, and an aspect may be named "
    "only once per sentence. The context window was set to 3,072 tokens after "
    "truncation was found to corrupt prompts; prompts exceeding the limit are "
    "rejected rather than truncated. Figure 3 shows the prompt structure.",
    "Reviews were processed in batches")

# insert the validation paragraph after the scoring formula explanation
set_body(
    "where Sa is the sentiment score for aspect a, Pa the number of positive "
    "mentions, Na the number of negative mentions and Ta the total number of "
    "mentions for that aspect. The score ranges from −1.0 to +1.0. Scores were "
    "categorised into five levels as shown in Table 4.",
    "Based on the formula 1")

print("=== results ===")
set_body(
    f"The 1,530 preprocessed reviews yielded {TOPIC_MAN['n_topics']} topics. "
    f"A total of {1530 - TOPIC_MAN['n_outliers']} reviews ({TOPIC_MAN['coverage_pct']}%) "
    f"were assigned to a topic and {TOPIC_MAN['n_outliers']} ({TOPIC_MAN['outlier_pct']}%) "
    f"remained unassigned as outliers. Coherence was c_v = {TOPIC_MAN['coherence']['c_v']} "
    f"and c_npmi = {TOPIC_MAN['coherence']['c_npmi']}, computed over the eight topics. "
    "The largest topic (T0, 797 reviews, 52.1%) is a beach and coastal cluster; "
    "T1 (450 reviews, 29.4%) groups urban and photo-oriented visits. The remaining "
    "six topics are small and highly specific: museums and guided tours (T2), "
    "Yogyakarta city landmarks (T3), temple complexes (T4), pilgrimage sites (T5), "
    "royal burial sites (T6) and sand-dune jeep tours (T7). Because topics are "
    "derived from the same 1,530 reviews and topic assignment is not disjoint per "
    "sentence, topic identity is used descriptively and is not cross-tabulated "
    "against aspect sentiment in this study.",
    "The 1,530 preprocessed reviews showed")

# topic paragraph 2
set_body(
    "Figure 4 shows the most representative terms per topic. The eight topics "
    "resolve into four interpretable groupings rather than the five thematic "
    "pillars commonly assumed for Yogyakarta: a dominant coastal cluster (T0, T7), "
    "an urban and photography cluster (T1, T3), a heritage cluster (T2, T4, T6) and "
    "a pilgrimage cluster (T5). Culinary tourism does not emerge as an independent "
    "topic under these settings. It appears instead inside the coastal cluster, "
    "where seafood and warung vocabulary co-occurs with beach vocabulary. The "
    "absence of a standalone culinary topic is a property of this corpus and "
    "configuration, not evidence that food is unimportant: at sentence level, food "
    "and beverage receives 662 aspect mentions, ranking sixth of twelve.",
    "Fig 4 presents the top five keywords")

print("=== results: ABSA rewritten ===")
set_body(
    "At sentence level the pipeline extracted 8,412 aspect mentions from 6,479 "
    "labelled sentences, an average of 1.30 mentions per sentence and 5.50 per "
    "review. The review-level formulation applied to the same corpus produced 3.72 "
    "mentions per review, so sentence-level extraction recovers 1.48 times as many "
    "distinct judgements. Of the 6,479 sentences, 5,170 (79.8%) contained at least "
    "one aspect. The remaining 1,309 (20.2%) were retained in the dataset with an "
    "explicit reason code but without aspect labels: 1,205 received an empty aspect "
    "list from the model, and 104 contained an aspect outside the 12-item taxonomy "
    "whose polarity was preserved separately (Section 4.2.1). Across the 8,412 "
    "mentions, 45.8% were positive, 37.8% neutral and 16.4% negative. The neutral "
    "share is high because a mention is neutral when the sentence names an aspect "
    "without evaluating it, for example when describing opening hours.",
    "The LLM-based ABSA pipeline extracted a total of 4,926")

# aspect table narrative
set_body(
    "Table 5 reports the distribution for all 12 aspects, ordered by mention "
    "volume. Scenery/view is discussed most often (1,277 mentions), followed by "
    "atmosphere (1,112), price/value (941), facilities (900) and accessibility "
    "(883). Safety (313), service (335) and cleanliness (375) are discussed least. "
    "Scenery/view and atmosphere are also the most polarised in the positive "
    "direction: 65.2% and 73.3% of their mentions are positive respectively, and "
    "net scores reach +0.59 and +0.63. Photo spots (+0.45), cleanliness (+0.37), "
    "service (+0.37) and historical value (+0.35) follow.",
    "Among the 12 aspects, Scenery/View achieved")

set_body(
    "Three aspects are net negative at sentence level: safety (−0.24), "
    "accessibility (−0.10) and crowd level (−0.02). Safety is the clearest case: "
    "51.8% of its 313 mentions are negative, concentrated in beach destinations "
    "and expressed through wave and current vocabulary. Accessibility carries the "
    "second-largest negative volume in absolute terms (313 negative mentions of "
    "883), driven by road conditions, distance and parking. Crowd level is nearly "
    "balanced (121 positive, 129 negative, 191 neutral), and its net score close to "
    "zero should be read as an average over opposed experiences rather than as "
    "indifference. These three aspects score positive under review-level "
    "aggregation; the sign reversal is analysed in Section 5.2.",
    "In contrast, several aspects revealed notably lower sentiment scores")

set_body(
    "The remaining aspects fall in the moderate range. Food and beverage scores "
    "+0.26 with 56.2% neutral mentions, reflecting that many food references "
    "describe what was eaten rather than how well it was rated. Price/value and "
    "facilities both score +0.16 and are the most neutral-dominated aspects "
    "(61.0% and 52.2% respectively), indicating that visitors frequently mention "
    "cost or infrastructure without expressing approval or disapproval. Historical "
    "value (+0.35) and photo spots (+0.45) complete the profile.",
    "The remaining aspects occupied the moderate-to-good sentiment range")

print("=== new section 4.2.1 on the no-aspect sentences ===")
NO_ASPECT_1 = (
    "Of the 1,309 sentences without an aspect, 1,205 received an empty aspect list "
    "from the model and 104 named an aspect outside the closed taxonomy. The 1,205 "
    "are not a residual to be filled: 841 (64.2%) are descriptive statements with no "
    "evaluative marker, 184 (14.1%) mention aspect vocabulary without evaluating "
    "it, 159 (12.1%) are shorter than three words, 109 (8.3%) contain an evaluative "
    "cue whose target is the visit as a whole rather than any of the twelve "
    "aspects, and 16 (1.2%) are questions. Labelling these as neutral would "
    "conflate \"no aspect to evaluate\" with \"evaluated as neutral\", two states "
    "the dataset must keep apart."
)
NO_ASPECT_2 = (
    "For the 104 sentences whose aspect fell outside the taxonomy, the model had "
    "already returned a polarity alongside the rejected aspect; that polarity is "
    "retained in a separate table without an aspect: 60 positive, 27 neutral, 14 "
    "negative and 3 mixed. These mentions are excluded from all aspect statistics "
    "reported here. The negative subset is small but substantive, containing "
    "statements such as \"Parah bgt sih.\" and a reviewer who would not recommend a "
    "beach for swimming at all. They are reported because discarding them would "
    "understate dissatisfaction that the aspect taxonomy cannot index."
)


def insert_named(text, anchor_prefix, style_name):
    """Insert one paragraph after the anchor, with an explicit style."""
    i = find(anchor_prefix)
    anchor = PARAS[i]
    p = anchor.insert_paragraph_before(text)
    try:
        p.style = doc.styles[style_name]
    except KeyError:
        p.style = anchor.style
    anchor._p.addnext(p._p)
    return p


for txt in [NO_ASPECT_2, NO_ASPECT_1]:
    insert_named(txt, "The remaining aspects fall in the moderate range", "Normal")
insert_named("4.2.1 Sentences without an aspect label",
             "The remaining aspects fall in the moderate range", "JISEBI Heading 3")

print("=== new section 4.3 granularity ===")
set_body([
    "Sentence-level extraction makes visible a structure that review-level "
    "labelling cannot represent. Of the 6,479 labelled sentences, 2,144 (33.1%) "
    "carry two or more aspect mentions. Within that group, 411 sentences (19.2%) "
    "contain at least one positive and one negative aspect, so a single label "
    "applied to the sentence would be wrong about part of it. These 411 are not "
    "concentrated in negative reviews: 382 (92.9%) occur in four- and five-star "
    "reviews, where a favourable overall verdict coexists with a specific "
    "complaint.",
    "Figure 6 shows both levels. At review level, 1,488 reviews contain at least "
    "one aspect mention; of these, 630 (42.3%) contain both positive and negative "
    "aspects. Any review-level representation must compress those 630 into one "
    "value, discarding the identity of whichever aspect loses. The sentence-level "
    "representation keeps both, and the aspect table in Section 4.2 is the direct "
    "consequence: the same corpus that appears overwhelmingly positive in "
    "aggregate contains three aspects with negative net scores.",
    "Figure 7 shows how aspect composition varies across the 14 most-reviewed "
    "destinations. The variation is large enough that a single national or "
    "regional average would describe none of them: temple destinations concentrate "
    "on historical value (0.26 of mentions at Prambanan), coastal destinations on "
    "atmosphere and food, and urban destinations on scenery and price. Because "
    "these are composition shares within a destination, they are descriptive and "
    "carry no inference about destination quality.",
], "The remaining aspects fall in the moderate range")

print("=== discussion ===")
set_body(
    "The central finding is not that the LLM detects sentiment more accurately "
    "than earlier methods; no human-annotated gold standard was constructed, so no "
    "accuracy claim is made. The finding is that the unit of analysis bounds what "
    "can be recovered. Under review-level aggregation the corpus reads as "
    "overwhelmingly positive: 81.9% positive in the earlier formulation, and three "
    "aspects that are negative at sentence level appear positive. The mechanism is "
    "arithmetic rather than methodological: when 42.3% of reviews contain both "
    "polarities, averaging within the review moves every aspect toward the "
    "review's dominant tone, and praise is more frequent than complaint in this "
    "corpus (78.6% of reviews carry five stars).",
    "The strong alignment between the overall positive sentiment distribution")

set_body(
    "This has a practical consequence for destination managers. A dashboard built "
    "on review-level scores would report safety as adequate (+0.101 in the earlier "
    "formulation) while 51.8% of safety mentions are negative and concentrated at "
    "beach sites. The sentence-level figure identifies where the problem is, not "
    "merely that the destination is liked overall. The same applies to "
    "accessibility, where 313 of 883 mentions are negative but the review-level "
    "aggregate is positive, and to crowd level, whose near-zero net score "
    "conceals opposed experiences rather than indifference.",
    "The near-equal positive and negative sentiment proportions recorded")

print("=== limitations: rewrite third limitation ===")
set_body(
    "Four limitations constrain these findings. First, the dataset is restricted "
    "to Google Maps reviews, which introduces platform sampling bias: users of this "
    "platform skew toward smartphone-literate, locally connected visitors and may "
    "underrepresent international tourists and older demographics. Second, the "
    "high five-star rate (78.6%) and mean rating (4.68/5) indicate a positive-skew "
    "bias in the source, which inflates positive sentiment proportions and makes "
    "the negative aspects reported here conservative rather than exaggerated. "
    "Third, the closed 12-aspect taxonomy means any dimension outside it is "
    "systematically absent; 104 sentences in this corpus named such an aspect, and "
    "their polarity was retained only without an aspect. Fourth, the scope is "
    "confined to Yogyakarta, and 18 of the 34 destinations are beaches, so the "
    "coastal concentration of safety concerns reflects the sampling frame as well "
    "as the destination. No human-annotated gold standard was constructed, so the "
    "extraction is evaluated for internal consistency and evidence grounding "
    "rather than against external labels.",
    "Four limitations constrain the generalizability")

print("=== conclusions ===")
set_body(
    "This study examined whether the unit of analysis changes what a tourism "
    "review corpus can reveal. Splitting 1,530 Google Maps reviews from 34 "
    "Yogyakarta destinations into 6,479 labelled sentences and extracting aspects "
    "at that level produced 8,412 mentions, 1.48 times the review-level yield from "
    "the same corpus. The additional structure is not noise: 33.1% of sentences "
    "discuss two or more aspects, 19.2% of those are internally contradictory, and "
    "42.3% of reviews carry competing polarities. Re-extraction at sentence level "
    "reverses the sign of three aspects, with safety (−0.24), accessibility (−0.10) "
    "and crowd level (−0.02) net negative where review-level aggregation scored "
    "them positive.",
    "This study successfully demonstrates the effectiveness")

set_body(
    "The primary contribution is evidence that aggregation, not model capability, "
    "governs what is visible in review analytics. A pipeline that assigns one label "
    "per review will report destination satisfaction accurately at the aggregate "
    "level and misreport specific operational problems. Sentence-level extraction "
    "with a closed aspect taxonomy and verbatim evidence grounding resolves this "
    "without requiring annotated training data, and the approach transfers to any "
    "domain where a single text carries mixed evaluations.",
    "The primary contribution of this work lies")

print("=== tables ===")
# TABLE 1 - destination examples: drop non-Yogyakarta entries
t1 = doc.tables[1]
fixes = {
    "Beaches": "Pantai Parangtritis, Pantai Indrayanti, Pantai Baron, Pantai Drini, Pantai Kukup",
    "Temples": "Candi Prambanan, Candi Kalasan, Candi Mendut",
    "Cultural Sites": "Tugu Jogja, Alun-alun Kidul, Prawirotaman, Malioboro",
    "Museums": "Museum Ullen Sentalu, Monumen Jogja Kembali, Studio Alam Gamplong",
    "Nature": "Gumuk Pasir Parangkusumo, Bukit Bintang, Pantai Wediombo",
    "Historical": "Makam Raja-Raja Imogiri, Situs Warungboto",
    "Religious": "Sendang Sono",
    "Agrotourism": "Agrowisata Bhumi Merapi",
}
for row in t1.rows[1:]:
    cat = row.cells[0].text.strip()
    if cat in fixes:
        cell = row.cells[1]
        if cell.paragraphs and cell.paragraphs[0].runs:
            cell.paragraphs[0].runs[0].text = fixes[cat]
            for r in cell.paragraphs[0].runs[1:]:
                r.text = ""

# TABLE 3 - dataset statistics
t3 = doc.tables[3]
stats = {
    "Total destinations": "34",
    "Original reviews collected": "~3,000",
    "After deduplication": "1,530",
    "Deduplication rate": "49%",
    "Average star rating": "4.68/5",
    "5-star reviews": "78.6%",
}
for row in t3.rows[1:]:
    k = row.cells[0].text.strip()
    if k in stats and row.cells[1].paragraphs[0].runs:
        row.cells[1].paragraphs[0].runs[0].text = stats[k]
        for r in row.cells[1].paragraphs[0].runs[1:]:
            r.text = ""

print("=== table 5 rebuild ===")
import pandas as pd
km = pd.read_csv(ROOT / "output/sentence_absa/kalimat_aspek_sentimen.csv")
agg = km.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
for c in ["pos", "neu", "neg"]:
    if c not in agg:
        agg[c] = 0
agg["total"] = agg.sum(axis=1)
agg = agg.sort_values("total", ascending=False)

LABEL = {
    "scenery": "Scenery/View", "atmosphere": "Atmosphere",
    "price": "Price/Value", "facilities": "Facilities",
    "accessibility": "Accessibility", "food": "Food & Beverage",
    "photo_spot": "Photo Spots", "historical_value": "Historical Value",
    "crowd": "Crowd Level", "cleanliness": "Cleanliness",
    "service": "Service", "safety": "Safety",
}
t5 = doc.tables[5]
rows = t5.rows
# existing table: header + 12 aspects + overall = 14 rows
for idx, (asp, r) in enumerate(agg.iterrows(), start=1):
    if idx >= len(rows):
        break
    cells = rows[idx].cells
    score = (r.pos - r.neg) / r.total
    vals = [
        LABEL.get(asp, asp),
        f"{int(r.total):,}",
        f"{100*r.pos/r.total:.1f}",
        f"{100*r.neg/r.total:.1f}",
        f"{100*r['neu']/r.total:.1f}",
        f"{score:+.3f}",
    ]
    for c, v in zip(cells, vals):
        if c.paragraphs and c.paragraphs[0].runs:
            c.paragraphs[0].runs[0].text = v
            for rr in c.paragraphs[0].runs[1:]:
                rr.text = ""
tot = agg.total.sum()
pos_p = 100 * agg.pos.sum() / tot
neg_p = 100 * agg.neg.sum() / tot
neu_p = 100 * agg["neu"].sum() / tot
last = rows[len(agg) + 1].cells
for c, v in zip(last, ["Overall", f"{tot:,}", f"{pos_p:.1f}",
                       f"{neg_p:.1f}", f"{neu_p:.1f}", "—"]):
    if c.paragraphs and c.paragraphs[0].runs:
        c.paragraphs[0].runs[0].text = v
        for rr in c.paragraphs[0].runs[1:]:
            rr.text = ""

print("=== figures / captions ===")
import zipfile  # noqa: E402

# Captions are numbered manually: the template's SEQ fields do not recalculate
# (word/settings.xml has no <w:updateFields/>), so a field-based caption would
# silently keep the old text. Slot order below is the drawing order in the body.
SLOTS = [
    "Fig. 1. Research workflow: reviews, sentence splitting, aspect extraction "
    "and sentence-level analysis.",
    "Fig. 2. BERTopic pipeline. Parameters are those of the final model "
    f"(min_cluster_size={TOPIC_MAN['min_cluster_size']}, "
    f"{TOPIC_MAN['cluster_selection_method']}).",
    "Fig. 3. Prompt structure for aspect extraction. Every keyword must be "
    "copied verbatim from the sentence.",
    "Fig. 4. Topic composition: {n} topics, {cov}% of reviews assigned, "
    "{out} outliers ({op}%).",
    "Fig. 5. Net sentiment score by aspect. Three aspects are negative at "
    "sentence level.",
    "Fig. 6. Share of each topic within the 16 most-reviewed destinations.",
    "Fig. 7. Aspect mentions by polarity (left) and net sentiment score per "
    "aspect (right). Three aspects are negative at sentence level.",
]


def _swap_caption(i, text):
    p = PARAS[i]
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""


# caption paragraph indices in body order (1-based slot -> paragraph index)
CAP_IDX = [26, 44, 56, 70, 74, 79, 97]
assert len(PARAS) > max(CAP_IDX)

n_topics = TOPIC_MAN["n_topics"]
fmt = {
    "n": n_topics,
    "cov": TOPIC_MAN["coverage_pct"],
    "out": TOPIC_MAN["n_outliers"],
    "op": TOPIC_MAN["outlier_pct"],
}
for idx, cap in zip(CAP_IDX, SLOTS):
    _swap_caption(idx, cap.format(**fmt))
print("captions rewritten")

# The two paragraphs between Fig 5 and Fig 6 described heatmap/dendrogram figures
# that no longer exist. Replace them with the granularity discussion.
set_body(
    "Figure 5 orders the same aspects by net score and makes the polarity "
    "structure explicit: nine aspects lie above zero, three below. Because the "
    "score is computed per aspect from sentence-anchored mentions, a negative "
    "value cannot be produced by a single negative review dominating an aspect; "
    "it requires negative mentions to outnumber positive ones within that aspect's "
    "own mention pool.",
    "The topic similarity heatmap in Fig 5")

set_body(
    "Figure 6 breaks the topic structure down by destination, showing the share of "
    "each topic within the 16 most-reviewed sites. The distribution is highly "
    "concentrated: the coastal topic T0 accounts for more than 80% of reviews at "
    "nine destinations, while the niche topics are almost entirely confined to one "
    "site each (T2 to Museum Ullen Sentalu, T5 to Sendang Sono, T6 to Makam "
    "Raja-Raja Imogiri). This is consistent with the topic sizes in Figure 4 and "
    "confirms that the small topics are site-specific rather than thinly spread.",
    "The hierarchical clustering dendrogram in Fig 6")

# The ABSA results section needs one forward reference to the new figures.
set_body(
    "Figure 7 reports the mention volume of each aspect split by polarity (left) "
    "and the resulting net score (right). The two views disagree informatively: an "
    "aspect can be both frequently discussed and weakly polarised, as with "
    "price/value, which receives 941 mentions but 61.0% of them neutral.",
    "Figure 5 orders the same aspects by net score")


# Insert Fig 7 reference after the granularity paragraphs, then save.
set_body(
    "Figure 7 shows the aspect profile that this recovers. The left panel reports "
    "the 8,412 mentions split by polarity, and the right panel the resulting net "
    "score per aspect. Two readings matter. First, volume and polarity are "
    "independent: price/value receives 941 mentions but 61.0% of them are neutral, "
    "so a frequently discussed aspect need not be a polarised one. Second, the "
    "three negative aspects are negative by a clear margin in their own mention "
    "pools (safety −0.24, accessibility −0.10, crowd level −0.02), not because a "
    "few reviews dominate.",
    "Figure 6 breaks the topic structure down")

doc.save(OUT)
print(f"saved {OUT}")

# now swap media
IMAGE_ORDER = [
    "fig1_workflow",             # 1 workflow
    "fig_bertopic_pipeline",     # 2 BERTopic steps
    "fig_prompt",                # 3 prompt
    "fig2_topic_sizes",          # 4 topic volumes
    "fig3_topic_words",          # 5 topic words
    "fig_topic_by_destination",  # 6 topic x destination
    "fig_aspects_combined",      # 7 aspects (volume + net score)
]
tmp = OUT.with_suffix(".tmp.docx")
with zipfile.ZipFile(OUT) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    media = sorted([n for n in zin.namelist() if n.startswith("word/media/image")],
                   key=lambda s: int(re.search(r"image(\d+)", s).group(1)))
    print("media slots:", media)
    mapping = dict(zip(media, IMAGE_ORDER))
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename in mapping:
            data = (FIG / f"{mapping[item.filename]}.png").read_bytes()
            print(f"  {item.filename} <- {mapping[item.filename]}.png ({len(data):,} bytes)")
        zout.writestr(item, data)
tmp.replace(OUT)
print(f"figures replaced in {OUT}")
