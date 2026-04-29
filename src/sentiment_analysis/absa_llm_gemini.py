"""
Aspect-Based Sentiment Analysis (ABSA) using Google Gemini LLM

This script performs ABSA on Indonesian tourism reviews using Google Gemini API
with few-shot prompting for accurate sentiment classification.

Features:
- Few-shot learning with domain-specific examples
- Handles multiple aspects per review
- Batch processing for efficiency
- Structured JSON output
- Rate limiting for free tier compliance

Requirements:
    pip install google-generativeai pandas tqdm

Setup:
    1. Get API key from https://makersuite.google.com/app/apikey
    2. Set environment variable: export GOOGLE_API_KEY="your-api-key"
    Or pass it directly when running the script

Usage:
    python absa_llm_gemini.py
    python absa_llm_gemini.py --api-key YOUR_API_KEY
    python absa_llm_gemini.py --sample 100  # Test with 100 samples
"""

import pandas as pd
import numpy as np
import json
import time
import os
import re
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

try:
    import google.generativeai as genai
except ImportError:
    print("Please install google-generativeai: pip install google-generativeai")
    exit(1)

# ============== Configuration ==============
INPUT_FILE = "data/processed/yogyakarta_tourism_reviews_preprocessed.csv"
OUTPUT_DIR = "output/absa_llm_gemini"
OUTPUT_RESULTS_FILE = "absa_llm_results.csv"
OUTPUT_SUMMARY_FILE = "absa_llm_summary.csv"
OUTPUT_DESTINATION_FILE = "absa_llm_by_destination.csv"
DOCUMENTATION_FILE = "absa_llm_documentation.txt"

# Gemini configuration
MODEL_NAME = "gemini-2.0-flash"  # Fast and cost-effective (use gemini-1.5-flash-latest as fallback)
BATCH_SIZE = 5  # Reviews per API call
RATE_LIMIT_DELAY = 4  # Seconds between requests (increased for free tier limits)
MAX_RETRIES = 3

# ============== Tourism Aspects Definition ==============
TOURISM_ASPECTS = {
    "cleanliness": {
        "name": "Cleanliness",
        "name_id": "Kebersihan",
        "description": "Cleanliness and hygiene of the destination"
    },
    "facilities": {
        "name": "Facilities",
        "name_id": "Fasilitas",
        "description": "Available facilities (toilet, parking, prayer room, etc.)"
    },
    "price": {
        "name": "Price/Value",
        "name_id": "Harga",
        "description": "Pricing, ticket costs, and value for money"
    },
    "service": {
        "name": "Service",
        "name_id": "Pelayanan",
        "description": "Staff service, tour guides, helpfulness"
    },
    "accessibility": {
        "name": "Accessibility",
        "name_id": "Aksesibilitas",
        "description": "Ease of access, transportation, parking"
    },
    "scenery": {
        "name": "Scenery/View",
        "name_id": "Pemandangan",
        "description": "Natural beauty, views, landscape"
    },
    "atmosphere": {
        "name": "Atmosphere",
        "name_id": "Suasana",
        "description": "Ambiance, comfort, peacefulness"
    },
    "food": {
        "name": "Food & Beverage",
        "name_id": "Makanan",
        "description": "Food quality, restaurants, culinary options"
    },
    "safety": {
        "name": "Safety",
        "name_id": "Keamanan",
        "description": "Safety, security, potential hazards"
    },
    "crowd": {
        "name": "Crowd Level",
        "name_id": "Keramaian",
        "description": "Crowding, visitor density, queues"
    },
    "photo_spot": {
        "name": "Photo Spots",
        "name_id": "Spot Foto",
        "description": "Photography opportunities, instagramable spots"
    },
    "historical_value": {
        "name": "Historical/Cultural Value",
        "name_id": "Nilai Sejarah",
        "description": "Historical significance, cultural importance"
    }
}

# ============== Few-Shot Examples ==============
FEW_SHOT_EXAMPLES = """
## Example 1: Positive multi-aspect review
Review: "Candi Prambanan emang nggak pernah gagal bikin kagum. Areanya luas, bersih, dan tertata rapi. Fasilitasnya lengkap, toilet bersih, mushala ada. Akses gampang, parkiran luas. Worth it banget!"

Analysis:
```json
{
  "aspects": [
    {"aspect": "scenery", "sentiment": "positive", "confidence": "high", "evidence": "nggak pernah gagal bikin kagum"},
    {"aspect": "cleanliness", "sentiment": "positive", "confidence": "high", "evidence": "bersih, tertata rapi"},
    {"aspect": "facilities", "sentiment": "positive", "confidence": "high", "evidence": "fasilitasnya lengkap, toilet bersih, mushala"},
    {"aspect": "accessibility", "sentiment": "positive", "confidence": "high", "evidence": "akses gampang, parkiran luas"},
    {"aspect": "price", "sentiment": "positive", "confidence": "high", "evidence": "worth it banget"}
  ],
  "overall_sentiment": "positive"
}
```

## Example 2: Mixed sentiment review
Review: "Pemandangannya bagus banget, tapi sayang parkirnya jauh dan bayarnya mahal. Toiletnya juga kurang bersih. Makanan di warung sekitar biasa aja."

Analysis:
```json
{
  "aspects": [
    {"aspect": "scenery", "sentiment": "positive", "confidence": "high", "evidence": "bagus banget"},
    {"aspect": "accessibility", "sentiment": "negative", "confidence": "high", "evidence": "parkirnya jauh"},
    {"aspect": "price", "sentiment": "negative", "confidence": "high", "evidence": "bayarnya mahal"},
    {"aspect": "cleanliness", "sentiment": "negative", "confidence": "medium", "evidence": "kurang bersih"},
    {"aspect": "food", "sentiment": "neutral", "confidence": "medium", "evidence": "biasa aja"}
  ],
  "overall_sentiment": "mixed"
}
```

## Example 3: Neutral/informational review
Review: "Tiket masuk 50rb per orang. Ada museum dan toko souvenir. Bawa topi karena panas."

Analysis:
```json
{
  "aspects": [
    {"aspect": "price", "sentiment": "neutral", "confidence": "medium", "evidence": "tiket masuk 50rb (informational, no opinion)"},
    {"aspect": "facilities", "sentiment": "neutral", "confidence": "low", "evidence": "ada museum dan toko souvenir (factual mention)"}
  ],
  "overall_sentiment": "neutral"
}
```

## Example 4: Service-focused review
Review: "Petugasnya ramah banget dan helpful. Tour guide-nya juga menjelaskan dengan detail. Pelayanannya memuaskan."

Analysis:
```json
{
  "aspects": [
    {"aspect": "service", "sentiment": "positive", "confidence": "high", "evidence": "ramah banget, helpful, menjelaskan detail, memuaskan"}
  ],
  "overall_sentiment": "positive"
}
```

## Example 5: Negative safety concern
Review: "Hati-hati kalau bawa anak kecil, banyak tangga curam dan licin. Ombaknya juga berbahaya untuk berenang."

Analysis:
```json
{
  "aspects": [
    {"aspect": "safety", "sentiment": "negative", "confidence": "high", "evidence": "tangga curam dan licin, ombak berbahaya"}
  ],
  "overall_sentiment": "negative"
}
```
"""

# ============== System Prompt ==============
SYSTEM_PROMPT = f"""You are an expert Indonesian tourism review sentiment analyzer. Your task is to analyze Indonesian tourism reviews and identify:

1. **Aspects mentioned**: Which tourism-related aspects are discussed in the review
2. **Sentiment per aspect**: Whether the sentiment toward each aspect is POSITIVE, NEGATIVE, or NEUTRAL
3. **Evidence**: The specific words/phrases that indicate the sentiment
4. **Confidence**: HIGH, MEDIUM, or LOW based on clarity of sentiment expression

## Available Aspects:
{json.dumps({k: v['description'] for k, v in TOURISM_ASPECTS.items()}, indent=2, ensure_ascii=False)}

## Important Guidelines:

### Sentiment Classification Rules:
- **POSITIVE**: Words like bagus, indah, bersih, nyaman, worth it, mantap, recommended, keren, enak, ramah, murah, lengkap
- **NEGATIVE**: Words like kotor, mahal, jelek, susah, bahaya, kurang, buruk, mengecewakan, jauh, sempit, rusak
- **NEUTRAL**: Factual statements without clear opinion, words like biasa aja, lumayan, cukup (without strong qualifier)

### Indonesian Informal Language:
- "gak/nggak/ga/engga" = tidak (negation)
- "banget/bgt" = sangat (very/intensifier)
- "mantap/mantul" = excellent
- "worth it/worthed" = worth the price
- "keren" = cool/awesome
- "zonk" = disappointing
- "biasa aja" = ordinary/neutral
- "lumayan" = fairly good (slightly positive)

### Context Awareness:
- Consider the star rating context if available
- "ramai" can be positive (popular) or negative (crowded) depending on context
- "sepi" can be positive (peaceful) or negative (boring) depending on context

## Output Format:
Return ONLY valid JSON with this structure:
```json
{{
  "aspects": [
    {{"aspect": "aspect_key", "sentiment": "positive|negative|neutral", "confidence": "high|medium|low", "evidence": "quoted text"}}
  ],
  "overall_sentiment": "positive|negative|neutral|mixed"
}}
```

If no aspects are mentioned, return: {{"aspects": [], "overall_sentiment": "neutral"}}
"""


class ABSAResults:
    """Class to store ABSA results and statistics"""
    def __init__(self):
        self.total_reviews = 0
        self.processed_reviews = 0
        self.failed_reviews = 0
        self.aspect_counts = defaultdict(int)
        self.aspect_sentiments = defaultdict(lambda: defaultdict(int))
        self.destination_aspects = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.confidence_scores = defaultdict(list)
        self.processing_time = 0
        self.model_name = ""
        self.api_calls = 0


def setup_gemini(api_key=None):
    """Initialize Gemini API"""
    if api_key:
        genai.configure(api_key=api_key)
    elif os.environ.get("GOOGLE_API_KEY"):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    else:
        raise ValueError(
            "No API key provided. Either:\n"
            "1. Pass --api-key argument\n"
            "2. Set GOOGLE_API_KEY environment variable\n"
            "Get your key at: https://makersuite.google.com/app/apikey"
        )

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={
            "temperature": 0.1,  # Low temperature for consistent outputs
            "top_p": 0.95,
            "max_output_tokens": 2048,
        }
    )
    return model


def load_data(input_file, sample_size=None):
    """Load preprocessed data"""
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"   Loaded {len(df)} reviews")

        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"   Sampled {len(df)} reviews for analysis")

        return df
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return None


def create_batch_prompt(reviews_batch):
    """Create prompt for a batch of reviews"""
    prompt = f"""Analyze the following Indonesian tourism reviews for aspect-based sentiment.

{FEW_SHOT_EXAMPLES}

---
Now analyze these reviews. Return a JSON array with one analysis object per review.

"""
    for i, review in enumerate(reviews_batch, 1):
        text = review.get('original_text', review.get('cleaned_text', ''))[:1000]  # Limit text length
        stars = review.get('stars', 'N/A')
        prompt += f"""
## Review {i} (Stars: {stars}):
"{text}"

"""

    prompt += """
Return your analysis as a JSON array:
```json
[
  {"review_id": 1, "aspects": [...], "overall_sentiment": "..."},
  {"review_id": 2, "aspects": [...], "overall_sentiment": "..."},
  ...
]
```
"""
    return prompt


def extract_json_from_response(response_text):
    """Extract JSON from model response"""
    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find raw JSON
        json_str = response_text.strip()

    # Clean up common issues
    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
    json_str = re.sub(r',\s*]', ']', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to extract individual objects
        objects = re.findall(r'\{[^{}]*\}', json_str)
        if objects:
            results = []
            for obj in objects:
                try:
                    results.append(json.loads(obj))
                except:
                    continue
            if results:
                return results
        raise e


def analyze_batch(model, reviews_batch, results_tracker, retry_count=0):
    """Analyze a batch of reviews using Gemini"""
    prompt = create_batch_prompt(reviews_batch)

    try:
        response = model.generate_content(
            [{"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}]
        )
        results_tracker.api_calls += 1

        # Extract JSON from response
        response_text = response.text
        parsed_results = extract_json_from_response(response_text)

        # Ensure it's a list
        if isinstance(parsed_results, dict):
            parsed_results = [parsed_results]

        return parsed_results

    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"   Retry {retry_count + 1}/{MAX_RETRIES} after error: {str(e)[:50]}")
            time.sleep(2 ** retry_count)  # Exponential backoff
            return analyze_batch(model, reviews_batch, results_tracker, retry_count + 1)
        else:
            print(f"   Failed after {MAX_RETRIES} retries: {str(e)[:50]}")
            return None


def run_absa_analysis(df, model, results_tracker):
    """Run ABSA analysis on all reviews"""
    print("\n" + "=" * 60)
    print("RUNNING LLM-BASED ASPECT-BASED SENTIMENT ANALYSIS")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total reviews: {len(df)}")
    print("=" * 60)

    start_time = datetime.now()
    all_results = []

    # Convert DataFrame to list of dicts for easier processing
    reviews_list = df.to_dict('records')

    # Process in batches
    num_batches = (len(reviews_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(reviews_list))
        batch = reviews_list[start_idx:end_idx]

        # Analyze batch
        batch_results = analyze_batch(model, batch, results_tracker)

        if batch_results is None:
            # Handle failed batch - create empty results
            for i, review in enumerate(batch):
                all_results.append({
                    'destination': review['destination'],
                    'username': review.get('username', ''),
                    'stars': review.get('stars', ''),
                    'original_text': str(review.get('original_text', ''))[:500],
                    'aspects_detected': '',
                    'aspect_sentiments': '',
                    'num_aspects': 0,
                    'overall_sentiment': 'error',
                    'analysis_status': 'failed'
                })
                results_tracker.failed_reviews += 1
        else:
            # Process successful results
            for i, review in enumerate(batch):
                result_idx = i

                # Get corresponding analysis result
                if result_idx < len(batch_results):
                    analysis = batch_results[result_idx]
                else:
                    analysis = {"aspects": [], "overall_sentiment": "neutral"}

                # Extract aspects
                aspects = analysis.get('aspects', [])

                # Create result record
                result_record = {
                    'destination': review['destination'],
                    'username': review.get('username', ''),
                    'stars': review.get('stars', ''),
                    'original_text': str(review.get('original_text', ''))[:500],
                    'analysis_status': 'success'
                }

                # Process each aspect
                aspects_found = []
                sentiments_found = []

                for aspect_data in aspects:
                    aspect_key = aspect_data.get('aspect', '').lower().replace(' ', '_')

                    # Validate aspect key
                    if aspect_key not in TOURISM_ASPECTS:
                        continue

                    sentiment = aspect_data.get('sentiment', 'neutral').lower()
                    if sentiment not in ['positive', 'negative', 'neutral']:
                        sentiment = 'neutral'

                    confidence = aspect_data.get('confidence', 'medium').lower()
                    evidence = aspect_data.get('evidence', '')

                    # Add to result record
                    result_record[f'{aspect_key}_sentiment'] = sentiment
                    result_record[f'{aspect_key}_confidence'] = confidence
                    result_record[f'{aspect_key}_evidence'] = evidence[:100]

                    aspects_found.append(aspect_key)
                    sentiments_found.append(f"{aspect_key}:{sentiment}")

                    # Update trackers
                    results_tracker.aspect_counts[aspect_key] += 1
                    results_tracker.aspect_sentiments[aspect_key][sentiment] += 1
                    results_tracker.destination_aspects[review['destination']][aspect_key][sentiment] += 1

                    # Track confidence
                    conf_value = {'high': 1.0, 'medium': 0.7, 'low': 0.4}.get(confidence, 0.7)
                    results_tracker.confidence_scores[aspect_key].append(conf_value)

                # Fill empty columns for non-detected aspects
                for aspect_key in TOURISM_ASPECTS.keys():
                    if aspect_key not in aspects_found:
                        result_record[f'{aspect_key}_sentiment'] = ''
                        result_record[f'{aspect_key}_confidence'] = ''
                        result_record[f'{aspect_key}_evidence'] = ''

                result_record['aspects_detected'] = ', '.join(aspects_found)
                result_record['aspect_sentiments'] = '; '.join(sentiments_found)
                result_record['num_aspects'] = len(aspects_found)
                result_record['overall_sentiment'] = analysis.get('overall_sentiment', 'neutral')

                all_results.append(result_record)
                results_tracker.processed_reviews += 1

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

    end_time = datetime.now()
    results_tracker.processing_time = (end_time - start_time).total_seconds()

    print(f"\nAnalysis completed in {results_tracker.processing_time:.2f} seconds")
    print(f"API calls made: {results_tracker.api_calls}")
    print(f"Successful: {results_tracker.processed_reviews}, Failed: {results_tracker.failed_reviews}")

    return pd.DataFrame(all_results)


def generate_summary(results_tracker):
    """Generate summary statistics"""
    summary_data = []

    for aspect_key, aspect_info in TOURISM_ASPECTS.items():
        total = results_tracker.aspect_counts[aspect_key]
        positive = results_tracker.aspect_sentiments[aspect_key]['positive']
        negative = results_tracker.aspect_sentiments[aspect_key]['negative']
        neutral = results_tracker.aspect_sentiments[aspect_key]['neutral']

        if total > 0:
            pos_pct = (positive / total) * 100
            neg_pct = (negative / total) * 100
            neu_pct = (neutral / total) * 100
            sentiment_score = (positive - negative) / total
            avg_conf = np.mean(results_tracker.confidence_scores[aspect_key]) if results_tracker.confidence_scores[aspect_key] else 0
        else:
            pos_pct = neg_pct = neu_pct = sentiment_score = avg_conf = 0

        summary_data.append({
            'aspect': aspect_key,
            'aspect_name': aspect_info['name'],
            'aspect_name_id': aspect_info['name_id'],
            'total_mentions': total,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'positive_pct': f"{pos_pct:.1f}%",
            'negative_pct': f"{neg_pct:.1f}%",
            'neutral_pct': f"{neu_pct:.1f}%",
            'avg_confidence': round(avg_conf, 3),
            'sentiment_score': round(sentiment_score, 3)
        })

    return pd.DataFrame(summary_data).sort_values('total_mentions', ascending=False)


def generate_destination_summary(results_tracker):
    """Generate destination-level summary"""
    dest_data = []

    for destination, aspects in results_tracker.destination_aspects.items():
        for aspect_key, sentiments in aspects.items():
            total = sum(sentiments.values())
            if total > 0:
                dest_data.append({
                    'destination': destination,
                    'aspect': aspect_key,
                    'aspect_name': TOURISM_ASPECTS[aspect_key]['name'],
                    'total': total,
                    'positive': sentiments['positive'],
                    'negative': sentiments['negative'],
                    'neutral': sentiments['neutral'],
                    'sentiment_score': round(
                        (sentiments['positive'] - sentiments['negative']) / total, 3
                    )
                })

    df = pd.DataFrame(dest_data)
    if len(df) > 0:
        return df.sort_values(['destination', 'total'], ascending=[True, False])
    return df


def save_results(df_results, df_summary, df_destination, output_dir):
    """Save all results to files"""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_path = os.path.join(output_dir, OUTPUT_RESULTS_FILE)
    summary_path = os.path.join(output_dir, OUTPUT_SUMMARY_FILE)
    dest_path = os.path.join(output_dir, OUTPUT_DESTINATION_FILE)

    print(f"\nSaving detailed results to {results_path}...")
    df_results.to_csv(results_path, index=False, encoding='utf-8')
    print(f"   Saved {len(df_results)} reviews")

    print(f"\nSaving summary to {summary_path}...")
    df_summary.to_csv(summary_path, index=False, encoding='utf-8')
    print(f"   Saved {len(df_summary)} aspect summaries")

    print(f"\nSaving destination analysis to {dest_path}...")
    df_destination.to_csv(dest_path, index=False, encoding='utf-8')
    print(f"   Saved {len(df_destination)} records")

    return results_path, summary_path, dest_path


def generate_documentation(results_tracker, df_summary, df_destination, output_dir):
    """Generate comprehensive documentation"""
    doc_path = os.path.join(output_dir, DOCUMENTATION_FILE)
    print(f"\nGenerating documentation ({doc_path})...")

    total_aspect_mentions = sum(results_tracker.aspect_counts.values())
    avg_aspects_per_review = total_aspect_mentions / max(results_tracker.processed_reviews, 1)

    doc_content = f"""================================================================================
ASPECT-BASED SENTIMENT ANALYSIS (ABSA) DOCUMENTATION
LLM-Based Analysis using Google Gemini
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {MODEL_NAME}
Input File: {INPUT_FILE}

================================================================================
1. METHODOLOGY
================================================================================

LLM-Based ABSA Approach:
------------------------
This analysis uses Google Gemini LLM with few-shot prompting to perform
aspect-based sentiment analysis. Unlike traditional ML models, this approach:

- Understands Indonesian informal language and slang
- Handles context-dependent sentiment (e.g., "ramai" as crowded vs popular)
- Provides evidence/reasoning for each classification
- Detects multiple aspects and sentiments per review

Pipeline Steps:
---------------
1. Batch Processing: Reviews are grouped into batches of {BATCH_SIZE}
2. Few-Shot Prompting: Each batch is analyzed with domain-specific examples
3. Structured Output: JSON format ensures consistent parsing
4. Aggregation: Results are aggregated per aspect and destination

Model Configuration:
--------------------
- Model: {results_tracker.model_name}
- Temperature: 0.1 (for consistent outputs)
- Batch Size: {BATCH_SIZE} reviews per API call
- Rate Limiting: {RATE_LIMIT_DELAY}s between requests

================================================================================
2. RESULTS SUMMARY
================================================================================

Processing Statistics:
----------------------
- Total reviews: {results_tracker.total_reviews:,}
- Successfully processed: {results_tracker.processed_reviews:,}
- Failed: {results_tracker.failed_reviews:,}
- API calls made: {results_tracker.api_calls:,}
- Total processing time: {results_tracker.processing_time:.2f} seconds
- Avg time per review: {results_tracker.processing_time/max(results_tracker.processed_reviews,1):.3f} seconds
- Total aspect mentions: {total_aspect_mentions:,}
- Avg aspects per review: {avg_aspects_per_review:.2f}

Aspect-Level Results:
---------------------
"""

    for _, row in df_summary.iterrows():
        doc_content += f"""
{row['aspect_name']} ({row['aspect']}):
  - Mentions: {row['total_mentions']:,} ({row['total_mentions']/max(results_tracker.processed_reviews,1)*100:.1f}% of reviews)
  - Positive: {row['positive']} ({row['positive_pct']})
  - Negative: {row['negative']} ({row['negative_pct']})
  - Neutral: {row['neutral']} ({row['neutral_pct']})
  - Avg Confidence: {row['avg_confidence']:.3f}
  - Sentiment Score: {row['sentiment_score']:+.3f}
"""

    if len(df_summary) > 0:
        best_aspects = df_summary.nlargest(3, 'sentiment_score')
        worst_aspects = df_summary.nsmallest(3, 'sentiment_score')
        most_mentioned = df_summary.nlargest(5, 'total_mentions')

        doc_content += f"""
================================================================================
3. KEY INSIGHTS
================================================================================

Most Discussed Aspects:
-----------------------
"""
        for i, (_, row) in enumerate(most_mentioned.iterrows(), 1):
            doc_content += f"{i}. {row['aspect_name']}: {row['total_mentions']} mentions\n"

        doc_content += """
Highest Rated Aspects (Strengths):
----------------------------------
"""
        for i, (_, row) in enumerate(best_aspects.iterrows(), 1):
            doc_content += f"{i}. {row['aspect_name']}: {row['sentiment_score']:+.3f} ({row['positive_pct']} positive)\n"

        doc_content += """
Lowest Rated Aspects (Areas for Improvement):
---------------------------------------------
"""
        for i, (_, row) in enumerate(worst_aspects.iterrows(), 1):
            doc_content += f"{i}. {row['aspect_name']}: {row['sentiment_score']:+.3f} ({row['negative_pct']} negative)\n"

    doc_content += f"""

================================================================================
4. COMPARISON WITH PREVIOUS APPROACHES
================================================================================

This LLM-based approach addresses the issues with previous methods:

Previous Zero-Shot (XLM-RoBERTa):
- Problem: 67-81% negative across ALL aspects
- Cause: Model bias and unsuitable hypothesis templates

Previous IndoBERTweet:
- Problem: 74-88% negative across ALL aspects
- Cause: Domain mismatch (trained on Twitter, not tourism)

LLM-Based (Gemini):
- Solution: Few-shot learning with domain-specific examples
- Benefit: Context-aware sentiment understanding
- Benefit: Handles Indonesian informal language properly

================================================================================
5. INTERPRETATION GUIDE
================================================================================

Sentiment Score:
- Range: -1.0 (all negative) to +1.0 (all positive)
- Score > +0.3: Strong positive (strength)
- Score -0.1 to +0.3: Moderately positive
- Score -0.3 to -0.1: Moderately negative
- Score < -0.3: Strong negative (needs improvement)

Confidence Levels:
- HIGH (1.0): Clear, explicit sentiment expression
- MEDIUM (0.7): Implied or moderate sentiment
- LOW (0.4): Ambiguous or weak signal

================================================================================
END OF DOCUMENTATION
================================================================================
"""

    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(doc_content)

    print(f"   Documentation saved to {doc_path}")
    return doc_path


def print_comparison(df_summary):
    """Print comparison with previous results"""
    print("\n" + "=" * 60)
    print("COMPARISON: LLM vs Previous Approaches")
    print("=" * 60)

    print("\n{:<20} {:>12} {:>12} {:>12}".format(
        "Aspect", "LLM Score", "IndoBERT*", "ZeroShot*"
    ))
    print("-" * 60)

    # Previous results (from the files we read earlier)
    previous_indobert = {
        'scenery': -0.815, 'accessibility': -0.700, 'price': -0.634,
        'food': -0.738, 'atmosphere': -0.794, 'facilities': -0.658,
        'crowd': -0.699, 'photo_spot': -0.703, 'cleanliness': -0.755,
        'safety': -0.816, 'historical_value': -0.621, 'service': -0.712
    }

    previous_zeroshot = {
        'accessibility': -0.524, 'scenery': -0.497, 'food': -0.457,
        'atmosphere': -0.569, 'safety': -0.508, 'price': -0.379,
        'facilities': -0.625, 'cleanliness': -0.418, 'photo_spot': -0.322,
        'crowd': -0.628, 'service': -0.391, 'historical_value': -0.289
    }

    for _, row in df_summary.iterrows():
        aspect = row['aspect']
        llm_score = row['sentiment_score']
        indobert = previous_indobert.get(aspect, 0)
        zeroshot = previous_zeroshot.get(aspect, 0)

        print("{:<20} {:>+12.3f} {:>+12.3f} {:>+12.3f}".format(
            row['aspect_name'][:20], llm_score, indobert, zeroshot
        ))

    print("-" * 60)
    print("* Previous scores shown for comparison (mostly negative due to bias)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='LLM-based ABSA using Google Gemini')
    parser.add_argument('--api-key', type=str, help='Google Gemini API key')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    parser.add_argument('--batch-size', type=int, default=5, help='Reviews per API call (default: 5)')
    args = parser.parse_args()

    # Use batch size from args
    batch_size = args.batch_size

    print("=" * 60)
    print("ASPECT-BASED SENTIMENT ANALYSIS (ABSA)")
    print("LLM-Based Analysis using Google Gemini")
    print("=" * 60)

    # Initialize results tracker
    results_tracker = ABSAResults()
    results_tracker.model_name = MODEL_NAME

    # Setup Gemini
    try:
        model = setup_gemini(args.api_key)
        print(f"\nGemini model initialized: {MODEL_NAME}")
    except Exception as e:
        print(f"\nError: {e}")
        return

    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # Load data
    input_path = project_root / INPUT_FILE
    df = load_data(str(input_path), sample_size=args.sample)
    if df is None:
        return

    results_tracker.total_reviews = len(df)

    # Run ABSA analysis
    df_results = run_absa_analysis(df, model, results_tracker)

    # Generate summaries
    print("\nGenerating summaries...")
    df_summary = generate_summary(results_tracker)
    df_destination = generate_destination_summary(results_tracker)

    # Save results
    output_dir = project_root / OUTPUT_DIR
    save_results(df_results, df_summary, df_destination, str(output_dir))

    # Generate documentation
    generate_documentation(results_tracker, df_summary, df_destination, str(output_dir))

    # Print comparison
    print_comparison(df_summary)

    # Print final summary
    print("\n" + "=" * 60)
    print("ABSA ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Total reviews: {results_tracker.total_reviews:,}")
    print(f"Processed: {results_tracker.processed_reviews:,}")
    print(f"Failed: {results_tracker.failed_reviews:,}")
    print(f"API calls: {results_tracker.api_calls:,}")
    print(f"Processing time: {results_tracker.processing_time:.2f} seconds")
    print(f"\nTop 5 aspects by mentions:")
    for _, row in df_summary.head(5).iterrows():
        print(f"  - {row['aspect_name']}: {row['total_mentions']} mentions, "
              f"score: {row['sentiment_score']:+.3f} ({row['positive_pct']} pos)")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
