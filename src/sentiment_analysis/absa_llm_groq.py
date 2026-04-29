"""
Aspect-Based Sentiment Analysis (ABSA) using Groq API (FREE)

This script performs ABSA on Indonesian tourism reviews using Groq's
ultra-fast LLM inference with Llama 3.1 or Mixtral models.

Features:
- FREE tier: 30 requests/min, 14,400 requests/day
- Ultra-fast inference (10x faster than GPU)
- High-quality models: Llama 3.1 70B, Mixtral 8x7B
- Few-shot prompting for accurate sentiment

Requirements:
    pip install groq pandas tqdm

Setup:
    1. Get FREE API key from https://console.groq.com/keys
    2. Set environment variable: export GROQ_API_KEY="your-api-key"
    Or pass it directly when running the script

Usage:
    python absa_llm_groq.py --api-key YOUR_API_KEY
    python absa_llm_groq.py --sample 50  # Test with 50 samples
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
    from groq import Groq
except ImportError:
    print("Please install groq: pip install groq")
    exit(1)

# ============== Configuration ==============
INPUT_FILE = "data/processed/yogyakarta_tourism_reviews_preprocessed.csv"
OUTPUT_DIR = "output/absa_llm_groq"
OUTPUT_RESULTS_FILE = "absa_llm_results.csv"
OUTPUT_SUMMARY_FILE = "absa_llm_summary.csv"
OUTPUT_DESTINATION_FILE = "absa_llm_by_destination.csv"
DOCUMENTATION_FILE = "absa_llm_documentation.txt"

# Groq configuration - FREE models
# Options: llama-3.1-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768, gemma2-9b-it
MODEL_NAME = "llama-3.1-8b-instant"  # Fast and good quality
BATCH_SIZE = 10  # Reviews per API call (larger batches = fewer API calls)
RATE_LIMIT_DELAY = 3.0  # Seconds between requests (20 req/min to stay safe)
MAX_RETRIES = 3
MAX_TOKENS = 4096

# ============== Tourism Aspects Definition ==============
TOURISM_ASPECTS = {
    "cleanliness": {"name": "Cleanliness", "name_id": "Kebersihan"},
    "facilities": {"name": "Facilities", "name_id": "Fasilitas"},
    "price": {"name": "Price/Value", "name_id": "Harga"},
    "service": {"name": "Service", "name_id": "Pelayanan"},
    "accessibility": {"name": "Accessibility", "name_id": "Aksesibilitas"},
    "scenery": {"name": "Scenery/View", "name_id": "Pemandangan"},
    "atmosphere": {"name": "Atmosphere", "name_id": "Suasana"},
    "food": {"name": "Food & Beverage", "name_id": "Makanan"},
    "safety": {"name": "Safety", "name_id": "Keamanan"},
    "crowd": {"name": "Crowd Level", "name_id": "Keramaian"},
    "photo_spot": {"name": "Photo Spots", "name_id": "Spot Foto"},
    "historical_value": {"name": "Historical/Cultural Value", "name_id": "Nilai Sejarah"}
}

# ============== Prompts ==============
SYSTEM_PROMPT = """You are an expert Indonesian tourism review sentiment analyzer. Analyze reviews and identify:
1. Aspects mentioned (cleanliness, facilities, price, service, accessibility, scenery, atmosphere, food, safety, crowd, photo_spot, historical_value)
2. Sentiment per aspect: POSITIVE, NEGATIVE, or NEUTRAL
3. Evidence: specific words indicating sentiment

SENTIMENT RULES:
- POSITIVE: bagus, indah, bersih, nyaman, worth it, mantap, keren, enak, ramah, murah, lengkap, recommended, luar biasa, memuaskan
- NEGATIVE: kotor, mahal, jelek, susah, bahaya, kurang, buruk, kecewa, mengecewakan, jauh, sempit, rusak
- NEUTRAL: biasa aja, lumayan, cukup, factual statements

INDONESIAN SLANG:
- gak/nggak/ga = tidak (not)
- banget/bgt = very
- mantap/mantul = excellent
- keren = cool/awesome
- zonk = disappointing

Return ONLY valid JSON array (no markdown):
[{"review_id": 1, "aspects": [{"aspect": "scenery", "sentiment": "positive", "evidence": "indah banget"}], "overall_sentiment": "positive"}]"""

FEW_SHOT_USER = """Examples:

Review 1 (5 stars): "Candi Prambanan nggak pernah gagal bikin kagum. Areanya luas, bersih. Fasilitasnya lengkap. Worth it banget!"
Output: [{"review_id": 1, "aspects": [{"aspect": "scenery", "sentiment": "positive", "evidence": "bikin kagum"}, {"aspect": "cleanliness", "sentiment": "positive", "evidence": "bersih"}, {"aspect": "facilities", "sentiment": "positive", "evidence": "lengkap"}, {"aspect": "price", "sentiment": "positive", "evidence": "worth it"}], "overall_sentiment": "positive"}]

Review 2 (3 stars): "Pemandangannya bagus, tapi parkirnya jauh dan mahal. Toiletnya kurang bersih."
Output: [{"review_id": 2, "aspects": [{"aspect": "scenery", "sentiment": "positive", "evidence": "bagus"}, {"aspect": "accessibility", "sentiment": "negative", "evidence": "jauh"}, {"aspect": "price", "sentiment": "negative", "evidence": "mahal"}, {"aspect": "cleanliness", "sentiment": "negative", "evidence": "kurang bersih"}], "overall_sentiment": "mixed"}]

Review 3 (4 stars): "Petugasnya ramah banget dan helpful. Pelayanannya memuaskan."
Output: [{"review_id": 3, "aspects": [{"aspect": "service", "sentiment": "positive", "evidence": "ramah, helpful, memuaskan"}], "overall_sentiment": "positive"}]

Now analyze these reviews and return JSON array:
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
        self.total_tokens = 0


def setup_groq(api_key=None):
    """Initialize Groq client"""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "No API key provided. Either:\n"
            "1. Pass --api-key argument\n"
            "2. Set GROQ_API_KEY environment variable\n"
            "Get FREE key at: https://console.groq.com/keys"
        )
    return Groq(api_key=key)


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
    prompt = FEW_SHOT_USER
    for i, review in enumerate(reviews_batch, 1):
        text = review.get('original_text', review.get('cleaned_text', ''))[:800]
        stars = review.get('stars', 'N/A')
        prompt += f'\nReview {i} ({stars} stars): "{text}"'
    prompt += "\n\nOutput:"
    return prompt


def extract_json_from_response(response_text):
    """Extract JSON from model response"""
    # Clean response
    text = response_text.strip()

    # Remove markdown code blocks if present
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```', '', text)

    # Find JSON array
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        json_str = match.group(0)
        # Clean up
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try to find individual objects
    objects = []
    for match in re.finditer(r'\{[^{}]*(?:"aspects"\s*:\s*\[[^\]]*\][^{}]*)?\}', text):
        try:
            obj = json.loads(match.group(0))
            if 'aspects' in obj or 'review_id' in obj:
                objects.append(obj)
        except:
            continue

    if objects:
        return objects

    raise ValueError("Could not parse JSON from response")


def analyze_batch(client, reviews_batch, results_tracker, retry_count=0):
    """Analyze a batch of reviews using Groq"""
    prompt = create_batch_prompt(reviews_batch)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.1
        )

        results_tracker.api_calls += 1
        results_tracker.total_tokens += response.usage.total_tokens

        response_text = response.choices[0].message.content
        return extract_json_from_response(response_text)

    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            if retry_count < MAX_RETRIES:
                wait_time = 30 * (retry_count + 1)  # Wait longer for rate limits
                print(f"   Rate limit, waiting {wait_time}s...")
                time.sleep(wait_time)
                return analyze_batch(client, reviews_batch, results_tracker, retry_count + 1)

        if retry_count < MAX_RETRIES:
            print(f"   Retry {retry_count + 1}/{MAX_RETRIES}: {error_msg[:50]}")
            time.sleep(5 * (retry_count + 1))
            return analyze_batch(client, reviews_batch, results_tracker, retry_count + 1)

        print(f"   Failed: {error_msg[:50]}")
        return None


def run_absa_analysis(df, client, results_tracker):
    """Run ABSA analysis on all reviews"""
    print("\n" + "=" * 60)
    print("RUNNING LLM-BASED ABSA WITH GROQ (FREE)")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total reviews: {len(df)}")
    print("=" * 60)

    start_time = datetime.now()
    all_results = []
    reviews_list = df.to_dict('records')
    num_batches = (len(reviews_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in tqdm(range(num_batches), desc="Processing"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(reviews_list))
        batch = reviews_list[start_idx:end_idx]

        batch_results = analyze_batch(client, batch, results_tracker)

        if batch_results is None:
            for review in batch:
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
            for i, review in enumerate(batch):
                analysis = batch_results[i] if i < len(batch_results) else {"aspects": [], "overall_sentiment": "neutral"}
                aspects = analysis.get('aspects', [])

                result_record = {
                    'destination': review['destination'],
                    'username': review.get('username', ''),
                    'stars': review.get('stars', ''),
                    'original_text': str(review.get('original_text', ''))[:500],
                    'analysis_status': 'success'
                }

                aspects_found = []
                sentiments_found = []

                for aspect_data in aspects:
                    aspect_key = aspect_data.get('aspect', '').lower().replace(' ', '_')
                    if aspect_key not in TOURISM_ASPECTS:
                        continue

                    sentiment = aspect_data.get('sentiment', 'neutral').lower()
                    if sentiment not in ['positive', 'negative', 'neutral']:
                        sentiment = 'neutral'

                    evidence = aspect_data.get('evidence', '')

                    result_record[f'{aspect_key}_sentiment'] = sentiment
                    result_record[f'{aspect_key}_evidence'] = str(evidence)[:100]

                    aspects_found.append(aspect_key)
                    sentiments_found.append(f"{aspect_key}:{sentiment}")

                    results_tracker.aspect_counts[aspect_key] += 1
                    results_tracker.aspect_sentiments[aspect_key][sentiment] += 1
                    results_tracker.destination_aspects[review['destination']][aspect_key][sentiment] += 1

                for aspect_key in TOURISM_ASPECTS.keys():
                    if aspect_key not in aspects_found:
                        result_record[f'{aspect_key}_sentiment'] = ''
                        result_record[f'{aspect_key}_evidence'] = ''

                result_record['aspects_detected'] = ', '.join(aspects_found)
                result_record['aspect_sentiments'] = '; '.join(sentiments_found)
                result_record['num_aspects'] = len(aspects_found)
                result_record['overall_sentiment'] = analysis.get('overall_sentiment', 'neutral')

                all_results.append(result_record)
                results_tracker.processed_reviews += 1

        time.sleep(RATE_LIMIT_DELAY)

    results_tracker.processing_time = (datetime.now() - start_time).total_seconds()

    print(f"\nCompleted in {results_tracker.processing_time:.2f} seconds")
    print(f"API calls: {results_tracker.api_calls}, Tokens: {results_tracker.total_tokens:,}")
    print(f"Success: {results_tracker.processed_reviews}, Failed: {results_tracker.failed_reviews}")

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
        else:
            pos_pct = neg_pct = neu_pct = sentiment_score = 0

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
                    'sentiment_score': round((sentiments['positive'] - sentiments['negative']) / total, 3)
                })

    df = pd.DataFrame(dest_data)
    return df.sort_values(['destination', 'total'], ascending=[True, False]) if len(df) > 0 else df


def save_results(df_results, df_summary, df_destination, output_dir):
    """Save all results"""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_path = os.path.join(output_dir, OUTPUT_RESULTS_FILE)
    df_results.to_csv(results_path, index=False, encoding='utf-8')
    print(f"   Saved {len(df_results)} reviews to {OUTPUT_RESULTS_FILE}")

    summary_path = os.path.join(output_dir, OUTPUT_SUMMARY_FILE)
    df_summary.to_csv(summary_path, index=False, encoding='utf-8')
    print(f"   Saved summary to {OUTPUT_SUMMARY_FILE}")

    dest_path = os.path.join(output_dir, OUTPUT_DESTINATION_FILE)
    df_destination.to_csv(dest_path, index=False, encoding='utf-8')
    print(f"   Saved {len(df_destination)} destination records")


def print_comparison(df_summary):
    """Print comparison with previous results"""
    print("\n" + "=" * 60)
    print("COMPARISON: Groq LLM vs Previous Approaches")
    print("=" * 60)

    previous_indobert = {
        'scenery': -0.815, 'accessibility': -0.700, 'price': -0.634,
        'food': -0.738, 'atmosphere': -0.794, 'facilities': -0.658,
        'crowd': -0.699, 'photo_spot': -0.703, 'cleanliness': -0.755,
        'safety': -0.816, 'historical_value': -0.621, 'service': -0.712
    }

    print("\n{:<20} {:>10} {:>10} {:>10}".format("Aspect", "Groq LLM", "IndoBERT", "Diff"))
    print("-" * 55)

    for _, row in df_summary.iterrows():
        aspect = row['aspect']
        llm_score = row['sentiment_score']
        indobert = previous_indobert.get(aspect, 0)
        diff = llm_score - indobert

        print("{:<20} {:>+10.3f} {:>+10.3f} {:>+10.3f}".format(
            row['aspect_name'][:20], llm_score, indobert, diff
        ))

    print("-" * 55)
    print("Positive diff = LLM found more positive sentiment (more accurate)")


def main():
    parser = argparse.ArgumentParser(description='ABSA using Groq (FREE)')
    parser.add_argument('--api-key', type=str, help='Groq API key')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    parser.add_argument('--model', type=str, default="llama-3.1-8b-instant",
                       help='Model: llama-3.1-8b-instant, llama-3.1-70b-versatile, mixtral-8x7b-32768')
    args = parser.parse_args()

    model_to_use = args.model

    print("=" * 60)
    print("ASPECT-BASED SENTIMENT ANALYSIS (ABSA)")
    print("Using Groq API (FREE)")
    print("=" * 60)

    results_tracker = ABSAResults()
    results_tracker.model_name = model_to_use

    try:
        client = setup_groq(args.api_key)
        print(f"\nGroq client ready: {model_to_use}")
    except Exception as e:
        print(f"\nError: {e}")
        return

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    input_path = project_root / INPUT_FILE
    df = load_data(str(input_path), sample_size=args.sample)
    if df is None:
        return

    results_tracker.total_reviews = len(df)

    df_results = run_absa_analysis(df, client, results_tracker)

    print("\nGenerating summaries...")
    df_summary = generate_summary(results_tracker)
    df_destination = generate_destination_summary(results_tracker)

    output_dir = project_root / OUTPUT_DIR
    save_results(df_results, df_summary, df_destination, str(output_dir))

    print_comparison(df_summary)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Model: {model_to_use}")
    print(f"Reviews: {results_tracker.processed_reviews}/{results_tracker.total_reviews}")
    print(f"Time: {results_tracker.processing_time:.1f}s")
    print(f"Cost: $0.00 (FREE)")
    print(f"\nTop aspects:")
    for _, row in df_summary.head(5).iterrows():
        print(f"  {row['aspect_name']}: {row['total_mentions']} mentions, score: {row['sentiment_score']:+.3f}")
    print(f"\nOutput: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
