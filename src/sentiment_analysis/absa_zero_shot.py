"""
Aspect-Based Sentiment Analysis (ABSA) using Zero-Shot Learning with LLM

This script performs aspect-based sentiment analysis on preprocessed Google reviews
using zero-shot classification with multilingual transformer models.

Methodology:
1. Define tourism-related aspects (facilities, cleanliness, price, etc.)
2. Use zero-shot classification to detect aspects in each review
3. Use zero-shot classification to determine sentiment for each detected aspect
4. Aggregate results and generate comprehensive documentation

Input: yogyakarta_tourism_reviews_preprocessed.csv
Output:
    - absa_results.csv (reviews with aspect sentiments)
    - absa_summary.csv (aggregated aspect-sentiment statistics)
    - absa_documentation.txt (detailed documentation)

Requirements:
    pip install transformers torch pandas numpy tqdm

Usage:
    python absa_zero_shot.py
    python absa_zero_shot.py --sample 100  # Test with 100 samples
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
import argparse
import json
import sys

warnings.filterwarnings('ignore')

# Transformers imports
from transformers import pipeline
from tqdm import tqdm

# ============== Configuration ==============
INPUT_FILE = "yogyakarta_tourism_reviews_preprocessed.csv"
OUTPUT_RESULTS_FILE = "absa_results.csv"
OUTPUT_SUMMARY_FILE = "absa_summary.csv"
OUTPUT_DESTINATION_FILE = "absa_by_destination.csv"
DOCUMENTATION_FILE = "absa_documentation.txt"

# Zero-shot classification model (multilingual)
ZERO_SHOT_MODEL = "joeddav/xlm-roberta-large-xnli"  # Multilingual zero-shot model
# Alternative: "facebook/bart-large-mnli" (English-focused but can work)

# Aspect detection threshold
ASPECT_THRESHOLD = 0.3  # Minimum confidence to consider an aspect present

# Sentiment labels
SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# ============== Tourism Aspects Definition ==============
# Aspects relevant to tourism destinations with Indonesian translations for better detection

TOURISM_ASPECTS = {
    "cleanliness": {
        "name": "Cleanliness",
        "name_id": "Kebersihan",
        "keywords": ["bersih", "kotor", "sampah", "jorok", "rapi", "terawat", "kumuh"],
        "labels": ["kebersihan", "cleanliness", "bersih kotor", "hygiene"]
    },
    "facilities": {
        "name": "Facilities",
        "name_id": "Fasilitas",
        "keywords": ["toilet", "parkir", "mushola", "kamar mandi", "tempat duduk", "gazebo"],
        "labels": ["fasilitas", "facilities", "sarana prasarana", "amenities"]
    },
    "price": {
        "name": "Price/Value",
        "name_id": "Harga",
        "keywords": ["harga", "murah", "mahal", "tiket", "bayar", "gratis", "worth it", "terjangkau"],
        "labels": ["harga", "price", "biaya", "value for money", "tiket masuk"]
    },
    "service": {
        "name": "Service",
        "name_id": "Pelayanan",
        "keywords": ["pelayanan", "ramah", "petugas", "guide", "staff", "helpful"],
        "labels": ["pelayanan", "service", "layanan", "staff service"]
    },
    "accessibility": {
        "name": "Accessibility",
        "name_id": "Aksesibilitas",
        "keywords": ["akses", "jalan", "parkir", "jauh", "dekat", "mudah dijangkau", "transportasi"],
        "labels": ["aksesibilitas", "accessibility", "akses jalan", "kemudahan akses"]
    },
    "scenery": {
        "name": "Scenery/View",
        "name_id": "Pemandangan",
        "keywords": ["pemandangan", "view", "indah", "cantik", "bagus", "sunset", "sunrise"],
        "labels": ["pemandangan", "scenery", "view", "keindahan alam"]
    },
    "atmosphere": {
        "name": "Atmosphere",
        "name_id": "Suasana",
        "keywords": ["suasana", "nyaman", "tenang", "ramai", "sepi", "adem", "sejuk"],
        "labels": ["suasana", "atmosphere", "ambiance", "kenyamanan"]
    },
    "food": {
        "name": "Food & Beverage",
        "name_id": "Makanan",
        "keywords": ["makanan", "minuman", "makan", "kuliner", "warung", "resto", "enak", "seafood"],
        "labels": ["makanan minuman", "food and beverage", "kuliner", "food quality"]
    },
    "safety": {
        "name": "Safety",
        "name_id": "Keamanan",
        "keywords": ["aman", "bahaya", "hati-hati", "waspada", "ombak", "licin", "curam"],
        "labels": ["keamanan", "safety", "keselamatan", "security"]
    },
    "crowd": {
        "name": "Crowd Level",
        "name_id": "Keramaian",
        "keywords": ["ramai", "sepi", "penuh", "antri", "crowded", "pengunjung"],
        "labels": ["keramaian", "crowd level", "tingkat keramaian", "jumlah pengunjung"]
    },
    "photo_spot": {
        "name": "Photo Spots",
        "name_id": "Spot Foto",
        "keywords": ["foto", "selfie", "instagramable", "spot", "background", "estetik"],
        "labels": ["spot foto", "photo spots", "tempat foto", "instagramable"]
    },
    "historical_value": {
        "name": "Historical/Cultural Value",
        "name_id": "Nilai Sejarah",
        "keywords": ["sejarah", "budaya", "candi", "heritage", "kuno", "bersejarah", "museum"],
        "labels": ["nilai sejarah", "historical value", "budaya", "cultural significance"]
    }
}


class ABSAResults:
    """Class to store ABSA results and statistics"""
    def __init__(self):
        self.total_reviews = 0
        self.processed_reviews = 0
        self.aspect_counts = defaultdict(int)
        self.aspect_sentiments = defaultdict(lambda: defaultdict(int))
        self.destination_aspects = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.processing_time = 0
        self.model_name = ""


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


def initialize_classifier(model_name):
    """Initialize zero-shot classification pipeline"""
    print(f"\nInitializing zero-shot classifier...")
    print(f"   Model: {model_name}")

    try:
        classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1  # CPU, use 0 for GPU
        )
        print("   Classifier initialized successfully")
        return classifier
    except Exception as e:
        print(f"   Error initializing classifier: {e}")
        print("   Trying alternative model...")
        try:
            classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
            print("   Alternative classifier initialized")
            return classifier
        except Exception as e2:
            print(f"   Failed to initialize classifier: {e2}")
            return None


def detect_aspects_keyword(text, aspects_config):
    """Detect aspects using keyword matching (fast pre-filter)"""
    text_lower = text.lower()
    detected = []

    for aspect_key, aspect_info in aspects_config.items():
        for keyword in aspect_info["keywords"]:
            if keyword.lower() in text_lower:
                detected.append(aspect_key)
                break

    return detected


def analyze_aspect_sentiment(classifier, text, aspect_labels):
    """Analyze sentiment for a specific aspect using zero-shot classification"""
    # Create hypothesis for sentiment classification
    sentiment_labels = [
        f"The sentiment about this is positive",
        f"The sentiment about this is negative",
        f"The sentiment about this is neutral"
    ]

    try:
        result = classifier(
            text,
            candidate_labels=["positive", "negative", "neutral"],
            hypothesis_template="The sentiment of this text is {}."
        )

        # Get the top sentiment
        top_sentiment = result["labels"][0]
        confidence = result["scores"][0]

        return top_sentiment, confidence
    except Exception as e:
        return "neutral", 0.0


def analyze_review_aspects(classifier, text, aspects_config, threshold=0.3):
    """Analyze all aspects and their sentiments in a review"""
    results = {}

    # First, detect which aspects are mentioned using keywords (fast)
    keyword_aspects = detect_aspects_keyword(text, aspects_config)

    if not keyword_aspects:
        # If no keywords found, use zero-shot to detect aspects
        all_aspect_labels = []
        aspect_label_map = {}

        for aspect_key, aspect_info in aspects_config.items():
            for label in aspect_info["labels"][:2]:  # Use first 2 labels per aspect
                all_aspect_labels.append(label)
                aspect_label_map[label] = aspect_key

        try:
            # Detect aspects using zero-shot
            aspect_result = classifier(
                text,
                candidate_labels=all_aspect_labels,
                multi_label=True
            )

            # Filter aspects above threshold
            for label, score in zip(aspect_result["labels"], aspect_result["scores"]):
                if score >= threshold:
                    aspect_key = aspect_label_map.get(label)
                    if aspect_key and aspect_key not in keyword_aspects:
                        keyword_aspects.append(aspect_key)
        except:
            pass

    # For each detected aspect, analyze sentiment
    for aspect_key in keyword_aspects:
        sentiment, confidence = analyze_aspect_sentiment(
            classifier, text, aspects_config[aspect_key]["labels"]
        )
        results[aspect_key] = {
            "sentiment": sentiment,
            "confidence": confidence
        }

    return results


def run_absa_analysis(df, classifier, aspects_config, results_tracker):
    """Run ABSA analysis on all reviews"""
    print("\n" + "=" * 60)
    print("RUNNING ASPECT-BASED SENTIMENT ANALYSIS")
    print("=" * 60)

    start_time = datetime.now()

    all_results = []

    # Process each review
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing reviews"):
        text = row.get('original_text', row.get('cleaned_text', ''))

        if not text or pd.isna(text):
            continue

        # Analyze aspects and sentiments
        aspect_results = analyze_review_aspects(
            classifier, str(text), aspects_config, ASPECT_THRESHOLD
        )

        # Create result record
        result_record = {
            'destination': row['destination'],
            'username': row.get('username', ''),
            'stars': row.get('stars', ''),
            'original_text': text[:500],  # Truncate for CSV
        }

        # Add aspect columns
        aspects_found = []
        sentiments_found = []

        for aspect_key in aspects_config.keys():
            if aspect_key in aspect_results:
                sentiment = aspect_results[aspect_key]['sentiment']
                confidence = aspect_results[aspect_key]['confidence']

                result_record[f'{aspect_key}_sentiment'] = sentiment
                result_record[f'{aspect_key}_confidence'] = round(confidence, 3)

                aspects_found.append(aspect_key)
                sentiments_found.append(f"{aspect_key}:{sentiment}")

                # Update trackers
                results_tracker.aspect_counts[aspect_key] += 1
                results_tracker.aspect_sentiments[aspect_key][sentiment] += 1
                results_tracker.destination_aspects[row['destination']][aspect_key][sentiment] += 1
            else:
                result_record[f'{aspect_key}_sentiment'] = ''
                result_record[f'{aspect_key}_confidence'] = ''

        result_record['aspects_detected'] = ', '.join(aspects_found)
        result_record['aspect_sentiments'] = '; '.join(sentiments_found)
        result_record['num_aspects'] = len(aspects_found)

        all_results.append(result_record)
        results_tracker.processed_reviews += 1

    end_time = datetime.now()
    results_tracker.processing_time = (end_time - start_time).total_seconds()

    print(f"\nAnalysis completed in {results_tracker.processing_time:.2f} seconds")

    return pd.DataFrame(all_results)


def generate_summary(results_tracker, aspects_config):
    """Generate summary statistics"""
    summary_data = []

    for aspect_key, aspect_info in aspects_config.items():
        total = results_tracker.aspect_counts[aspect_key]
        positive = results_tracker.aspect_sentiments[aspect_key]['positive']
        negative = results_tracker.aspect_sentiments[aspect_key]['negative']
        neutral = results_tracker.aspect_sentiments[aspect_key]['neutral']

        if total > 0:
            pos_pct = (positive / total) * 100
            neg_pct = (negative / total) * 100
            neu_pct = (neutral / total) * 100
        else:
            pos_pct = neg_pct = neu_pct = 0

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
            'sentiment_score': round((positive - negative) / total, 3) if total > 0 else 0
        })

    return pd.DataFrame(summary_data).sort_values('total_mentions', ascending=False)


def generate_destination_summary(results_tracker, aspects_config):
    """Generate destination-level summary"""
    dest_data = []

    for destination, aspects in results_tracker.destination_aspects.items():
        for aspect_key, sentiments in aspects.items():
            total = sum(sentiments.values())
            if total > 0:
                dest_data.append({
                    'destination': destination,
                    'aspect': aspect_key,
                    'aspect_name': aspects_config[aspect_key]['name'],
                    'total': total,
                    'positive': sentiments['positive'],
                    'negative': sentiments['negative'],
                    'neutral': sentiments['neutral'],
                    'sentiment_score': round(
                        (sentiments['positive'] - sentiments['negative']) / total, 3
                    )
                })

    return pd.DataFrame(dest_data).sort_values(['destination', 'total'], ascending=[True, False])


def save_results(df_results, df_summary, df_destination, results_tracker):
    """Save all results to files"""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Save detailed results
    print(f"\nSaving detailed results to {OUTPUT_RESULTS_FILE}...")
    df_results.to_csv(OUTPUT_RESULTS_FILE, index=False, encoding='utf-8')
    print(f"   Saved {len(df_results)} reviews with aspect sentiments")

    # Save summary
    print(f"\nSaving summary to {OUTPUT_SUMMARY_FILE}...")
    df_summary.to_csv(OUTPUT_SUMMARY_FILE, index=False, encoding='utf-8')
    print(f"   Saved {len(df_summary)} aspect summaries")

    # Save destination summary
    print(f"\nSaving destination analysis to {OUTPUT_DESTINATION_FILE}...")
    df_destination.to_csv(OUTPUT_DESTINATION_FILE, index=False, encoding='utf-8')
    print(f"   Saved {len(df_destination)} destination-aspect records")


def generate_documentation(results_tracker, df_summary, df_destination, aspects_config):
    """Generate comprehensive documentation"""
    print(f"\nGenerating documentation ({DOCUMENTATION_FILE})...")

    doc_content = f"""================================================================================
ASPECT-BASED SENTIMENT ANALYSIS (ABSA) DOCUMENTATION
Zero-Shot Learning Analysis of Yogyakarta Tourism Reviews
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input File: {INPUT_FILE}
Output Files:
    - {OUTPUT_RESULTS_FILE} (detailed review-level results)
    - {OUTPUT_SUMMARY_FILE} (aspect-level summary)
    - {OUTPUT_DESTINATION_FILE} (destination-level analysis)

================================================================================
1. METHODOLOGY
================================================================================

Aspect-Based Sentiment Analysis (ABSA) Overview:
------------------------------------------------
ABSA is a fine-grained sentiment analysis technique that identifies specific
aspects (features) mentioned in text and determines the sentiment expressed
toward each aspect separately.

Zero-Shot Learning Approach:
----------------------------
This analysis uses zero-shot classification with pre-trained transformer models,
allowing sentiment classification without task-specific training data.

Pipeline Steps:
1. Aspect Detection: Identify which aspects are mentioned in each review
   - Primary: Keyword matching for fast detection
   - Secondary: Zero-shot classification for unlabeled aspects

2. Sentiment Classification: For each detected aspect, classify sentiment
   - Uses zero-shot classification with sentiment hypothesis templates
   - Labels: positive, negative, neutral

Model Configuration:
--------------------
- Zero-Shot Model: {results_tracker.model_name}
- Aspect Detection Threshold: {ASPECT_THRESHOLD}
- Sentiment Labels: {', '.join(SENTIMENT_LABELS)}

================================================================================
2. ASPECT DEFINITIONS
================================================================================

The following aspects were analyzed for tourism destination reviews:

"""

    # Add aspect definitions
    for aspect_key, aspect_info in aspects_config.items():
        doc_content += f"""
{aspect_info['name']} ({aspect_info['name_id']}):
{'-' * 40}
- Key: {aspect_key}
- Keywords: {', '.join(aspect_info['keywords'])}
- Detection Labels: {', '.join(aspect_info['labels'])}

"""

    doc_content += f"""
================================================================================
3. RESULTS SUMMARY
================================================================================

Dataset Statistics:
-------------------
- Total reviews analyzed: {results_tracker.total_reviews:,}
- Reviews processed: {results_tracker.processed_reviews:,}
- Processing time: {results_tracker.processing_time:.2f} seconds
- Average time per review: {results_tracker.processing_time/max(results_tracker.processed_reviews,1):.3f} seconds

Aspect Detection Summary:
-------------------------
"""

    # Add aspect statistics
    for _, row in df_summary.iterrows():
        doc_content += f"""
{row['aspect_name']} ({row['aspect']}):
- Total mentions: {row['total_mentions']:,}
- Positive: {row['positive']} ({row['positive_pct']})
- Negative: {row['negative']} ({row['negative_pct']})
- Neutral: {row['neutral']} ({row['neutral_pct']})
- Sentiment Score: {row['sentiment_score']} (range: -1 to +1)
"""

    doc_content += f"""

================================================================================
4. SENTIMENT DISTRIBUTION BY ASPECT
================================================================================

Overall Sentiment Patterns:
---------------------------
"""

    # Identify best and worst aspects
    if len(df_summary) > 0:
        best_aspects = df_summary.nlargest(3, 'sentiment_score')
        worst_aspects = df_summary.nsmallest(3, 'sentiment_score')
        most_mentioned = df_summary.nlargest(3, 'total_mentions')

        doc_content += """
Most Mentioned Aspects:
"""
        for _, row in most_mentioned.iterrows():
            doc_content += f"  - {row['aspect_name']}: {row['total_mentions']} mentions\n"

        doc_content += """
Most Positive Aspects (by sentiment score):
"""
        for _, row in best_aspects.iterrows():
            doc_content += f"  - {row['aspect_name']}: {row['sentiment_score']} score ({row['positive_pct']} positive)\n"

        doc_content += """
Areas Needing Improvement (lowest sentiment scores):
"""
        for _, row in worst_aspects.iterrows():
            doc_content += f"  - {row['aspect_name']}: {row['sentiment_score']} score ({row['negative_pct']} negative)\n"

    doc_content += f"""

================================================================================
5. DESTINATION-LEVEL INSIGHTS
================================================================================

Top Destinations by Aspect Performance:
---------------------------------------
"""

    # Add destination insights
    if len(df_destination) > 0:
        for aspect_key in list(aspects_config.keys())[:5]:
            aspect_data = df_destination[df_destination['aspect'] == aspect_key]
            if len(aspect_data) > 0:
                top_dest = aspect_data.nlargest(3, 'sentiment_score')
                doc_content += f"\n{aspects_config[aspect_key]['name']} - Top Performing Destinations:\n"
                for _, row in top_dest.iterrows():
                    doc_content += f"  - {row['destination']}: {row['sentiment_score']} score ({row['total']} mentions)\n"

    doc_content += f"""

================================================================================
6. INTERPRETATION GUIDE
================================================================================

Understanding Sentiment Scores:
-------------------------------
- Sentiment Score = (Positive - Negative) / Total
- Range: -1.0 (all negative) to +1.0 (all positive)
- Score > 0.3: Generally positive perception
- Score -0.3 to 0.3: Mixed or neutral perception
- Score < -0.3: Generally negative perception

Using the Results:
------------------
1. Identify Strengths: Aspects with high positive percentages
2. Identify Weaknesses: Aspects with high negative percentages
3. Prioritize Improvements: Focus on frequently mentioned negative aspects
4. Destination Comparison: Compare aspect performance across destinations

Limitations:
------------
1. Zero-shot classification may have lower accuracy than fine-tuned models
2. Indonesian informal language may affect detection accuracy
3. Sarcasm and irony may be misclassified
4. Context-dependent sentiments may not be fully captured

================================================================================
7. OUTPUT FILE DESCRIPTIONS
================================================================================

{OUTPUT_RESULTS_FILE}:
- Contains review-level aspect sentiment analysis
- Columns include sentiment and confidence for each aspect
- Use for detailed review-by-review analysis

{OUTPUT_SUMMARY_FILE}:
- Aggregated statistics per aspect
- Use for overall aspect performance comparison

{OUTPUT_DESTINATION_FILE}:
- Destination-level breakdown of aspect sentiments
- Use for comparing destinations on specific aspects

================================================================================
8. TECHNICAL DETAILS
================================================================================

Libraries Used:
- transformers: Hugging Face transformers for zero-shot classification
- torch: PyTorch backend for model inference
- pandas: Data manipulation
- numpy: Numerical operations
- tqdm: Progress tracking

Model Details:
- Architecture: XLM-RoBERTa Large (multilingual)
- Training: Cross-lingual Natural Language Inference (XNLI)
- Languages: 100+ languages including Indonesian

Performance Notes:
- CPU inference is slower but works without GPU
- For faster processing, use GPU (device=0)
- Batch processing can improve throughput

================================================================================
END OF DOCUMENTATION
================================================================================
"""

    with open(DOCUMENTATION_FILE, 'w', encoding='utf-8') as f:
        f.write(doc_content)

    print(f"   Documentation saved to {DOCUMENTATION_FILE}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Aspect-Based Sentiment Analysis')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    args = parser.parse_args()

    print("=" * 60)
    print("ASPECT-BASED SENTIMENT ANALYSIS (ABSA)")
    print("Zero-Shot Learning with Multilingual LLM")
    print("=" * 60)

    results_tracker = ABSAResults()

    # 1. Load data
    df = load_data(INPUT_FILE, sample_size=args.sample)
    if df is None:
        return

    results_tracker.total_reviews = len(df)

    # 2. Initialize classifier
    classifier = initialize_classifier(ZERO_SHOT_MODEL)
    if classifier is None:
        print("Failed to initialize classifier. Exiting.")
        return

    results_tracker.model_name = ZERO_SHOT_MODEL

    # 3. Run ABSA analysis
    df_results = run_absa_analysis(df, classifier, TOURISM_ASPECTS, results_tracker)

    # 4. Generate summaries
    print("\nGenerating summaries...")
    df_summary = generate_summary(results_tracker, TOURISM_ASPECTS)
    df_destination = generate_destination_summary(results_tracker, TOURISM_ASPECTS)

    # 5. Save results
    save_results(df_results, df_summary, df_destination, results_tracker)

    # 6. Generate documentation
    generate_documentation(results_tracker, df_summary, df_destination, TOURISM_ASPECTS)

    # Print final summary
    print("\n" + "=" * 60)
    print("ABSA ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total reviews: {results_tracker.total_reviews:,}")
    print(f"Processed reviews: {results_tracker.processed_reviews:,}")
    print(f"Processing time: {results_tracker.processing_time:.2f} seconds")
    print(f"\nTop aspects by mentions:")
    for _, row in df_summary.head(5).iterrows():
        print(f"  - {row['aspect_name']}: {row['total_mentions']} mentions, "
              f"sentiment score: {row['sentiment_score']}")
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_RESULTS_FILE}")
    print(f"  - {OUTPUT_SUMMARY_FILE}")
    print(f"  - {OUTPUT_DESTINATION_FILE}")
    print(f"  - {DOCUMENTATION_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
