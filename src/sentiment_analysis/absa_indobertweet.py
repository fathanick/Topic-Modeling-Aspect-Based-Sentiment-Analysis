"""
Aspect-Based Sentiment Analysis (ABSA) using IndoBERTweet

This script performs aspect-based sentiment analysis on Google reviews
using IndoBERTweet - a BERT model pre-trained on Indonesian Twitter data.

For sentiment classification, we use a fine-tuned Indonesian sentiment model
based on IndoBERT/IndoBERTweet architecture.

Input: yogyakarta_tourism_reviews_preprocessed.csv
Output:
    - absa_indobertweet_results.csv (reviews with aspect sentiments)
    - absa_indobertweet_summary.csv (aggregated statistics)
    - absa_indobertweet_by_destination.csv (destination analysis)
    - absa_indobertweet_documentation.txt (documentation)

Requirements:
    pip install transformers torch pandas numpy tqdm

Usage:
    python absa_indobertweet.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
import torch
import re

warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

# ============== Configuration ==============
INPUT_FILE = "yogyakarta_tourism_reviews_preprocessed.csv"
OUTPUT_RESULTS_FILE = "absa_indobertweet_results.csv"
OUTPUT_SUMMARY_FILE = "absa_indobertweet_summary.csv"
OUTPUT_DESTINATION_FILE = "absa_indobertweet_by_destination.csv"
DOCUMENTATION_FILE = "absa_indobertweet_documentation.txt"

# IndoBERTweet-based sentiment model
# Options:
# - "indolem/indobertweet-base-uncased" (base model, needs fine-tuning)
# - "mdhugol/indonesia-bert-sentiment-classification" (fine-tuned for sentiment)
# - "w11wo/indonesian-roberta-base-sentiment-classifier" (RoBERTa-based)
# - "ayameRushia/indobert-lite-base-p1-finetuned-sentiment-analysis-indonlu" (IndoBERT sentiment)

SENTIMENT_MODEL = "mdhugol/indonesia-bert-sentiment-classification"
# Alternative Indonesian sentiment models to try if primary fails
FALLBACK_MODELS = [
    "w11wo/indonesian-roberta-base-sentiment-classifier",
    "ayameRushia/indobert-lite-base-p1-finetuned-sentiment-analysis-indonlu",
    "cahya/bert-base-indonesian-sentiment",
]

# Batch processing configuration
BATCH_SIZE = 16
MAX_LENGTH = 256

# ============== Tourism Aspects Definition ==============
TOURISM_ASPECTS = {
    "cleanliness": {
        "name": "Cleanliness",
        "name_id": "Kebersihan",
        "keywords": ["bersih", "kotor", "sampah", "jorok", "rapi", "terawat", "kumuh", "higenis"],
        "description": "Cleanliness and hygiene of the destination"
    },
    "facilities": {
        "name": "Facilities",
        "name_id": "Fasilitas",
        "keywords": ["toilet", "parkir", "mushola", "kamar mandi", "tempat duduk", "gazebo",
                     "wc", "musholla", "masjid", "fasilitas", "penginapan"],
        "description": "Available facilities and amenities"
    },
    "price": {
        "name": "Price/Value",
        "name_id": "Harga",
        "keywords": ["harga", "murah", "mahal", "tiket", "bayar", "gratis", "worth", "terjangkau",
                     "biaya", "tarif", "htm", "retribusi", "rb", "ribu", "rupiah"],
        "description": "Pricing and value for money"
    },
    "service": {
        "name": "Service",
        "name_id": "Pelayanan",
        "keywords": ["pelayanan", "ramah", "petugas", "guide", "staff", "helpful", "layanan",
                     "pegawai", "penjaga", "tour guide", "pemandu"],
        "description": "Service quality and staff behavior"
    },
    "accessibility": {
        "name": "Accessibility",
        "name_id": "Aksesibilitas",
        "keywords": ["akses", "jalan", "jauh", "dekat", "mudah", "transportasi", "kendaraan",
                     "motor", "mobil", "ojek", "angkot", "bus", "rute", "arah", "lokasi"],
        "description": "Ease of access and transportation"
    },
    "scenery": {
        "name": "Scenery/View",
        "name_id": "Pemandangan",
        "keywords": ["pemandangan", "view", "indah", "cantik", "bagus", "sunset", "sunrise",
                     "panorama", "landscape", "alam", "laut", "gunung", "pantai", "hijau"],
        "description": "Natural beauty and scenery"
    },
    "atmosphere": {
        "name": "Atmosphere",
        "name_id": "Suasana",
        "keywords": ["suasana", "nyaman", "tenang", "ramai", "sepi", "adem", "sejuk", "damai",
                     "asri", "teduh", "rileks", "santai", "cozy"],
        "description": "Ambiance and atmosphere"
    },
    "food": {
        "name": "Food & Beverage",
        "name_id": "Makanan",
        "keywords": ["makanan", "minuman", "makan", "kuliner", "warung", "resto", "enak", "seafood",
                     "ikan", "minum", "kopi", "cafe", "restoran", "kedai", "lezat", "segar"],
        "description": "Food and beverage quality"
    },
    "safety": {
        "name": "Safety",
        "name_id": "Keamanan",
        "keywords": ["aman", "bahaya", "hati-hati", "waspada", "ombak", "licin", "curam",
                     "keamanan", "selamat", "berbahaya", "resiko", "peringatan"],
        "description": "Safety and security"
    },
    "crowd": {
        "name": "Crowd Level",
        "name_id": "Keramaian",
        "keywords": ["ramai", "sepi", "penuh", "antri", "crowded", "pengunjung", "wisatawan",
                     "padat", "kosong", "antrian", "menunggu"],
        "description": "Crowding and visitor density"
    },
    "photo_spot": {
        "name": "Photo Spots",
        "name_id": "Spot Foto",
        "keywords": ["foto", "selfie", "instagramable", "spot", "background", "estetik",
                     "photogenic", "kamera", "jepretan", "dokumentasi", "ig"],
        "description": "Photography opportunities"
    },
    "historical_value": {
        "name": "Historical/Cultural Value",
        "name_id": "Nilai Sejarah",
        "keywords": ["sejarah", "budaya", "candi", "heritage", "kuno", "bersejarah", "museum",
                     "peninggalan", "arkeologi", "tradisi", "adat", "sakral", "religi"],
        "description": "Historical and cultural significance"
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
        self.avg_confidence = defaultdict(list)


def load_data(input_file):
    """Load preprocessed data"""
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"   Loaded {len(df)} reviews")
        return df
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return None


def initialize_sentiment_classifier():
    """Initialize Indonesian sentiment classifier"""
    print(f"\nInitializing IndoBERT sentiment classifier...")

    models_to_try = [SENTIMENT_MODEL] + FALLBACK_MODELS

    for model_name in models_to_try:
        try:
            print(f"   Trying model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Create pipeline
            classifier = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=-1,  # CPU, use 0 for GPU
                max_length=MAX_LENGTH,
                truncation=True
            )

            # Test the classifier
            test_result = classifier("Tempat ini sangat bagus dan indah")
            print(f"   Model loaded successfully: {model_name}")
            print(f"   Test result: {test_result}")

            return classifier, model_name

        except Exception as e:
            print(f"   Failed to load {model_name}: {str(e)[:100]}")
            continue

    print("   All models failed. Using fallback zero-shot approach.")
    return None, None


def detect_aspects(text, aspects_config):
    """Detect aspects mentioned in the review using keyword matching"""
    if not text or pd.isna(text):
        return []

    text_lower = str(text).lower()
    detected = []

    for aspect_key, aspect_info in aspects_config.items():
        for keyword in aspect_info["keywords"]:
            # Use word boundary matching for better accuracy
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text_lower):
                detected.append(aspect_key)
                break

    return detected


def extract_aspect_context(text, aspect_keywords, context_window=50):
    """Extract text context around aspect keywords for better sentiment analysis"""
    if not text:
        return text

    text_lower = text.lower()
    contexts = []

    for keyword in aspect_keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        for match in re.finditer(pattern, text_lower):
            start = max(0, match.start() - context_window)
            end = min(len(text), match.end() + context_window)
            contexts.append(text[start:end])

    if contexts:
        return ' '.join(contexts)
    return text[:MAX_LENGTH]


def normalize_sentiment_label(label, score):
    """Normalize sentiment labels from different models to standard format"""
    label_lower = str(label).lower()

    # Map various label formats to standard positive/negative/neutral
    positive_labels = ['positive', 'positif', 'pos', 'label_2', '2', 'good', 'bagus']
    negative_labels = ['negative', 'negatif', 'neg', 'label_0', '0', 'bad', 'buruk']
    neutral_labels = ['neutral', 'netral', 'label_1', '1', 'mixed']

    if any(pos in label_lower for pos in positive_labels):
        return 'positive', score
    elif any(neg in label_lower for neg in negative_labels):
        return 'negative', score
    elif any(neu in label_lower for neu in neutral_labels):
        return 'neutral', score
    else:
        # If unknown label, use score to determine
        if score > 0.6:
            return 'positive', score
        elif score < 0.4:
            return 'negative', score
        else:
            return 'neutral', score


def analyze_sentiment_batch(classifier, texts, batch_size=16):
    """Analyze sentiment for a batch of texts"""
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            batch_results = classifier(batch)
            for result in batch_results:
                label, score = normalize_sentiment_label(result['label'], result['score'])
                results.append({'sentiment': label, 'confidence': score})
        except Exception as e:
            # Fallback for failed batch
            for _ in batch:
                results.append({'sentiment': 'neutral', 'confidence': 0.5})

    return results


def run_absa_analysis(df, classifier, model_name, aspects_config, results_tracker):
    """Run ABSA analysis on all reviews"""
    print("\n" + "=" * 60)
    print("RUNNING ASPECT-BASED SENTIMENT ANALYSIS")
    print(f"Model: {model_name}")
    print(f"Total reviews: {len(df)}")
    print("=" * 60)

    start_time = datetime.now()
    all_results = []

    # Process each review
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing reviews"):
        original_text = str(row.get('original_text', ''))
        cleaned_text = str(row.get('cleaned_text', ''))

        # Use original text for aspect detection (has more context)
        # Use cleaned text for sentiment (less noise)
        text_for_aspects = original_text if original_text else cleaned_text
        text_for_sentiment = cleaned_text if cleaned_text else original_text

        if not text_for_aspects or pd.isna(text_for_aspects):
            continue

        # Detect aspects
        detected_aspects = detect_aspects(text_for_aspects, aspects_config)

        # Create result record
        result_record = {
            'destination': row['destination'],
            'username': row.get('username', ''),
            'stars': row.get('stars', ''),
            'original_text': original_text[:500],
        }

        aspects_found = []
        sentiments_found = []

        # Analyze sentiment for each detected aspect
        for aspect_key in detected_aspects:
            aspect_info = aspects_config[aspect_key]

            # Extract context around aspect keywords
            context_text = extract_aspect_context(
                text_for_sentiment,
                aspect_info['keywords'],
                context_window=100
            )

            # Get sentiment for this aspect context
            try:
                sent_result = classifier(context_text[:MAX_LENGTH])
                if isinstance(sent_result, list):
                    sent_result = sent_result[0]

                sentiment, confidence = normalize_sentiment_label(
                    sent_result['label'],
                    sent_result['score']
                )
            except Exception as e:
                sentiment, confidence = 'neutral', 0.5

            result_record[f'{aspect_key}_sentiment'] = sentiment
            result_record[f'{aspect_key}_confidence'] = round(confidence, 3)

            aspects_found.append(aspect_key)
            sentiments_found.append(f"{aspect_key}:{sentiment}")

            # Update trackers
            results_tracker.aspect_counts[aspect_key] += 1
            results_tracker.aspect_sentiments[aspect_key][sentiment] += 1
            results_tracker.destination_aspects[row['destination']][aspect_key][sentiment] += 1
            results_tracker.avg_confidence[aspect_key].append(confidence)

        # Fill empty columns for non-detected aspects
        for aspect_key in aspects_config.keys():
            if aspect_key not in detected_aspects:
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
    print(f"Average time per review: {results_tracker.processing_time/max(results_tracker.processed_reviews,1):.3f} seconds")

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
            avg_conf = np.mean(results_tracker.avg_confidence[aspect_key])
            sentiment_score = (positive - negative) / total
        else:
            pos_pct = neg_pct = neu_pct = avg_conf = sentiment_score = 0

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


def save_results(df_results, df_summary, df_destination):
    """Save all results to files"""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    print(f"\nSaving detailed results to {OUTPUT_RESULTS_FILE}...")
    df_results.to_csv(OUTPUT_RESULTS_FILE, index=False, encoding='utf-8')
    print(f"   Saved {len(df_results)} reviews")

    print(f"\nSaving summary to {OUTPUT_SUMMARY_FILE}...")
    df_summary.to_csv(OUTPUT_SUMMARY_FILE, index=False, encoding='utf-8')
    print(f"   Saved {len(df_summary)} aspect summaries")

    print(f"\nSaving destination analysis to {OUTPUT_DESTINATION_FILE}...")
    df_destination.to_csv(OUTPUT_DESTINATION_FILE, index=False, encoding='utf-8')
    print(f"   Saved {len(df_destination)} records")


def generate_documentation(results_tracker, df_summary, df_destination, aspects_config):
    """Generate comprehensive documentation"""
    print(f"\nGenerating documentation ({DOCUMENTATION_FILE})...")

    # Calculate additional statistics
    total_aspect_mentions = sum(results_tracker.aspect_counts.values())
    avg_aspects_per_review = total_aspect_mentions / max(results_tracker.processed_reviews, 1)

    doc_content = f"""================================================================================
ASPECT-BASED SENTIMENT ANALYSIS (ABSA) DOCUMENTATION
IndoBERTweet Analysis of Yogyakarta Tourism Reviews
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
ABSA identifies specific aspects mentioned in reviews and determines the
sentiment expressed toward each aspect separately, providing fine-grained
insights into customer opinions.

Model Selection:
----------------
- Primary Model: {results_tracker.model_name}
- Model Type: Indonesian BERT-based Sentiment Classifier
- Architecture: BERT/IndoBERT fine-tuned on Indonesian sentiment data

Why IndoBERT/IndoBERTweet?
--------------------------
- Pre-trained on Indonesian text (including Twitter/social media)
- Better understanding of Indonesian informal language
- Fine-tuned for sentiment classification task
- Handles Indonesian slang and abbreviations common in reviews

Pipeline Steps:
---------------
1. Aspect Detection:
   - Keyword-based matching with expanded Indonesian vocabulary
   - Word boundary detection for accurate matching
   - Context extraction around detected keywords

2. Sentiment Classification:
   - Extract text context around aspect keywords (100 chars window)
   - Apply IndoBERT sentiment classifier to context
   - Normalize labels to: positive, negative, neutral

3. Aggregation:
   - Aggregate sentiments per aspect across all reviews
   - Calculate sentiment scores per destination
   - Generate comprehensive statistics

================================================================================
2. ASPECT DEFINITIONS
================================================================================

12 Tourism Aspects Analyzed:

"""

    for aspect_key, aspect_info in aspects_config.items():
        doc_content += f"""
{aspect_info['name']} ({aspect_info['name_id']}):
{'-' * 50}
- Description: {aspect_info['description']}
- Keywords ({len(aspect_info['keywords'])}): {', '.join(aspect_info['keywords'][:10])}{'...' if len(aspect_info['keywords']) > 10 else ''}

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
- Avg time per review: {results_tracker.processing_time/max(results_tracker.processed_reviews,1):.3f} seconds
- Total aspect mentions: {total_aspect_mentions:,}
- Avg aspects per review: {avg_aspects_per_review:.2f}

Aspect-Level Results:
---------------------
"""

    for _, row in df_summary.iterrows():
        doc_content += f"""
{row['aspect_name']} ({row['aspect']}):
  - Mentions: {row['total_mentions']:,} ({row['total_mentions']/results_tracker.processed_reviews*100:.1f}% of reviews)
  - Positive: {row['positive']} ({row['positive_pct']})
  - Negative: {row['negative']} ({row['negative_pct']})
  - Neutral: {row['neutral']} ({row['neutral_pct']})
  - Avg Confidence: {row['avg_confidence']:.3f}
  - Sentiment Score: {row['sentiment_score']:+.3f}
"""

    # Identify insights
    if len(df_summary) > 0:
        best_aspects = df_summary.nlargest(3, 'sentiment_score')
        worst_aspects = df_summary.nsmallest(3, 'sentiment_score')
        most_mentioned = df_summary.nlargest(5, 'total_mentions')

        doc_content += f"""
================================================================================
4. KEY INSIGHTS
================================================================================

Most Discussed Aspects:
-----------------------
"""
        for i, (_, row) in enumerate(most_mentioned.iterrows(), 1):
            doc_content += f"{i}. {row['aspect_name']}: {row['total_mentions']} mentions ({row['total_mentions']/results_tracker.processed_reviews*100:.1f}%)\n"

        doc_content += """
Highest Rated Aspects (Strengths):
----------------------------------
"""
        for i, (_, row) in enumerate(best_aspects.iterrows(), 1):
            doc_content += f"{i}. {row['aspect_name']}: {row['sentiment_score']:+.3f} score ({row['positive_pct']} positive)\n"

        doc_content += """
Lowest Rated Aspects (Areas for Improvement):
---------------------------------------------
"""
        for i, (_, row) in enumerate(worst_aspects.iterrows(), 1):
            doc_content += f"{i}. {row['aspect_name']}: {row['sentiment_score']:+.3f} score ({row['negative_pct']} negative)\n"

    doc_content += f"""
================================================================================
5. DESTINATION ANALYSIS
================================================================================

Top 10 Destinations by Number of Aspect Mentions:
-------------------------------------------------
"""

    if len(df_destination) > 0:
        dest_totals = df_destination.groupby('destination')['total'].sum().sort_values(ascending=False)
        for i, (dest, total) in enumerate(dest_totals.head(10).items(), 1):
            dest_data = df_destination[df_destination['destination'] == dest]
            avg_score = dest_data['sentiment_score'].mean()
            doc_content += f"{i}. {dest}: {total} aspect mentions (avg score: {avg_score:+.3f})\n"

        doc_content += """
Best Performing Destinations by Aspect:
---------------------------------------
"""
        for aspect_key in list(aspects_config.keys())[:6]:
            aspect_data = df_destination[df_destination['aspect'] == aspect_key]
            if len(aspect_data) >= 3:
                top = aspect_data.nlargest(1, 'sentiment_score').iloc[0]
                doc_content += f"- {aspects_config[aspect_key]['name']}: {top['destination']} ({top['sentiment_score']:+.3f})\n"

    doc_content += f"""
================================================================================
6. INTERPRETATION GUIDE
================================================================================

Sentiment Score Interpretation:
-------------------------------
- Score = (Positive - Negative) / Total Mentions
- Range: -1.0 (all negative) to +1.0 (all positive)

  Score > +0.3  : Strong positive sentiment (strength)
  Score +0.1 to +0.3: Moderately positive
  Score -0.1 to +0.1: Mixed/Neutral sentiment
  Score -0.3 to -0.1: Moderately negative
  Score < -0.3  : Strong negative sentiment (needs improvement)

Confidence Score:
-----------------
- Range: 0.0 to 1.0
- Higher confidence indicates model certainty
- Scores > 0.7 are considered reliable
- Scores < 0.5 may need manual verification

Practical Applications:
-----------------------
1. Identify strengths to highlight in marketing
2. Prioritize improvements for low-scoring aspects
3. Compare performance across destinations
4. Track sentiment changes over time
5. Target specific aspects for intervention

================================================================================
7. TECHNICAL DETAILS
================================================================================

Model Information:
------------------
- Model: {results_tracker.model_name}
- Max sequence length: {MAX_LENGTH}
- Batch size: {BATCH_SIZE}
- Device: CPU

Libraries Used:
---------------
- transformers: Hugging Face transformers library
- torch: PyTorch backend
- pandas: Data manipulation
- numpy: Numerical operations
- tqdm: Progress tracking

Performance Notes:
------------------
- Total processing time: {results_tracker.processing_time:.2f} seconds
- Average per review: {results_tracker.processing_time/max(results_tracker.processed_reviews,1):.3f} seconds
- For faster processing, use GPU (device=0)

================================================================================
8. LIMITATIONS AND CONSIDERATIONS
================================================================================

1. Keyword-based aspect detection may miss aspects expressed differently
2. Context window extraction may not capture full sentiment context
3. Indonesian informal language may affect accuracy
4. Sarcasm and irony are difficult to detect
5. Model was trained on general Indonesian text, not tourism-specific

Recommendations for Improvement:
--------------------------------
1. Fine-tune model on tourism review dataset
2. Expand keyword lists based on missed aspects
3. Implement aspect-level sentiment fine-tuning
4. Add domain-specific preprocessing
5. Consider ensemble of multiple models

================================================================================
END OF DOCUMENTATION
================================================================================
"""

    with open(DOCUMENTATION_FILE, 'w', encoding='utf-8') as f:
        f.write(doc_content)

    print(f"   Documentation saved to {DOCUMENTATION_FILE}")


def main():
    """Main function"""
    print("=" * 60)
    print("ASPECT-BASED SENTIMENT ANALYSIS (ABSA)")
    print("Using IndoBERTweet / Indonesian BERT Sentiment Model")
    print("=" * 60)

    results_tracker = ABSAResults()

    # 1. Load data
    df = load_data(INPUT_FILE)
    if df is None:
        return

    results_tracker.total_reviews = len(df)

    # 2. Initialize sentiment classifier
    classifier, model_name = initialize_sentiment_classifier()
    if classifier is None:
        print("Failed to initialize classifier. Exiting.")
        return

    results_tracker.model_name = model_name

    # 3. Run ABSA analysis
    df_results = run_absa_analysis(df, classifier, model_name, TOURISM_ASPECTS, results_tracker)

    # 4. Generate summaries
    print("\nGenerating summaries...")
    df_summary = generate_summary(results_tracker, TOURISM_ASPECTS)
    df_destination = generate_destination_summary(results_tracker, TOURISM_ASPECTS)

    # 5. Save results
    save_results(df_results, df_summary, df_destination)

    # 6. Generate documentation
    generate_documentation(results_tracker, df_summary, df_destination, TOURISM_ASPECTS)

    # Print final summary
    print("\n" + "=" * 60)
    print("ABSA ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Model used: {model_name}")
    print(f"Total reviews: {results_tracker.total_reviews:,}")
    print(f"Processed reviews: {results_tracker.processed_reviews:,}")
    print(f"Processing time: {results_tracker.processing_time:.2f} seconds")
    print(f"\nTop 5 aspects by mentions:")
    for _, row in df_summary.head(5).iterrows():
        print(f"  - {row['aspect_name']}: {row['total_mentions']} mentions, "
              f"score: {row['sentiment_score']:+.3f} ({row['positive_pct']} pos)")
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_RESULTS_FILE}")
    print(f"  - {OUTPUT_SUMMARY_FILE}")
    print(f"  - {OUTPUT_DESTINATION_FILE}")
    print(f"  - {DOCUMENTATION_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
