"""
ABSA Aspect Visualizations
Generate word clouds and supporting visualizations for each of the 12 tourism aspects
"""

import sys
sys.path.insert(0, 'venv/lib/python3.9/site-packages')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import os

# Configuration
INPUT_FILE = "output/absa_llm_groq/absa_llm_results.csv"
SUMMARY_FILE = "output/absa_llm_groq/absa_llm_summary.csv"
OUTPUT_DIR = "visualizations/absa_llm"

# 12 Tourism Aspects with display names
ASPECTS = {
    'scenery': 'Scenery/View',
    'cleanliness': 'Cleanliness',
    'facilities': 'Facilities',
    'price': 'Price/Value',
    'atmosphere': 'Atmosphere',
    'historical_value': 'Historical Value',
    'food': 'Food & Beverage',
    'photo_spot': 'Photo Spots',
    'crowd': 'Crowd Level',
    'service': 'Service',
    'accessibility': 'Accessibility',
    'safety': 'Safety'
}

# Indonesian stop words to filter
STOP_WORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'adalah', 'ini', 'itu',
    'juga', 'ada', 'bisa', 'sangat', 'sekali', 'banget', 'nya', 'ya', 'sih', 'aja',
    'sudah', 'tidak', 'ga', 'gak', 'nggak', 'tapi', 'atau', 'kalau', 'kalo', 'jadi',
    'buat', 'sama', 'lagi', 'pas', 'saat', 'waktu', 'disini', 'kesini', 'sini',
    'tempat', 'tempatnya', 'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on'
}


def load_data():
    """Load and prepare the ABSA results data"""
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} reviews")
    return df


def load_summary():
    """Load the official ABSA summary statistics"""
    summary_df = pd.read_csv(SUMMARY_FILE)
    print(f"Loaded summary with {len(summary_df)} aspects")
    return summary_df


def clean_text(text):
    """Clean and tokenize text for word cloud"""
    if pd.isna(text) or text == '':
        return []

    # Convert to lowercase
    text = str(text).lower()

    # Remove special characters but keep Indonesian words
    text = re.sub(r'[^\w\s]', ' ', text)

    # Tokenize
    words = text.split()

    # Filter stop words and short words
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]

    return words


def get_aspect_data(df, aspect):
    """Extract sentiment and evidence data for a specific aspect"""
    sentiment_col = f"{aspect}_sentiment"
    evidence_col = f"{aspect}_evidence"

    # Get non-empty entries
    mask = df[sentiment_col].notna() & (df[sentiment_col] != '')

    data = {
        'positive': [],
        'negative': [],
        'neutral': [],
        'all_evidence': []
    }

    for _, row in df[mask].iterrows():
        sentiment = row[sentiment_col].lower() if pd.notna(row[sentiment_col]) else ''
        evidence = row[evidence_col] if pd.notna(row[evidence_col]) else ''

        if evidence:
            data['all_evidence'].append(evidence)
            if 'positive' in sentiment:
                data['positive'].append(evidence)
            elif 'negative' in sentiment:
                data['negative'].append(evidence)
            else:
                data['neutral'].append(evidence)

    return data


def create_wordcloud(texts, title, filename, colormap='viridis'):
    """Generate and save a word cloud from text data"""
    if not texts:
        print(f"  No text data for {title}")
        return None

    # Combine all texts and clean
    all_words = []
    for text in texts:
        all_words.extend(clean_text(text))

    if not all_words:
        print(f"  No words after cleaning for {title}")
        return None

    # Create word frequency
    word_freq = Counter(all_words)

    # Generate word cloud
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=100,
        min_font_size=10,
        max_font_size=100,
        random_state=42
    )

    wc.generate_from_frequencies(word_freq)

    # Save
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filename}")
    return word_freq


def create_all_wordclouds(df):
    """Create word clouds for all 12 aspects"""
    print("\n=== Creating Word Clouds per Aspect ===")

    aspect_word_freqs = {}

    for aspect, display_name in ASPECTS.items():
        print(f"\nProcessing: {display_name}")

        data = get_aspect_data(df, aspect)

        # Create combined word cloud (all sentiments)
        word_freq = create_wordcloud(
            data['all_evidence'],
            f"{display_name} - Common Expressions",
            f"wordcloud_{aspect}.png",
            colormap='Blues'
        )

        if word_freq:
            aspect_word_freqs[aspect] = word_freq

        # Create sentiment-specific word clouds if enough data
        if len(data['positive']) >= 10:
            create_wordcloud(
                data['positive'],
                f"{display_name} - Positive Expressions",
                f"wordcloud_{aspect}_positive.png",
                colormap='Greens'
            )

        if len(data['negative']) >= 10:
            create_wordcloud(
                data['negative'],
                f"{display_name} - Negative Expressions",
                f"wordcloud_{aspect}_negative.png",
                colormap='Reds'
            )

    return aspect_word_freqs


def create_sentiment_distribution_chart(summary_df):
    """Create stacked bar chart showing sentiment distribution per aspect"""
    print("\n=== Creating Sentiment Distribution Chart ===")

    sentiment_data = []

    for _, row in summary_df.iterrows():
        aspect = row['aspect']
        if aspect not in ASPECTS:
            continue

        display_name = ASPECTS[aspect]

        # Parse percentage strings (remove % sign)
        pos_pct = float(str(row['positive_pct']).replace('%', ''))
        neg_pct = float(str(row['negative_pct']).replace('%', ''))
        neu_pct = float(str(row['neutral_pct']).replace('%', ''))

        sentiment_data.append({
            'Aspect': display_name,
            'Positive': row['positive'],
            'Negative': row['negative'],
            'Neutral': row['neutral'],
            'Total': row['total_mentions'],
            'Pos%': pos_pct,
            'Neg%': neg_pct,
            'Neu%': neu_pct,
            'Score': row['sentiment_score']
        })

    # Sort by sentiment score
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df = sentiment_df.sort_values('Score', ascending=True)

    # Create horizontal stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = range(len(sentiment_df))

    # Plot stacked bars
    bars_pos = ax.barh(y_pos, sentiment_df['Pos%'], color='#2ecc71', label='Positive')
    bars_neu = ax.barh(y_pos, sentiment_df['Neu%'], left=sentiment_df['Pos%'], color='#95a5a6', label='Neutral')
    bars_neg = ax.barh(y_pos, sentiment_df['Neg%'], left=sentiment_df['Pos%'] + sentiment_df['Neu%'], color='#e74c3c', label='Negative')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sentiment_df['Aspect'])
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_title('Sentiment Distribution by Aspect', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 100)

    # Add total counts as annotations
    for i, (_, row) in enumerate(sentiment_df.iterrows()):
        ax.annotate(f"n={row['Total']}", xy=(102, i), va='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_distribution_by_aspect.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: sentiment_distribution_by_aspect.png")

    return sentiment_df


def create_sentiment_score_chart(summary_df):
    """Create horizontal bar chart showing sentiment scores"""
    print("\n=== Creating Sentiment Score Chart ===")

    scores = []
    for _, row in summary_df.iterrows():
        aspect = row['aspect']
        if aspect not in ASPECTS:
            continue

        display_name = ASPECTS[aspect]

        scores.append({
            'Aspect': display_name,
            'Score': row['sentiment_score'],
            'Total': row['total_mentions']
        })

    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.sort_values('Score', ascending=True)

    # Create bar chart with color gradient
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#e74c3c' if s < 0.3 else '#f39c12' if s < 0.6 else '#2ecc71' for s in scores_df['Score']]

    bars = ax.barh(scores_df['Aspect'], scores_df['Score'], color=colors, edgecolor='white')

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Sentiment Score (Positive - Negative) / Total', fontsize=11)
    ax.set_title('Aspect Sentiment Scores', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.2, 1.0)

    # Add score labels
    for bar, score in zip(bars, scores_df['Score']):
        ax.annotate(f'+{score:.3f}' if score >= 0 else f'{score:.3f}',
                   xy=(score + 0.02, bar.get_y() + bar.get_height()/2),
                   va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_scores_by_aspect.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: sentiment_scores_by_aspect.png")


def create_aspect_cooccurrence_heatmap(df):
    """Create heatmap showing aspect co-occurrence"""
    print("\n=== Creating Aspect Co-occurrence Heatmap ===")

    aspect_list = list(ASPECTS.keys())
    n_aspects = len(aspect_list)

    # Initialize co-occurrence matrix
    cooccur = np.zeros((n_aspects, n_aspects))

    for _, row in df.iterrows():
        # Find which aspects are present in this review
        present = []
        for i, aspect in enumerate(aspect_list):
            sentiment_col = f"{aspect}_sentiment"
            if pd.notna(row[sentiment_col]) and row[sentiment_col] != '':
                present.append(i)

        # Update co-occurrence matrix
        for i in present:
            for j in present:
                cooccur[i, j] += 1

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    display_names = [ASPECTS[a] for a in aspect_list]

    # Normalize by diagonal (self-occurrence)
    cooccur_norm = cooccur.copy()
    for i in range(n_aspects):
        if cooccur[i, i] > 0:
            cooccur_norm[i, :] = cooccur[i, :] / cooccur[i, i]

    sns.heatmap(cooccur_norm,
                xticklabels=display_names,
                yticklabels=display_names,
                cmap='YlOrRd',
                annot=True,
                fmt='.2f',
                ax=ax,
                vmin=0,
                vmax=1)

    ax.set_title('Aspect Co-occurrence Matrix\n(Normalized: How often aspects appear together)',
                fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'aspect_cooccurrence_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: aspect_cooccurrence_heatmap.png")


def create_top_keywords_chart(aspect_word_freqs):
    """Create bar charts showing top keywords per aspect"""
    print("\n=== Creating Top Keywords Charts ===")

    # Create a combined figure with subplots
    n_aspects = len(aspect_word_freqs)
    n_cols = 3
    n_rows = (n_aspects + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for idx, (aspect, word_freq) in enumerate(aspect_word_freqs.items()):
        ax = axes[idx]

        # Get top 10 words
        top_words = word_freq.most_common(10)
        words = [w[0] for w in top_words]
        counts = [w[1] for w in top_words]

        # Create horizontal bar chart
        y_pos = range(len(words))
        ax.barh(y_pos, counts, color=plt.cm.Blues(0.6 + idx * 0.03))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency')
        ax.set_title(ASPECTS[aspect], fontsize=11, fontweight='bold')

    # Hide empty subplots
    for idx in range(len(aspect_word_freqs), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Top Keywords by Aspect', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_keywords_by_aspect.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: top_keywords_by_aspect.png")


def create_mentions_by_aspect(summary_df):
    """Create bar chart showing number of mentions per aspect"""
    print("\n=== Creating Mentions by Aspect Chart ===")

    mentions = []
    for _, row in summary_df.iterrows():
        aspect = row['aspect']
        if aspect not in ASPECTS:
            continue

        display_name = ASPECTS[aspect]
        mentions.append({'Aspect': display_name, 'Mentions': row['total_mentions']})

    mentions_df = pd.DataFrame(mentions)
    mentions_df = mentions_df.sort_values('Mentions', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(mentions_df)))
    bars = ax.barh(mentions_df['Aspect'], mentions_df['Mentions'], color=colors)

    ax.set_xlabel('Number of Mentions', fontsize=11)
    ax.set_title('Aspect Mention Frequency', fontsize=14, fontweight='bold')

    # Add count labels
    for bar, count in zip(bars, mentions_df['Mentions']):
        ax.annotate(f'{count}',
                   xy=(count + 10, bar.get_y() + bar.get_height()/2),
                   va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mentions_by_aspect.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: mentions_by_aspect.png")


def create_summary_report(summary_df, aspect_word_freqs):
    """Generate a text summary of key findings per aspect"""
    print("\n=== Generating Summary Report ===")

    report_lines = ["=" * 70]
    report_lines.append("ASPECT SENTIMENT ANALYSIS - KEY FINDINGS")
    report_lines.append("(Values from official absa_llm_summary.csv)")
    report_lines.append("=" * 70)
    report_lines.append("")

    for _, row in summary_df.iterrows():
        aspect = row['aspect']
        if aspect not in ASPECTS:
            continue

        display_name = ASPECTS[aspect]
        total = row['total_mentions']
        pos = row['positive']
        neg = row['negative']
        neu = row['neutral']
        score = row['sentiment_score']

        report_lines.append(f"## {display_name}")
        report_lines.append(f"   Mentions: {total}")
        report_lines.append(f"   Sentiment: {pos} positive ({row['positive_pct']}), {neg} negative ({row['negative_pct']}), {neu} neutral ({row['neutral_pct']})")
        report_lines.append(f"   Score: +{score:.3f}")

        # Top keywords
        if aspect in aspect_word_freqs:
            top_words = aspect_word_freqs[aspect].most_common(5)
            keywords = ", ".join([f"'{w[0]}' ({w[1]})" for w in top_words])
            report_lines.append(f"   Top Keywords: {keywords}")

        report_lines.append("")

    report_text = "\n".join(report_lines)

    # Save report
    with open(os.path.join(OUTPUT_DIR, 'aspect_analysis_summary.txt'), 'w') as f:
        f.write(report_text)

    print("  Saved: aspect_analysis_summary.txt")
    print(report_text)


def main():
    """Main execution function"""
    print("=" * 70)
    print("ABSA Aspect Visualization Generator")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = load_data()
    summary_df = load_summary()

    # Generate visualizations
    # Word clouds use raw data (for evidence text)
    aspect_word_freqs = create_all_wordclouds(df)

    # Charts use official summary statistics
    create_sentiment_distribution_chart(summary_df)
    create_sentiment_score_chart(summary_df)
    create_aspect_cooccurrence_heatmap(df)  # Co-occurrence needs raw data
    create_top_keywords_chart(aspect_word_freqs)
    create_mentions_by_aspect(summary_df)
    create_summary_report(summary_df, aspect_word_freqs)

    print("\n" + "=" * 70)
    print("All visualizations saved to:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
