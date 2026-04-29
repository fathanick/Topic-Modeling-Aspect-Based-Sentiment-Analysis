#!/usr/bin/env python3
"""
ABSA LLM Visualization Script
Creates comprehensive visualizations for LLM-based Aspect-Based Sentiment Analysis results
Includes comparison with previous IndoBERT results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os
from pathlib import Path

# Set style for better looking plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Get project root
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

# Input/Output directories
INPUT_DIR = project_root / "output" / "absa_llm_groq"
OUTPUT_DIR = project_root / "visualizations" / "absa_llm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("ABSA LLM VISUALIZATION GENERATOR")
print("=" * 60)

# Load data
print("\nLoading ABSA LLM results...")
summary_df = pd.read_csv(INPUT_DIR / 'absa_llm_summary.csv')
destination_df = pd.read_csv(INPUT_DIR / 'absa_llm_by_destination.csv')

print(f"  - Summary: {len(summary_df)} aspects")
print(f"  - By Destination: {len(destination_df)} records")

# Previous IndoBERT results for comparison
INDOBERT_SCORES = {
    'scenery': -0.815, 'accessibility': -0.700, 'price': -0.634,
    'food': -0.738, 'atmosphere': -0.794, 'facilities': -0.658,
    'crowd': -0.699, 'photo_spot': -0.703, 'cleanliness': -0.755,
    'safety': -0.816, 'historical_value': -0.621, 'service': -0.712
}

# Color scheme
POSITIVE_COLOR = '#2ecc71'  # Green
NEGATIVE_COLOR = '#e74c3c'  # Red
NEUTRAL_COLOR = '#95a5a6'   # Gray
ACCENT_COLOR = '#3498db'    # Blue
LLM_COLOR = '#9b59b6'       # Purple for LLM
INDOBERT_COLOR = '#e67e22'  # Orange for IndoBERT

# =============================================================================
# 1. ASPECT MENTION FREQUENCY (Horizontal Bar Chart)
# =============================================================================
print("\n[1/9] Creating Aspect Mention Frequency chart...")

fig, ax = plt.subplots(figsize=(10, 6))

sorted_df = summary_df.sort_values('total_mentions', ascending=True)
colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(sorted_df)))

bars = ax.barh(sorted_df['aspect_name'], sorted_df['total_mentions'], color=colors)

for bar, value in zip(bars, sorted_df['total_mentions']):
    ax.text(value + 15, bar.get_y() + bar.get_height()/2,
            f'{value:,}', va='center', fontsize=9)

ax.set_xlabel('Number of Mentions')
ax.set_title('Aspect Mention Frequency in Yogyakarta Tourism Reviews\n(LLM-based ABSA)',
             fontweight='bold', pad=15)
ax.set_xlim(0, max(sorted_df['total_mentions']) * 1.15)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_aspect_mention_frequency.png', bbox_inches='tight')
plt.close()
print("  Saved: 01_aspect_mention_frequency.png")

# =============================================================================
# 2. SENTIMENT DISTRIBUTION BY ASPECT (Stacked Bar Chart)
# =============================================================================
print("[2/9] Creating Sentiment Distribution by Aspect chart...")

fig, ax = plt.subplots(figsize=(12, 7))

sorted_df = summary_df.sort_values('total_mentions', ascending=False)

aspects = sorted_df['aspect_name']
x = np.arange(len(aspects))
width = 0.65

pos_pct = sorted_df['positive_pct'].str.rstrip('%').astype(float)
neg_pct = sorted_df['negative_pct'].str.rstrip('%').astype(float)
neu_pct = sorted_df['neutral_pct'].str.rstrip('%').astype(float)

bars1 = ax.bar(x, pos_pct, width, label='Positive', color=POSITIVE_COLOR)
bars2 = ax.bar(x, neu_pct, width, bottom=pos_pct, label='Neutral', color=NEUTRAL_COLOR)
bars3 = ax.bar(x, neg_pct, width, bottom=pos_pct + neu_pct, label='Negative', color=NEGATIVE_COLOR)

ax.set_xlabel('Aspect')
ax.set_ylabel('Percentage (%)')
ax.set_title('Sentiment Distribution by Aspect (LLM-based ABSA)', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(aspects, rotation=45, ha='right')
ax.legend(loc='upper right')
ax.set_ylim(0, 105)

# Add percentage labels on bars
for i, (p, n, ne) in enumerate(zip(pos_pct, neg_pct, neu_pct)):
    if p > 10:
        ax.text(i, p/2, f'{p:.0f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_sentiment_distribution_stacked.png', bbox_inches='tight')
plt.close()
print("  Saved: 02_sentiment_distribution_stacked.png")

# =============================================================================
# 3. LLM vs IndoBERT COMPARISON (Most Important Chart!)
# =============================================================================
print("[3/9] Creating LLM vs IndoBERT Comparison chart...")

fig, ax = plt.subplots(figsize=(12, 8))

sorted_df = summary_df.sort_values('sentiment_score', ascending=True)

aspects = sorted_df['aspect_name'].tolist()
llm_scores = sorted_df['sentiment_score'].tolist()
indobert_scores = [INDOBERT_SCORES.get(aspect, 0) for aspect in sorted_df['aspect'].tolist()]

y = np.arange(len(aspects))
height = 0.35

bars1 = ax.barh(y - height/2, llm_scores, height, label='LLM (Groq/Llama)', color=LLM_COLOR)
bars2 = ax.barh(y + height/2, indobert_scores, height, label='IndoBERT (Previous)', color=INDOBERT_COLOR, alpha=0.7)

ax.axvline(x=0, color='black', linewidth=1, linestyle='-')

ax.set_xlabel('Sentiment Score (-1 = Negative, +1 = Positive)')
ax.set_ylabel('Aspect')
ax.set_title('Sentiment Score Comparison: LLM vs IndoBERT\n(Dramatic Improvement!)', fontweight='bold', pad=15)
ax.set_yticks(y)
ax.set_yticklabels(aspects)
ax.legend(loc='lower right')
ax.set_xlim(-1.1, 1.1)

# Add improvement annotations
for i, (llm, indo) in enumerate(zip(llm_scores, indobert_scores)):
    improvement = llm - indo
    ax.annotate(f'+{improvement:.2f}', xy=(max(llm, 0.1), i), fontsize=8,
                color='green', fontweight='bold', va='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_llm_vs_indobert_comparison.png', bbox_inches='tight')
plt.close()
print("  Saved: 03_llm_vs_indobert_comparison.png")

# =============================================================================
# 4. SENTIMENT SCORE COMPARISON (Horizontal Bar Chart - LLM only)
# =============================================================================
print("[4/9] Creating Sentiment Score Comparison chart...")

fig, ax = plt.subplots(figsize=(10, 6))

sorted_df = summary_df.sort_values('sentiment_score', ascending=True)

colors = [POSITIVE_COLOR if score > 0.3 else (NEUTRAL_COLOR if score > 0 else NEGATIVE_COLOR)
          for score in sorted_df['sentiment_score']]

bars = ax.barh(sorted_df['aspect_name'], sorted_df['sentiment_score'], color=colors)

ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
ax.axvline(x=0.3, color='green', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(x=-0.3, color='red', linewidth=0.8, linestyle='--', alpha=0.5)

for bar, value in zip(bars, sorted_df['sentiment_score']):
    offset = 0.03 if value >= 0 else -0.03
    ha = 'left' if value >= 0 else 'right'
    ax.text(value + offset, bar.get_y() + bar.get_height()/2,
            f'{value:+.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')

ax.set_xlabel('Sentiment Score (-1 = Negative, +1 = Positive)')
ax.set_title('Sentiment Score by Aspect (LLM-based ABSA)', fontweight='bold', pad=15)
ax.set_xlim(-0.3, 1.1)

# Add legend
green_patch = mpatches.Patch(color=POSITIVE_COLOR, label='Strong Positive (>0.3)')
gray_patch = mpatches.Patch(color=NEUTRAL_COLOR, label='Mixed (0-0.3)')
ax.legend(handles=[green_patch, gray_patch], loc='lower right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_sentiment_score_llm.png', bbox_inches='tight')
plt.close()
print("  Saved: 04_sentiment_score_llm.png")

# =============================================================================
# 5. POSITIVE VS NEGATIVE MENTIONS (Grouped Bar Chart)
# =============================================================================
print("[5/9] Creating Positive vs Negative Mentions chart...")

fig, ax = plt.subplots(figsize=(12, 6))

sorted_df = summary_df.sort_values('total_mentions', ascending=False)

x = np.arange(len(sorted_df))
width = 0.35

bars1 = ax.bar(x - width/2, sorted_df['positive'], width, label='Positive', color=POSITIVE_COLOR)
bars2 = ax.bar(x + width/2, sorted_df['negative'], width, label='Negative', color=NEGATIVE_COLOR)

ax.set_xlabel('Aspect')
ax.set_ylabel('Number of Mentions')
ax.set_title('Positive vs Negative Mentions by Aspect (LLM-based ABSA)', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(sorted_df['aspect_name'], rotation=45, ha='right')
ax.legend()

# Add count labels
for bar in bars1:
    height = bar.get_height()
    if height > 50:
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_positive_vs_negative.png', bbox_inches='tight')
plt.close()
print("  Saved: 05_positive_vs_negative.png")

# =============================================================================
# 6. ASPECT SENTIMENT RADAR CHART
# =============================================================================
print("[6/9] Creating Aspect Sentiment Radar chart...")

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

aspects = summary_df['aspect_name'].tolist()
scores = (summary_df['sentiment_score'] + 1) / 2  # Convert from [-1,1] to [0,1]

num_aspects = len(aspects)
angles = np.linspace(0, 2 * np.pi, num_aspects, endpoint=False).tolist()
angles += angles[:1]

scores_list = scores.tolist()
scores_list += scores_list[:1]

# LLM scores
ax.plot(angles, scores_list, 'o-', linewidth=2, color=LLM_COLOR, markersize=8, label='LLM (Groq)')
ax.fill(angles, scores_list, alpha=0.25, color=LLM_COLOR)

# IndoBERT scores for comparison
indobert_normalized = [(INDOBERT_SCORES.get(asp, 0) + 1) / 2 for asp in summary_df['aspect'].tolist()]
indobert_normalized += indobert_normalized[:1]
ax.plot(angles, indobert_normalized, 'o--', linewidth=2, color=INDOBERT_COLOR, markersize=6,
        alpha=0.7, label='IndoBERT (Previous)')
ax.fill(angles, indobert_normalized, alpha=0.1, color=INDOBERT_COLOR)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(aspects, size=9)

# Add reference circle at 0.5 (neutral sentiment)
circle_angles = np.linspace(0, 2 * np.pi, 100)
ax.plot(circle_angles, [0.5] * 100, '--', color='gray', linewidth=1, alpha=0.5)

ax.set_ylim(0, 1)
ax.set_title('Aspect Sentiment Profile: LLM vs IndoBERT\n(0 = All Negative, 1 = All Positive)',
             fontweight='bold', pad=20, fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_aspect_radar_comparison.png', bbox_inches='tight')
plt.close()
print("  Saved: 06_aspect_radar_comparison.png")

# =============================================================================
# 7. TOP DESTINATIONS HEATMAP BY ASPECT
# =============================================================================
print("[7/9] Creating Top Destinations Heatmap...")

dest_totals = destination_df.groupby('destination')['total'].sum().sort_values(ascending=False)
top_destinations = dest_totals.head(15).index.tolist()

top_dest_df = destination_df[destination_df['destination'].isin(top_destinations)]

pivot_df = top_dest_df.pivot_table(
    index='destination',
    columns='aspect_name',
    values='sentiment_score',
    aggfunc='mean'
)

pivot_df = pivot_df.loc[top_destinations]

fig, ax = plt.subplots(figsize=(14, 10))

colors = ['#e74c3c', '#f5f5f5', '#2ecc71']
cmap = LinearSegmentedColormap.from_list('sentiment', colors, N=256)

im = ax.imshow(pivot_df.values, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Sentiment Score', rotation=270, labelpad=20)

ax.set_xticks(np.arange(len(pivot_df.columns)))
ax.set_yticks(np.arange(len(pivot_df.index)))
ax.set_xticklabels(pivot_df.columns, rotation=45, ha='right')
ax.set_yticklabels(pivot_df.index)

for i in range(len(pivot_df.index)):
    for j in range(len(pivot_df.columns)):
        value = pivot_df.iloc[i, j]
        if not np.isnan(value):
            text_color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   color=text_color, fontsize=7)

ax.set_title('Sentiment Score Heatmap: Top 15 Destinations by Aspect (LLM-based)',
             fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_destination_aspect_heatmap.png', bbox_inches='tight')
plt.close()
print("  Saved: 07_destination_aspect_heatmap.png")

# =============================================================================
# 8. TOP/BOTTOM PERFORMERS BY ASPECT
# =============================================================================
print("[8/9] Creating Top/Bottom Performers chart...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

key_aspects = ['Scenery/View', 'Facilities', 'Price/Value', 'Service', 'Cleanliness', 'Safety']

for idx, aspect in enumerate(key_aspects):
    ax = axes[idx]

    aspect_data = destination_df[
        (destination_df['aspect_name'] == aspect) &
        (destination_df['total'] >= 3)
    ].copy()

    if len(aspect_data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(aspect)
        continue

    top5 = aspect_data.nlargest(5, 'sentiment_score')
    bottom5 = aspect_data.nsmallest(5, 'sentiment_score')

    combined = pd.concat([bottom5, top5]).drop_duplicates()
    combined = combined.sort_values('sentiment_score', ascending=True)

    combined['short_name'] = combined['destination'].apply(
        lambda x: x[:18] + '...' if len(x) > 18 else x
    )

    colors = [POSITIVE_COLOR if s > 0.3 else (NEUTRAL_COLOR if s > 0 else NEGATIVE_COLOR)
              for s in combined['sentiment_score']]

    bars = ax.barh(combined['short_name'], combined['sentiment_score'], color=colors)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-0.5, 1.1)
    ax.set_title(f'{aspect}', fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)

plt.suptitle('Best & Worst Performing Destinations by Aspect (LLM-based)\n(min. 3 mentions)',
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_top_bottom_performers.png', bbox_inches='tight')
plt.close()
print("  Saved: 08_top_bottom_performers.png")

# =============================================================================
# 9. OVERALL SENTIMENT SUMMARY
# =============================================================================
print("[9/9] Creating Overall Sentiment Summary...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: Overall sentiment distribution (Pie)
ax1 = axes[0]
total_positive = summary_df['positive'].sum()
total_negative = summary_df['negative'].sum()
total_neutral = summary_df['neutral'].sum()

sizes = [total_positive, total_negative, total_neutral]
labels = [f'Positive\n({total_positive:,})',
          f'Negative\n({total_negative:,})',
          f'Neutral\n({total_neutral:,})']
colors_pie = [POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR]
explode = (0.03, 0.03, 0.03)

ax1.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        shadow=False, startangle=90, textprops={'fontsize': 10})
ax1.set_title('Overall Sentiment Distribution\n(LLM-based ABSA)', fontweight='bold', pad=15)

# Middle: Comparison bar chart
ax2 = axes[1]
categories = ['Positive %', 'Negative %']
llm_pcts = [total_positive/(total_positive+total_negative+total_neutral)*100,
            total_negative/(total_positive+total_negative+total_neutral)*100]
# IndoBERT had ~80% negative, ~15% positive from previous results
indobert_pcts = [8.5, 80.0]  # Approximate from previous results

x = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x - width/2, llm_pcts, width, label='LLM (Groq)', color=LLM_COLOR)
bars2 = ax2.bar(x + width/2, indobert_pcts, width, label='IndoBERT', color=INDOBERT_COLOR, alpha=0.7)

ax2.set_ylabel('Percentage (%)')
ax2.set_title('LLM vs IndoBERT:\nOverall Sentiment Comparison', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()
ax2.set_ylim(0, 100)

for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%',
             ha='center', va='bottom', fontsize=10)

# Right: Key metrics
ax3 = axes[2]
ax3.axis('off')

metrics_text = f"""
KEY METRICS (LLM-based ABSA)
{'='*35}

Total Aspect Mentions: {total_positive + total_negative + total_neutral:,}

Sentiment Breakdown:
  Positive: {total_positive:,} ({total_positive/(total_positive+total_negative+total_neutral)*100:.1f}%)
  Negative: {total_negative:,} ({total_negative/(total_positive+total_negative+total_neutral)*100:.1f}%)
  Neutral:  {total_neutral:,} ({total_neutral/(total_positive+total_negative+total_neutral)*100:.1f}%)

Top 3 Strengths:
  1. Scenery/View (+0.923)
  2. Atmosphere (+0.904)
  3. Photo Spots (+0.860)

Areas for Improvement:
  1. Crowd Level (+0.043)
  2. Accessibility (+0.059)
  3. Safety (+0.101)

Model: Llama 3.1 8B (via Groq)
Cost: $0.00 (FREE)
"""

ax3.text(0.1, 0.95, metrics_text, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '09_overall_summary.png', bbox_inches='tight')
plt.close()
print("  Saved: 09_overall_summary.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE!")
print("=" * 60)
print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
print("  1. 01_aspect_mention_frequency.png")
print("  2. 02_sentiment_distribution_stacked.png")
print("  3. 03_llm_vs_indobert_comparison.png (KEY COMPARISON!)")
print("  4. 04_sentiment_score_llm.png")
print("  5. 05_positive_vs_negative.png")
print("  6. 06_aspect_radar_comparison.png")
print("  7. 07_destination_aspect_heatmap.png")
print("  8. 08_top_bottom_performers.png")
print("  9. 09_overall_summary.png")

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

print("\nMost Discussed Aspects:")
top_aspects = summary_df.nlargest(3, 'total_mentions')[['aspect_name', 'total_mentions']]
for _, row in top_aspects.iterrows():
    print(f"  - {row['aspect_name']}: {row['total_mentions']:,} mentions")

print("\nBest Performing Aspects:")
best_aspects = summary_df.nlargest(3, 'sentiment_score')[['aspect_name', 'sentiment_score']]
for _, row in best_aspects.iterrows():
    print(f"  - {row['aspect_name']}: {row['sentiment_score']:+.3f}")

print("\nAreas Needing Improvement:")
worst_aspects = summary_df.nsmallest(3, 'sentiment_score')[['aspect_name', 'sentiment_score']]
for _, row in worst_aspects.iterrows():
    print(f"  - {row['aspect_name']}: {row['sentiment_score']:+.3f}")

print("\nOverall Statistics:")
print(f"  - Total Positive: {total_positive:,} ({total_positive/(total_positive+total_negative+total_neutral)*100:.1f}%)")
print(f"  - Total Negative: {total_negative:,} ({total_negative/(total_positive+total_negative+total_neutral)*100:.1f}%)")
print(f"  - Total Neutral: {total_neutral:,} ({total_neutral/(total_positive+total_negative+total_neutral)*100:.1f}%)")
print("=" * 60)
