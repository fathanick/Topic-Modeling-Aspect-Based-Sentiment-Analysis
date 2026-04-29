#!/usr/bin/env python3
"""
ABSA Visualization Script
Creates comprehensive visualizations for Aspect-Based Sentiment Analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Output directory
OUTPUT_DIR = "absa_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("ABSA VISUALIZATION GENERATOR")
print("=" * 60)

# Load data
print("\nLoading ABSA results...")
summary_df = pd.read_csv('absa_indobertweet_summary.csv')
destination_df = pd.read_csv('absa_indobertweet_by_destination.csv')

print(f"  - Summary: {len(summary_df)} aspects")
print(f"  - By Destination: {len(destination_df)} records")

# Color scheme
POSITIVE_COLOR = '#2ecc71'  # Green
NEGATIVE_COLOR = '#e74c3c'  # Red
NEUTRAL_COLOR = '#95a5a6'   # Gray
ACCENT_COLOR = '#3498db'    # Blue

# =============================================================================
# 1. ASPECT MENTION FREQUENCY (Horizontal Bar Chart)
# =============================================================================
print("\n[1/8] Creating Aspect Mention Frequency chart...")

fig, ax = plt.subplots(figsize=(10, 6))

# Sort by total mentions
sorted_df = summary_df.sort_values('total_mentions', ascending=True)

colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_df)))

bars = ax.barh(sorted_df['aspect_name'], sorted_df['total_mentions'], color=colors)

# Add value labels
for bar, value in zip(bars, sorted_df['total_mentions']):
    ax.text(value + 10, bar.get_y() + bar.get_height()/2,
            f'{value:,}', va='center', fontsize=9)

ax.set_xlabel('Number of Mentions')
ax.set_title('Aspect Mention Frequency in Yogyakarta Tourism Reviews', fontweight='bold', pad=15)
ax.set_xlim(0, max(sorted_df['total_mentions']) * 1.15)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_aspect_mention_frequency.png', bbox_inches='tight')
plt.close()
print("  Saved: 01_aspect_mention_frequency.png")

# =============================================================================
# 2. SENTIMENT DISTRIBUTION BY ASPECT (Stacked Bar Chart)
# =============================================================================
print("[2/8] Creating Sentiment Distribution by Aspect chart...")

fig, ax = plt.subplots(figsize=(12, 7))

# Sort by total mentions for better visualization
sorted_df = summary_df.sort_values('total_mentions', ascending=False)

aspects = sorted_df['aspect_name']
x = np.arange(len(aspects))
width = 0.65

# Convert percentage strings to floats
pos_pct = sorted_df['positive_pct'].str.rstrip('%').astype(float)
neg_pct = sorted_df['negative_pct'].str.rstrip('%').astype(float)
neu_pct = sorted_df['neutral_pct'].str.rstrip('%').astype(float)

# Create stacked bars
bars1 = ax.bar(x, pos_pct, width, label='Positive', color=POSITIVE_COLOR)
bars2 = ax.bar(x, neu_pct, width, bottom=pos_pct, label='Neutral', color=NEUTRAL_COLOR)
bars3 = ax.bar(x, neg_pct, width, bottom=pos_pct + neu_pct, label='Negative', color=NEGATIVE_COLOR)

ax.set_xlabel('Aspect')
ax.set_ylabel('Percentage (%)')
ax.set_title('Sentiment Distribution by Aspect', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(aspects, rotation=45, ha='right')
ax.legend(loc='upper right')
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_sentiment_distribution_stacked.png', bbox_inches='tight')
plt.close()
print("  Saved: 02_sentiment_distribution_stacked.png")

# =============================================================================
# 3. SENTIMENT SCORE COMPARISON (Horizontal Bar Chart)
# =============================================================================
print("[3/8] Creating Sentiment Score Comparison chart...")

fig, ax = plt.subplots(figsize=(10, 6))

# Sort by sentiment score
sorted_df = summary_df.sort_values('sentiment_score', ascending=True)

# Color bars based on sentiment score
colors = [POSITIVE_COLOR if score > 0 else NEGATIVE_COLOR
          for score in sorted_df['sentiment_score']]

bars = ax.barh(sorted_df['aspect_name'], sorted_df['sentiment_score'], color=colors)

# Add zero line
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

# Add value labels
for bar, value in zip(bars, sorted_df['sentiment_score']):
    offset = 0.02 if value < 0 else -0.02
    ha = 'left' if value < 0 else 'right'
    ax.text(value + offset, bar.get_y() + bar.get_height()/2,
            f'{value:.3f}', va='center', ha=ha, fontsize=9)

ax.set_xlabel('Sentiment Score (-1 = Negative, +1 = Positive)')
ax.set_title('Sentiment Score by Aspect', fontweight='bold', pad=15)
ax.set_xlim(-1, 0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_sentiment_score_comparison.png', bbox_inches='tight')
plt.close()
print("  Saved: 03_sentiment_score_comparison.png")

# =============================================================================
# 4. POSITIVE VS NEGATIVE MENTIONS (Grouped Bar Chart)
# =============================================================================
print("[4/8] Creating Positive vs Negative Mentions chart...")

fig, ax = plt.subplots(figsize=(12, 6))

sorted_df = summary_df.sort_values('total_mentions', ascending=False)

x = np.arange(len(sorted_df))
width = 0.35

bars1 = ax.bar(x - width/2, sorted_df['positive'], width, label='Positive', color=POSITIVE_COLOR)
bars2 = ax.bar(x + width/2, sorted_df['negative'], width, label='Negative', color=NEGATIVE_COLOR)

ax.set_xlabel('Aspect')
ax.set_ylabel('Number of Mentions')
ax.set_title('Positive vs Negative Mentions by Aspect', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(sorted_df['aspect_name'], rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_positive_vs_negative.png', bbox_inches='tight')
plt.close()
print("  Saved: 04_positive_vs_negative.png")

# =============================================================================
# 5. ASPECT SENTIMENT RADAR CHART
# =============================================================================
print("[5/8] Creating Aspect Sentiment Radar chart...")

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Prepare data
aspects = summary_df['aspect_name'].tolist()
# Normalize sentiment scores to 0-1 range for radar chart
scores = (summary_df['sentiment_score'] + 1) / 2  # Convert from [-1,1] to [0,1]

# Number of aspects
num_aspects = len(aspects)
angles = np.linspace(0, 2 * np.pi, num_aspects, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

scores_list = scores.tolist()
scores_list += scores_list[:1]  # Complete the circle

# Plot
ax.plot(angles, scores_list, 'o-', linewidth=2, color=ACCENT_COLOR, markersize=8)
ax.fill(angles, scores_list, alpha=0.25, color=ACCENT_COLOR)

# Add aspect labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(aspects, size=9)

# Add reference circle at 0.5 (neutral sentiment)
circle_angles = np.linspace(0, 2 * np.pi, 100)
ax.plot(circle_angles, [0.5] * 100, '--', color='gray', linewidth=1, alpha=0.5)
ax.annotate('Neutral', xy=(0, 0.5), fontsize=8, color='gray')

ax.set_ylim(0, 1)
ax.set_title('Aspect Sentiment Profile\n(0 = All Negative, 1 = All Positive)',
             fontweight='bold', pad=20, fontsize=12)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_aspect_radar_chart.png', bbox_inches='tight')
plt.close()
print("  Saved: 05_aspect_radar_chart.png")

# =============================================================================
# 6. TOP DESTINATIONS HEATMAP BY ASPECT
# =============================================================================
print("[6/8] Creating Top Destinations Heatmap...")

# Get top 15 destinations by total mentions
dest_totals = destination_df.groupby('destination')['total'].sum().sort_values(ascending=False)
top_destinations = dest_totals.head(15).index.tolist()

# Filter data for top destinations
top_dest_df = destination_df[destination_df['destination'].isin(top_destinations)]

# Create pivot table
pivot_df = top_dest_df.pivot_table(
    index='destination',
    columns='aspect_name',
    values='sentiment_score',
    aggfunc='mean'
)

# Reorder by total mentions
pivot_df = pivot_df.loc[top_destinations]

fig, ax = plt.subplots(figsize=(14, 10))

# Create custom colormap (red to white to green)
colors = ['#e74c3c', '#f5f5f5', '#2ecc71']
cmap = LinearSegmentedColormap.from_list('sentiment', colors, N=256)

# Create heatmap
im = ax.imshow(pivot_df.values, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Sentiment Score', rotation=270, labelpad=20)

# Set labels
ax.set_xticks(np.arange(len(pivot_df.columns)))
ax.set_yticks(np.arange(len(pivot_df.index)))
ax.set_xticklabels(pivot_df.columns, rotation=45, ha='right')
ax.set_yticklabels(pivot_df.index)

# Add value annotations
for i in range(len(pivot_df.index)):
    for j in range(len(pivot_df.columns)):
        value = pivot_df.iloc[i, j]
        if not np.isnan(value):
            text_color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   color=text_color, fontsize=7)

ax.set_title('Sentiment Score Heatmap: Top 15 Destinations by Aspect',
             fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_destination_aspect_heatmap.png', bbox_inches='tight')
plt.close()
print("  Saved: 06_destination_aspect_heatmap.png")

# =============================================================================
# 7. TOP/BOTTOM PERFORMERS BY ASPECT
# =============================================================================
print("[7/8] Creating Top/Bottom Performers chart...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Select key aspects to visualize
key_aspects = ['Scenery/View', 'Facilities', 'Price/Value', 'Food & Beverage', 'Cleanliness', 'Safety']

for idx, aspect in enumerate(key_aspects):
    ax = axes[idx]

    # Filter by aspect and minimum mentions
    aspect_data = destination_df[
        (destination_df['aspect_name'] == aspect) &
        (destination_df['total'] >= 5)
    ].copy()

    if len(aspect_data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(aspect)
        continue

    # Get top 5 and bottom 5
    top5 = aspect_data.nlargest(5, 'sentiment_score')
    bottom5 = aspect_data.nsmallest(5, 'sentiment_score')

    # Combine and sort
    combined = pd.concat([top5, bottom5]).drop_duplicates()
    combined = combined.sort_values('sentiment_score', ascending=True)

    # Shorten destination names if too long
    combined['short_name'] = combined['destination'].apply(
        lambda x: x[:20] + '...' if len(x) > 20 else x
    )

    colors = [POSITIVE_COLOR if s > -0.3 else NEGATIVE_COLOR
              for s in combined['sentiment_score']]

    bars = ax.barh(combined['short_name'], combined['sentiment_score'], color=colors)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-1.1, 0.6)
    ax.set_title(f'{aspect}', fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)

plt.suptitle('Best & Worst Performing Destinations by Aspect\n(min. 5 mentions)',
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_top_bottom_performers.png', bbox_inches='tight')
plt.close()
print("  Saved: 07_top_bottom_performers.png")

# =============================================================================
# 8. OVERALL SENTIMENT SUMMARY PIE CHART
# =============================================================================
print("[8/8] Creating Overall Sentiment Summary...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Overall sentiment distribution
total_positive = summary_df['positive'].sum()
total_negative = summary_df['negative'].sum()
total_neutral = summary_df['neutral'].sum()

sizes = [total_positive, total_negative, total_neutral]
labels = [f'Positive\n({total_positive:,})',
          f'Negative\n({total_negative:,})',
          f'Neutral\n({total_neutral:,})']
colors = [POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR]
explode = (0.02, 0.02, 0.02)

ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=False, startangle=90, textprops={'fontsize': 10})
ax1.set_title('Overall Sentiment Distribution\n(All Aspects Combined)',
              fontweight='bold', pad=15)

# Right: Confidence distribution
avg_confidence = summary_df['avg_confidence'].mean()

# Create gauge-like visualization
theta = np.linspace(0, np.pi, 100)
r = 1

# Background arc
ax2.plot(np.cos(theta), np.sin(theta), 'lightgray', linewidth=20)

# Confidence arc
confidence_angle = np.pi * avg_confidence
theta_conf = np.linspace(0, confidence_angle, 50)
ax2.plot(np.cos(theta_conf), np.sin(theta_conf), color=ACCENT_COLOR, linewidth=20)

# Add needle
needle_angle = np.pi * avg_confidence
ax2.arrow(0, 0, 0.7*np.cos(needle_angle), 0.7*np.sin(needle_angle),
          head_width=0.05, head_length=0.02, fc='black', ec='black')

ax2.text(0, 0.3, f'{avg_confidence:.1%}', ha='center', va='center',
         fontsize=24, fontweight='bold')
ax2.text(0, 0.1, 'Average Confidence', ha='center', va='center', fontsize=10)

ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-0.3, 1.3)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('Model Confidence Score', fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_overall_summary.png', bbox_inches='tight')
plt.close()
print("  Saved: 08_overall_summary.png")

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
print("  3. 03_sentiment_score_comparison.png")
print("  4. 04_positive_vs_negative.png")
print("  5. 05_aspect_radar_chart.png")
print("  6. 06_destination_aspect_heatmap.png")
print("  7. 07_top_bottom_performers.png")
print("  8. 08_overall_summary.png")

# Print key insights
print("\n" + "=" * 60)
print("KEY INSIGHTS FROM VISUALIZATIONS")
print("=" * 60)

print("\nMost Discussed Aspects:")
top_aspects = summary_df.nlargest(3, 'total_mentions')[['aspect_name', 'total_mentions']]
for _, row in top_aspects.iterrows():
    print(f"  - {row['aspect_name']}: {row['total_mentions']:,} mentions")

print("\nBest Performing Aspects (by sentiment score):")
best_aspects = summary_df.nlargest(3, 'sentiment_score')[['aspect_name', 'sentiment_score']]
for _, row in best_aspects.iterrows():
    print(f"  - {row['aspect_name']}: {row['sentiment_score']:.3f}")

print("\nAreas Needing Improvement:")
worst_aspects = summary_df.nsmallest(3, 'sentiment_score')[['aspect_name', 'sentiment_score']]
for _, row in worst_aspects.iterrows():
    print(f"  - {row['aspect_name']}: {row['sentiment_score']:.3f}")

print("\nOverall Statistics:")
print(f"  - Total Positive: {total_positive:,} ({total_positive/(total_positive+total_negative+total_neutral)*100:.1f}%)")
print(f"  - Total Negative: {total_negative:,} ({total_negative/(total_positive+total_negative+total_neutral)*100:.1f}%)")
print(f"  - Total Neutral: {total_neutral:,} ({total_neutral/(total_positive+total_negative+total_neutral)*100:.1f}%)")
print(f"  - Average Confidence: {avg_confidence:.1%}")
