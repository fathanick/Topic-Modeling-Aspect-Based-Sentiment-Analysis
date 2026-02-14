# Topic Modeling & LLM-Based Aspect Sentiment Analysis

Analysis of Yogyakarta Tourism Reviews using BERTopic and LLM-based ABSA.

## Overview

This project analyzes Google reviews from 60 Yogyakarta tourism destinations to understand visitor experiences through:
- **Topic Modeling** using BERTopic
- **Aspect-Based Sentiment Analysis** using LLM (Llama 3.1 via Groq API)

## Quick Stats

| Metric | Value |
|--------|-------|
| Reviews Analyzed | 1,530 |
| Destinations | 60 |
| Topics Discovered | 23 |
| Aspects Analyzed | 12 |
| Overall Positive Sentiment | 81.9% |
| Overall Negative Sentiment | 14.2% |

## Project Structure

```
.
├── data/
│   ├── raw/                          # Original data files
│   │   ├── destinations.csv          # List of 60 tourism destinations
│   │   └── yogyakarta_tourism_reviews.csv
│   └── processed/                    # Preprocessed data
│       └── yogyakarta_tourism_reviews_preprocessed.csv
├── src/
│   ├── scraping/                     # Web scraping scripts
│   │   ├── google_reviews_scraper.py
│   │   └── destinations_scraper.py
│   ├── preprocessing/                # Data cleaning
│   │   └── preprocessing.py
│   ├── topic_modeling/               # BERTopic analysis
│   │   └── topic_modeling_bertopic.py
│   ├── sentiment_analysis/           # ABSA scripts
│   │   └── absa_llm_groq.py          # LLM-based ABSA (Llama 3.1)
│   └── visualization/                # Chart generation
│       └── absa_llm_visualizations.py
├── output/
│   ├── topic_modeling/               # BERTopic results
│   └── absa_llm_groq/                # LLM ABSA results
├── visualizations/
│   ├── topic_modeling/               # Interactive HTML visualizations
│   └── absa_llm/                     # ABSA charts (PNG)
├── docs/                             # Documentation
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing
```bash
python src/preprocessing/preprocessing.py
```

### 2. Topic Modeling
```bash
python src/topic_modeling/topic_modeling_bertopic.py
```

### 3. Aspect-Based Sentiment Analysis
```bash
export GROQ_API_KEY="your-api-key"
python src/sentiment_analysis/absa_llm_groq.py
```

### 4. Generate Visualizations
```bash
python src/visualization/absa_llm_visualizations.py
```

## Key Findings

### Most Discussed Aspects
| Aspect | Mentions | Positive% | Sentiment Score |
|--------|----------|-----------|-----------------|
| Scenery/View | 1,188 | 95.6% | +0.923 |
| Facilities | 805 | 89.2% | +0.817 |
| Price/Value | 719 | 81.6% | +0.682 |
| Accessibility | 357 | 49.9% | +0.059 |
| Crowd Level | 302 | 48.0% | +0.043 |

### Highest Rated Aspects
- **Scenery/View** (+0.923) - Natural beauty highly appreciated
- **Atmosphere** (+0.904) - Peaceful, comfortable ambiance
- **Photo Spots** (+0.860) - Strong social media tourism potential
- **Historical Value** (+0.857) - Cultural heritage valued

### Areas Needing Improvement
- **Accessibility** (+0.059) - Road conditions, parking, distance
- **Crowd Level** (+0.043) - Visitor management during peak times
- **Safety** (+0.101) - Warning signs, lifeguards at beaches

## Models Used

| Task | Model |
|------|-------|
| Document Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Topic Modeling | BERTopic (UMAP + HDBSCAN + c-TF-IDF) |
| Sentiment Analysis | Llama 3.1 8B via Groq API (few-shot prompting) |

## BERTopic Results

23 distinct topics discovered from 1,530 reviews (56.8% clustered, 43.2% outliers):

| Topic | Theme | Size | Top Keywords |
|-------|-------|------|--------------|
| 0 | Food & Cafes | 79 | cafe, makanan, nongkrong |
| 1 | Beach Activities | 66 | snorkeling, pantai, laut |
| 2 | Seafood Dining | 54 | seafood, ikan, warung |
| 3 | Cultural Tours | 53 | museum, tour, guide |
| 4 | Photo Spots | 50 | foto, spot, tiket |
| 5 | Urban Attractions | 45 | malioboro, tugu, jalan |
| 6 | Photography Services | 45 | foto, fotografer, jasa |
| 7 | Beach Destinations | 44 | pantai, krakal, drini |
| 8 | Temple Heritage | 42 | candi, buddha, relief |
| 9 | Fishing Ports | 40 | kapal, pelabuhan, nelayan |

## ABSA Visualizations

Located in `visualizations/absa_llm/`:
- `01_aspect_mention_frequency.png` - Aspect mention distribution
- `02_sentiment_distribution_stacked.png` - Sentiment by aspect
- `03_llm_vs_indobert_comparison.png` - Model comparison
- `04_sentiment_score_llm.png` - Sentiment scores
- `05_positive_vs_negative.png` - Positive/negative breakdown
- `06_aspect_radar_comparison.png` - Multi-aspect radar
- `07_destination_aspect_heatmap.png` - Destination-aspect matrix
- `08_top_bottom_performers.png` - Best/worst destinations

## Topic Modeling Visualizations

Located in `visualizations/topic_modeling/` (interactive HTML):
- `topic_visualization_intertopic_distance.html` - 2D topic map
- `topic_visualization_hierarchy.html` - Topic dendrogram
- `topic_visualization_barchart.html` - Top words per topic
- `topic_visualization_heatmap.html` - Word-topic relevance

## 12 Tourism Aspects Analyzed

1. **Scenery/View** (Pemandangan) - Natural beauty, landscapes
2. **Facilities** (Fasilitas) - Toilets, parking, infrastructure
3. **Price/Value** (Harga) - Ticket prices, value for money
4. **Accessibility** (Aksesibilitas) - Road conditions, distance, parking
5. **Crowd Level** (Keramaian) - Crowdedness, visitor management
6. **Atmosphere** (Suasana) - Ambiance, mood, tranquility
7. **Service** (Pelayanan) - Staff helpfulness, tour guides
8. **Food & Beverage** (Makanan) - Restaurant quality, local cuisine
9. **Historical Value** (Nilai Sejarah) - Cultural significance, heritage
10. **Cleanliness** (Kebersihan) - Hygiene, trash management
11. **Safety** (Keamanan) - Security, warning signs
12. **Photo Spots** (Spot Foto) - Instagram-worthy locations

## Requirements

- Python 3.8+
- pandas, numpy
- sentence-transformers
- bertopic
- matplotlib, plotly
- groq (for LLM API)
- nltk

## License

This project is for academic research purposes.
