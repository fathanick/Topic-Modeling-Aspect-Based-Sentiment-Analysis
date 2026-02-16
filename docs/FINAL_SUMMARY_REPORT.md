# Final Summary Report: Yogyakarta Tourism Reviews Analysis

## Topic Modeling and Aspect-Based Sentiment Analysis Using LLM

**Project Date:** February 2026
**Data Source:** Google Reviews of Yogyakarta Tourism Destinations
**Total Reviews Analyzed:** 1,530 (after preprocessing)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Data Collection](#2-data-collection)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Topic Modeling Analysis](#4-topic-modeling-analysis)
5. [Aspect-Based Sentiment Analysis](#5-aspect-based-sentiment-analysis)
6. [Key Findings & Recommendations](#6-key-findings--recommendations)
7. [Technical Implementation](#7-technical-implementation)
8. [File Structure](#8-file-structure)

---

## 1. Executive Summary

This project analyzes Google reviews of **60 Yogyakarta tourism destinations** to understand visitor experiences through:

1. **Topic Modeling** - Discovering hidden themes in reviews using BERTopic
2. **Aspect-Based Sentiment Analysis (ABSA)** - Evaluating sentiment across 12 tourism aspects using LLM (Groq)

### Key Metrics

| Metric | Value |
|--------|-------|
| Original Reviews | 3,000 |
| After Deduplication | 1,530 |
| Topics Discovered | 23 |
| Aspects Analyzed | 12 |
| Total Aspect Mentions | 4,926 |
| Average Aspects/Review | 3.2 |

### Overall Sentiment Distribution

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Positive | 4,032 | 81.9% |
| Negative | 698 | 14.2% |
| Neutral | 196 | 4.0% |

---

## 2. Data Collection

### 2.1 Data Source
- **Platform:** Google Maps Reviews
- **Method:** Web scraping using Playwright browser automation
- **Destinations:** 60 Yogyakarta tourism locations
- **Source List:** yogyes.com/id/yogyakarta-tourism-object/

### 2.2 Scraping Configuration
- **Max Reviews per Destination:** 100
- **Scroll Pause Time:** 1.5 seconds
- **Request Delay:** 2-5 seconds (random)
- **Timeout:** 15,000 ms
- **Browser:** Chromium with anti-detection measures

### 2.3 Destinations Covered

The dataset includes diverse tourism categories:

| Category | Examples |
|----------|----------|
| Beaches | Pantai Parangtritis, Pantai Indrayanti, Pantai Baron, Pantai Drini |
| Temples | Candi Prambanan, Candi Borobudur, Candi Kalasan, Candi Mendut |
| Cultural Sites | Malioboro, Tugu Jogja, Alun-alun Kidul, Prawirotaman |
| Museums | Museum Ullen Sentalu |
| Nature | Gumuk Pasir Parangkusumo, Bukit Bintang |
| Historical | Makam Raja-Raja Imogiri, Situs Warungboto |
| Religious | Sendang Sono |
| Agrotourism | Agrowisata Bhumi Merapi |

### 2.4 Data Fields Collected

| Field | Description |
|-------|-------------|
| destination | Name of the tourism destination |
| username | Reviewer's display name |
| user_url | Link to reviewer's profile |
| stars | Rating (1-5 stars) |
| time | Review timestamp (relative) |
| text | Review content |

### 2.5 Scraping Process Technical Details

The data collection pipeline consists of two specialized scrapers:

#### 2.5.1 Destinations Scraper (`destinations_scraper.py`)

| Component | Description |
|-----------|-------------|
| **Purpose** | Scrape list of tourism destinations from yogyes.com |
| **Source URL** | https://www.yogyes.com/id/yogyakarta-tourism-object/ |
| **Method** | HTTP request + BeautifulSoup HTML parsing |
| **Extraction** | Regex pattern matching on `<h2>` tags |
| **Output** | `destinations.csv` with name and Google Maps search queries |

**Process Flow:**
1. Fetch webpage content using `requests` library
2. Parse HTML with BeautifulSoup (`html.parser`)
3. Extract numbered destination names from `<h2>` tags using regex: `r'^\d+\.\s*(.+)$'`
4. Generate Google Maps search queries by appending "Yogyakarta"
5. Export to CSV file

#### 2.5.2 Google Reviews Scraper (`google_reviews_scraper.py`)

| Component | Description |
|-----------|-------------|
| **Purpose** | Scrape Google Maps reviews for each destination |
| **Technology** | Playwright async browser automation |
| **Browser** | Chromium with anti-detection measures |
| **Max Reviews** | 100 per destination |
| **Output** | `yogyakarta_tourism_reviews.csv` |

**Browser Configuration:**
- Viewport: 1280 x 800 pixels
- Locale: en-US
- User-Agent: Chrome 120 on macOS
- Anti-detection: `--disable-blink-features=AutomationControlled`

**Scraping Workflow:**
1. **Consent Handling** - Detect and click Google consent popup (supports EN/ID)
2. **Search Navigation** - Open Google Maps → Enter search query → Navigate to Reviews tab
3. **Scroll Loading** - Auto-scroll reviews panel to load more (lazy loading)
4. **Data Extraction** - Extract username, stars, time, and review text
5. **Text Expansion** - Click "More" button to get full review content
6. **Save Results** - Intermediate saves every 5 destinations

**Review Element Selectors:**
| Element | Selector(s) |
|---------|------------|
| Username | `button[data-href*='/contrib/']`, `.d4r55` |
| Stars | `[aria-label*='star']`, `span.kvMYJc` |
| Time | `.rsqaWe` |
| Text | `span.wiI7pd`, `div[tabindex='-1'][lang]` |
| Expand | `button:has-text('Lainnya')`, `.w8nwRe.kyuRq` |

**Error Handling:**
- Multiple fallback selectors for each element
- Graceful handling of missing data
- Intermediate saves to prevent data loss
- Rate limiting with random delays (2-5 seconds)

### 2.6 Scraping Documentation

For detailed technical documentation on the scraping process, see:
- `docs/scraping_documentation.txt`

---

## 3. Data Preprocessing

### 3.1 Preprocessing Pipeline

| Step | Description |
|------|-------------|
| 1. Load Data | Read raw reviews from CSV |
| 2. Remove Duplicates | Eliminate exact duplicate reviews (first occurrence kept) |
| 3. Handle Empty Reviews | Remove null or empty text content |
| 4. Text Cleaning | Lowercase, remove URLs, emails, numbers |
| 5. Remove Punctuation | Keep only alphanumeric characters and spaces |
| 6. Remove Stop Words | Apply bilingual stop word removal |
| 7. Finalize Output | Replace newlines, remove empty reviews |

### 3.2 Preprocessing Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total Reviews | 3,000 | 1,530 | -49.0% |
| Total Words | 85,208 | 53,278 | -37.5% |
| Avg Words/Review | 28.4 | 34.8 | +22.5% |

### 3.3 Stop Words Configuration

| Category | Count | Examples |
|----------|-------|----------|
| Indonesian | 255 | yang, dan, di, ini, itu, dengan, untuk, ke, dari |
| Indonesian (Slang) | - | yg, dgn, utk, gak, ga, nggak, bgt, aja, dong, deh |
| English (NLTK) | 198 | the, a, an, is, are, was, were, be, been |
| **Total Unique** | **453** | - |

### 3.4 Rating Distribution

| Stars | Count | Percentage |
|-------|-------|------------|
| 5 stars | 1,203 | 78.6% |
| 4 stars | 239 | 15.6% |
| 3 stars | 45 | 2.9% |
| 2 stars | 14 | 0.9% |
| 1 star | 29 | 1.9% |

---

## 4. Topic Modeling Analysis

### 4.1 Methodology

**Algorithm:** BERTopic (Embedding + UMAP + HDBSCAN + c-TF-IDF)

| Component | Configuration |
|-----------|---------------|
| Embedding Model | paraphrase-multilingual-MiniLM-L12-v2 |
| UMAP | n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine |
| HDBSCAN | min_cluster_size=10, metric=euclidean, cluster_selection_method=eom |
| Vectorizer | min_df=2, ngram_range=(1,2), Indonesian stop words |

### 4.2 Results Overview

| Metric | Value |
|--------|-------|
| Total Topics | 23 |
| Clustered Documents | 869 (56.8%) |
| Outliers (Topic -1) | 661 (43.2%) |
| Processing Time | 7.81 seconds |

### 4.3 Discovered Topics

| Topic | Size | % | Top Keywords | Theme |
|-------|------|---|--------------|-------|
| 0 | 79 | 5.2% | cafe, makanan, jalan, nongkrong | Cafes & Hanging Out |
| 1 | 66 | 4.3% | snorkeling, pantai, nglambor, air | Water Activities |
| 2 | 54 | 3.5% | seafood, ikan, makan, enak | Seafood Dining |
| 3 | 53 | 3.5% | museum, tour, guide, sejarah | Museum Tours |
| 4 | 50 | 3.3% | foto, spot, tiket, bagus | Photo Spots |
| 5 | 45 | 2.9% | jogja, malioboro, tugu, ramai | Malioboro Area |
| 6 | 45 | 2.9% | foto, bali, fotografer, pantai | Photography Services |
| 7 | 44 | 2.9% | pantai, yogyakarta, krakal, pasir | Beach Variety |
| 8 | 42 | 2.7% | candi, pesanggrahan, kalasan, buddha | Temple Heritage |
| 9 | 40 | 2.6% | kapal, pelabuhan, nelayan, ikan | Fishing Harbor |
| 10 | 38 | 2.5% | ombaknya, sunset, hati hati | Waves & Sunset |
| 11 | 37 | 2.4% | toilet, mandi, mushola, warung | Facilities |
| 12 | 37 | 2.4% | parkir, mobil, masuk, tiket | Parking & Entry |
| 13 | 35 | 2.3% | makam, raja, tangga, sultan | Royal Tombs |
| 14 | 34 | 2.2% | hewan, zoo, mini, anak | Mini Zoo |
| 15 | 26 | 1.7% | camping, jeep, tebing, akses | Adventure Activities |
| 16 | 26 | 1.7% | berdoa, maria, ziarah, katolik | Religious Pilgrimage |
| 17 | 24 | 1.6% | jeep, gumuk, pasir, adrenalin | Sand Dunes Adventure |
| 18 | 22 | 1.4% | candi, prambanan, pemandangan | Prambanan Temple |
| 19 | 21 | 1.4% | tugu, yogyakarta, kota, ikonik | City Landmarks |
| 20 | 18 | 1.2% | solo, candi, adventure | Cross-City Tours |
| 21 | 17 | 1.1% | pantai, bersantai, suasana | Beach Relaxation |
| 22 | 16 | 1.0% | parkir, rb, bayar, mobil | Parking Fees |

### 4.4 Topic-Rating Relationship

| Topic | Theme | Avg Rating |
|-------|-------|------------|
| Topic 16 | Religious Pilgrimage | 5.00 stars |
| Topic 20 | Cross-City Tours | 5.00 stars |
| Topic 5 | Malioboro Area | 4.96 stars |
| Topic 18 | Prambanan Temple | 4.95 stars |
| Topic 8 | Temple Heritage | 4.93 stars |
| Topic 1 | Water Activities | 4.89 stars |

### 4.5 Visualizations Generated

1. **topic_visualization_intertopic_distance.html** - 2D topic relationship map
2. **topic_visualization_hierarchy.html** - Topic dendrogram
3. **topic_visualization_barchart.html** - Top words per topic
4. **topic_visualization_heatmap.html** - Topic similarity matrix

---

## 5. Aspect-Based Sentiment Analysis

### 5.1 Methodology

**Approach:** LLM-based ABSA using Few-Shot Prompting

| Configuration | Value |
|---------------|-------|
| Model | Llama 3.1 8B Instant |
| Provider | Groq API (Free Tier) |
| Batch Size | 10 reviews per API call |
| Temperature | 0.1 (low for consistency) |
| Max Tokens | 4,096 |
| Rate Limit Delay | 3.0 seconds |

### 5.2 Prompt Engineering

**System Prompt Features:**
- Role: Expert Indonesian tourism review sentiment analyzer
- Task: Identify aspects and classify sentiment (POSITIVE/NEGATIVE/NEUTRAL)
- Rules: Indonesian sentiment keywords (bagus, indah, kotor, mahal, etc.)
- Slang Mapping: gak/nggak = tidak, banget = very, mantap = excellent
- Output: Structured JSON with aspect, sentiment, and evidence

**Few-Shot Examples:** 3 examples covering:
1. Positive multi-aspect review (5 stars)
2. Mixed sentiment review (3 stars)
3. Service-focused review (4 stars)

### 5.3 Tourism Aspects Analyzed (12 Total)

| Aspect Key | English Name | Indonesian Name |
|------------|--------------|-----------------|
| cleanliness | Cleanliness | Kebersihan |
| facilities | Facilities | Fasilitas |
| price | Price/Value | Harga |
| service | Service | Pelayanan |
| accessibility | Accessibility | Aksesibilitas |
| scenery | Scenery/View | Pemandangan |
| atmosphere | Atmosphere | Suasana |
| food | Food & Beverage | Makanan |
| safety | Safety | Keamanan |
| crowd | Crowd Level | Keramaian |
| photo_spot | Photo Spots | Spot Foto |
| historical_value | Historical/Cultural | Nilai Sejarah |

### 5.4 Aspect Sentiment Results

| Aspect | Mentions | Positive | Negative | Neutral | Score |
|--------|----------|----------|----------|---------|-------|
| Scenery/View | 1,188 | 95.6% | 3.4% | 1.0% | **+0.923** |
| Atmosphere | 293 | 94.9% | 4.4% | 0.7% | **+0.904** |
| Photo Spots | 129 | 89.1% | 3.1% | 7.8% | **+0.860** |
| Historical Value | 231 | 90.5% | 4.8% | 4.8% | **+0.857** |
| Facilities | 805 | 89.2% | 7.5% | 3.4% | **+0.817** |
| Food & Beverage | 278 | 83.8% | 9.4% | 6.8% | **+0.745** |
| Price/Value | 719 | 81.6% | 13.5% | 4.9% | **+0.682** |
| Service | 283 | 80.2% | 18.7% | 1.1% | **+0.615** |
| Cleanliness | 212 | 65.6% | 24.1% | 10.4% | **+0.415** |
| Safety | 129 | 51.9% | 41.9% | 6.2% | **+0.101** |
| Accessibility | 357 | 49.9% | 44.0% | 6.2% | **+0.059** |
| Crowd Level | 302 | 48.0% | 43.7% | 8.3% | **+0.043** |

### 5.5 Sentiment Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| +0.7 to +1.0 | Excellent (strong positive) |
| +0.3 to +0.7 | Good (moderate positive) |
| 0.0 to +0.3 | Mixed (slight positive or balanced) |
| -0.3 to 0.0 | Concerning (slight negative) |
| -1.0 to -0.3 | Poor (strong negative) |

**Formula:** `Sentiment Score = (Positive - Negative) / Total`

---

## 6. Key Findings & Recommendations

### 6.1 Key Findings

#### Strongest Aspects (Score > +0.8)
| Aspect | Score | Insight |
|--------|-------|---------|
| Scenery/View | +0.923 | Visitors highly appreciate natural beauty |
| Atmosphere | +0.904 | Destinations provide excellent ambiance |
| Photo Spots | +0.860 | Strong appeal for photography tourism |
| Historical Value | +0.857 | Cultural heritage well-received |

#### Areas Needing Attention (Score < +0.3)
| Aspect | Score | Insight |
|--------|-------|---------|
| Crowd Level | +0.043 | Crowding is a concern, especially on weekends |
| Accessibility | +0.059 | Road conditions and parking need improvement |
| Safety | +0.101 | Safety measures at beaches need enhancement |

#### Most Discussed Aspects
1. **Scenery/View** (1,188 mentions) - Core tourism appeal
2. **Facilities** (805 mentions) - Practical visitor needs
3. **Price/Value** (719 mentions) - Value for money matters

### 6.2 Recommendations

#### For Tourism Authorities

1. **Crowd Management**
   - Implement visitor capacity limits during peak times
   - Develop online ticketing with time slots
   - Create alternative routes for popular destinations

2. **Accessibility Improvements**
   - Improve road conditions to destinations
   - Expand parking capacity
   - Add clear signage and directions

3. **Safety Enhancements**
   - Install warning signs at beaches
   - Provide lifeguards at popular beach destinations
   - Add safety barriers at dangerous locations

4. **Cleanliness Initiatives**
   - Increase garbage collection frequency
   - Add more trash bins throughout destinations
   - Implement waste management education

#### Priority Actions by Destination Type

| Destination Type | Priority Action |
|------------------|-----------------|
| Beach destinations | Enhance safety measures |
| Temple complexes | Improve crowd management |
| Food destinations | Maintain cleanliness |
| Photo spots | Manage visitor flow |

---

## 7. Technical Implementation

### 7.1 Technology Stack

| Component | Technology |
|-----------|------------|
| Web Scraping | Playwright (browser automation) |
| Data Processing | Pandas, NumPy |
| NLP Preprocessing | NLTK (English stop words) |
| Topic Modeling | BERTopic, Sentence-Transformers, UMAP, HDBSCAN |
| Sentiment Analysis | Groq API (Llama 3.1 8B) |
| Visualization | Matplotlib, Plotly |

### 7.2 Models Used

| Task | Model | Source |
|------|-------|--------|
| Document Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | Sentence-Transformers |
| Topic Modeling | BERTopic | MaartenGr/BERTopic |
| Sentiment Analysis | Llama 3.1 8B Instant | Groq API |

### 7.3 Processing Performance

| Process | Reviews | Time | Rate |
|---------|---------|------|------|
| Preprocessing | 3,000 → 1,530 | ~5 sec | 306/sec |
| Topic Modeling | 1,530 | 7.8 sec | 196/sec |
| ABSA (LLM) | 1,530 | ~8 min | 3.2/sec |

### 7.4 ABSA Processing Details

| Metric | Value |
|--------|-------|
| API Calls | 153 batches |
| Batch Size | 10 reviews |
| Success Rate | 100% |
| Avg Aspects/Review | 3.2 |

---

## 8. File Structure

```
Topic-Modeling-Aspect-Based-Sentiment-Analysis/
├── data/
│   ├── raw/
│   │   ├── destinations.csv (60 destinations)
│   │   └── yogyakarta_tourism_reviews.csv (~3,000 reviews)
│   └── processed/
│       └── yogyakarta_tourism_reviews_preprocessed.csv (1,530 reviews)
├── src/
│   ├── scraping/
│   │   ├── google_reviews_scraper.py (Playwright-based)
│   │   └── destinations_scraper.py
│   ├── preprocessing/
│   │   └── preprocessing.py
│   ├── topic_modeling/
│   │   └── topic_modeling_bertopic.py
│   └── sentiment_analysis/
│       └── absa_llm_groq.py
├── output/
│   ├── topic_modeling/
│   │   ├── topic_modeling_results.csv
│   │   ├── topics_summary.csv
│   │   └── bertopic_model/
│   └── absa_llm_groq/
│       ├── absa_llm_results.csv
│       ├── absa_llm_summary.csv
│       └── absa_llm_by_destination.csv
├── visualizations/
│   └── topic_modeling/
│       ├── topic_visualization_barchart.html
│       ├── topic_visualization_heatmap.html
│       ├── topic_visualization_hierarchy.html
│       └── topic_visualization_intertopic_distance.html
├── docs/
│   ├── scraping_documentation.txt
│   ├── preprocessing_documentation.txt
│   ├── topic_modeling_documentation.txt
│   ├── absa_llm_groq_documentation.txt
│   └── FINAL_SUMMARY_REPORT.md (this file)
├── README.md
└── requirements.txt
```

---

## Appendix A: Sample LLM Analysis Output

**Input Review:**
> "Candi Prambanan nggak pernah gagal bikin kagum. Areanya luas, bersih. Fasilitasnya lengkap. Worth it banget!"

**LLM Output:**
```json
{
  "review_id": 1,
  "aspects": [
    {"aspect": "scenery", "sentiment": "positive", "evidence": "bikin kagum"},
    {"aspect": "cleanliness", "sentiment": "positive", "evidence": "bersih"},
    {"aspect": "facilities", "sentiment": "positive", "evidence": "lengkap"},
    {"aspect": "price", "sentiment": "positive", "evidence": "worth it"}
  ],
  "overall_sentiment": "positive"
}
```

---

## Appendix B: Indonesian Sentiment Keywords

**Positive Indicators:**
bagus, indah, bersih, nyaman, worth it, mantap, keren, enak, ramah, murah, lengkap, recommended, luar biasa, memuaskan

**Negative Indicators:**
kotor, mahal, jelek, susah, bahaya, kurang, buruk, kecewa, mengecewakan, jauh, sempit, rusak

**Indonesian Slang Mappings:**
- gak/nggak/ga = tidak (not)
- banget/bgt = very
- mantap/mantul = excellent
- keren = cool/awesome
- zonk = disappointing

---

**Report Generated:** February 2026
**Analysis Tool:** Python with BERTopic and Groq LLM (Llama 3.1 8B)
**Author:** Automated Analysis Pipeline
