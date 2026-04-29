"""
Topic Modeling using BERTopic

This script performs topic modeling on preprocessed Google reviews
using BERTopic - a topic modeling technique that leverages transformers
and c-TF-IDF to create dense clusters with interpretable topics.

Input: yogyakarta_tourism_reviews_preprocessed.csv
Output:
    - topic_modeling_results.csv (reviews with assigned topics)
    - topic_modeling_documentation.txt (detailed documentation)
    - topics_summary.csv (topic information)
    - Visualizations (HTML files)

Requirements:
    pip install bertopic pandas numpy scikit-learn sentence-transformers umap-learn hdbscan

Usage:
    python topic_modeling_bertopic.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# BERTopic imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# ============== Configuration ==============
INPUT_FILE = "yogyakarta_tourism_reviews_preprocessed.csv"
OUTPUT_RESULTS_FILE = "topic_modeling_results.csv"
OUTPUT_TOPICS_FILE = "topics_summary.csv"
DOCUMENTATION_FILE = "topic_modeling_documentation.txt"

# BERTopic Configuration
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual model for Indonesian
MIN_TOPIC_SIZE = 10  # Minimum number of documents per topic
N_NEIGHBORS = 15  # UMAP parameter
N_COMPONENTS = 5  # UMAP dimensions
MIN_CLUSTER_SIZE = 10  # HDBSCAN parameter
RANDOM_STATE = 42

# Indonesian stop words for CountVectorizer (additional filtering)
INDONESIAN_STOP_WORDS = [
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk', 'pada',
    'adalah', 'sebagai', 'dalam', 'tidak', 'akan', 'tetapi', 'juga', 'atau',
    'ada', 'bisa', 'sudah', 'saya', 'kami', 'kita', 'mereka', 'sangat', 'sekali',
    'lebih', 'banyak', 'hanya', 'saja', 'karena', 'jadi', 'kalau', 'kalo',
    'banget', 'aja', 'gak', 'nya', 'yg', 'udah', 'buat', 'pas', 'lagi'
]


class TopicModelingResults:
    """Class to store topic modeling results and statistics"""
    def __init__(self):
        self.total_documents = 0
        self.total_topics = 0
        self.outlier_count = 0
        self.topics_info = None
        self.topic_words = {}
        self.topic_sizes = {}
        self.model_params = {}
        self.processing_time = 0


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


def create_bertopic_model():
    """Create and configure BERTopic model"""
    print("\nConfiguring BERTopic model...")

    # 1. Embedding Model - Multilingual for Indonesian text
    print(f"   Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # 2. UMAP - Dimensionality reduction
    print(f"   Configuring UMAP (n_neighbors={N_NEIGHBORS}, n_components={N_COMPONENTS})")
    umap_model = UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=N_COMPONENTS,
        min_dist=0.0,
        metric='cosine',
        random_state=RANDOM_STATE
    )

    # 3. HDBSCAN - Clustering
    print(f"   Configuring HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE})")
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    # 4. CountVectorizer - For c-TF-IDF
    print("   Configuring CountVectorizer with Indonesian stop words")
    vectorizer_model = CountVectorizer(
        stop_words=INDONESIAN_STOP_WORDS,
        min_df=2,
        ngram_range=(1, 2)  # Include bigrams
    )

    # 5. Create BERTopic model
    print("   Creating BERTopic model...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=MIN_TOPIC_SIZE,
        verbose=True,
        calculate_probabilities=True
    )

    return topic_model


def run_topic_modeling(topic_model, documents):
    """Run topic modeling on documents"""
    print("\n" + "=" * 60)
    print("RUNNING TOPIC MODELING")
    print("=" * 60)

    start_time = datetime.now()

    # Fit the model
    print("\nFitting BERTopic model (this may take a few minutes)...")
    topics, probabilities = topic_model.fit_transform(documents)

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    print(f"\nTopic modeling completed in {processing_time:.2f} seconds")

    return topics, probabilities, processing_time


def analyze_results(topic_model, topics, documents, results):
    """Analyze topic modeling results"""
    print("\n" + "=" * 60)
    print("ANALYZING RESULTS")
    print("=" * 60)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    results.topics_info = topic_info

    # Count statistics
    results.total_documents = len(documents)
    results.total_topics = len(topic_info) - 1  # Exclude outlier topic (-1)
    results.outlier_count = (np.array(topics) == -1).sum()

    print(f"\nTotal documents: {results.total_documents}")
    print(f"Total topics discovered: {results.total_topics}")
    print(f"Outliers (Topic -1): {results.outlier_count} ({results.outlier_count/results.total_documents*100:.1f}%)")

    # Get topic words and sizes
    print("\n" + "-" * 40)
    print("TOPIC SUMMARY")
    print("-" * 40)

    for topic_id in topic_info['Topic'].values:
        if topic_id == -1:
            results.topic_words[-1] = "Outliers"
            results.topic_sizes[-1] = results.outlier_count
            continue

        # Get top words for this topic
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            words = [word for word, _ in topic_words[:10]]
            results.topic_words[topic_id] = words
            results.topic_sizes[topic_id] = (np.array(topics) == topic_id).sum()

            print(f"\nTopic {topic_id} ({results.topic_sizes[topic_id]} documents):")
            print(f"   Keywords: {', '.join(words[:5])}")

    return results


def save_results(df, topics, probabilities, topic_model, results):
    """Save results to files"""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # 1. Save reviews with topics
    print(f"\nSaving results to {OUTPUT_RESULTS_FILE}...")
    df_results = df.copy()
    df_results['topic'] = topics
    df_results['topic_probability'] = [max(prob) if prob is not None and len(prob) > 0 else 0 for prob in probabilities]

    # Add topic labels
    topic_labels = {}
    for topic_id, words in results.topic_words.items():
        if topic_id == -1:
            topic_labels[-1] = "Outlier"
        else:
            topic_labels[topic_id] = f"Topic_{topic_id}: {', '.join(words[:3]) if isinstance(words, list) else words}"

    df_results['topic_label'] = df_results['topic'].map(topic_labels)
    df_results.to_csv(OUTPUT_RESULTS_FILE, index=False, encoding='utf-8')
    print(f"   Saved {len(df_results)} reviews with topic assignments")

    # 2. Save topics summary
    print(f"\nSaving topics summary to {OUTPUT_TOPICS_FILE}...")
    topics_data = []
    for topic_id in sorted(results.topic_words.keys()):
        if topic_id == -1:
            continue
        words = results.topic_words[topic_id]
        topics_data.append({
            'topic_id': topic_id,
            'document_count': results.topic_sizes[topic_id],
            'percentage': f"{results.topic_sizes[topic_id]/results.total_documents*100:.2f}%",
            'top_words': ', '.join(words) if isinstance(words, list) else words
        })

    df_topics = pd.DataFrame(topics_data)
    df_topics.to_csv(OUTPUT_TOPICS_FILE, index=False, encoding='utf-8')
    print(f"   Saved {len(df_topics)} topics")

    # 3. Save visualizations
    print("\nGenerating visualizations...")
    try:
        # Topic visualization
        fig_topics = topic_model.visualize_topics()
        fig_topics.write_html("topic_visualization_intertopic_distance.html")
        print("   Saved: topic_visualization_intertopic_distance.html")
    except Exception as e:
        print(f"   Could not save intertopic distance map: {e}")

    try:
        # Barchart of topics
        fig_barchart = topic_model.visualize_barchart(top_n_topics=min(10, results.total_topics))
        fig_barchart.write_html("topic_visualization_barchart.html")
        print("   Saved: topic_visualization_barchart.html")
    except Exception as e:
        print(f"   Could not save barchart: {e}")

    try:
        # Hierarchy
        fig_hierarchy = topic_model.visualize_hierarchy()
        fig_hierarchy.write_html("topic_visualization_hierarchy.html")
        print("   Saved: topic_visualization_hierarchy.html")
    except Exception as e:
        print(f"   Could not save hierarchy: {e}")

    try:
        # Heatmap
        fig_heatmap = topic_model.visualize_heatmap()
        fig_heatmap.write_html("topic_visualization_heatmap.html")
        print("   Saved: topic_visualization_heatmap.html")
    except Exception as e:
        print(f"   Could not save heatmap: {e}")

    # 4. Save the model
    print("\nSaving BERTopic model...")
    try:
        topic_model.save("bertopic_model")
        print("   Saved: bertopic_model/")
    except Exception as e:
        print(f"   Could not save model: {e}")

    return df_results


def generate_documentation(results, df_results, topic_model):
    """Generate comprehensive documentation"""
    print(f"\nGenerating documentation ({DOCUMENTATION_FILE})...")

    # Get topic info for documentation
    topic_info = topic_model.get_topic_info()

    doc_content = f"""================================================================================
TOPIC MODELING DOCUMENTATION
BERTopic Analysis of Yogyakarta Tourism Reviews
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input File: {INPUT_FILE}
Output Files:
    - {OUTPUT_RESULTS_FILE} (reviews with topic assignments)
    - {OUTPUT_TOPICS_FILE} (topic summary)
    - topic_visualization_*.html (visualizations)
    - bertopic_model/ (saved model)

================================================================================
1. METHODOLOGY
================================================================================

BERTopic Overview:
------------------
BERTopic is a topic modeling technique that leverages transformer-based
embeddings and a class-based TF-IDF (c-TF-IDF) to create dense clusters
allowing for easily interpretable topics while keeping important words
in the topic descriptions.

Pipeline Steps:
1. Document Embedding: Convert documents to embeddings using Sentence-BERT
2. Dimensionality Reduction: Use UMAP to reduce embedding dimensions
3. Clustering: Use HDBSCAN to cluster similar documents
4. Topic Representation: Extract topic words using c-TF-IDF

Model Configuration:
--------------------
- Embedding Model: {EMBEDDING_MODEL}
  (Multilingual model suitable for Indonesian text)

- UMAP Parameters:
  - n_neighbors: {N_NEIGHBORS}
  - n_components: {N_COMPONENTS}
  - min_dist: 0.0
  - metric: cosine
  - random_state: {RANDOM_STATE}

- HDBSCAN Parameters:
  - min_cluster_size: {MIN_CLUSTER_SIZE}
  - metric: euclidean
  - cluster_selection_method: eom

- Vectorizer:
  - min_df: 2
  - ngram_range: (1, 2) - unigrams and bigrams
  - Additional Indonesian stop words applied

================================================================================
2. RESULTS SUMMARY
================================================================================

Dataset Statistics:
-------------------
- Total documents analyzed: {results.total_documents:,}
- Processing time: {results.processing_time:.2f} seconds

Topic Discovery:
----------------
- Total topics discovered: {results.total_topics}
- Outliers (unassigned documents): {results.outlier_count:,} ({results.outlier_count/results.total_documents*100:.1f}%)
- Documents assigned to topics: {results.total_documents - results.outlier_count:,} ({(results.total_documents - results.outlier_count)/results.total_documents*100:.1f}%)

================================================================================
3. TOPIC DETAILS
================================================================================

"""

    # Add detailed topic information
    for topic_id in sorted(results.topic_words.keys()):
        if topic_id == -1:
            doc_content += f"""
Topic -1: OUTLIERS
------------------
- Document count: {results.outlier_count:,}
- Percentage: {results.outlier_count/results.total_documents*100:.2f}%
- Description: Documents that don't fit well into any discovered topic

"""
            continue

        words = results.topic_words[topic_id]
        size = results.topic_sizes[topic_id]
        percentage = size / results.total_documents * 100

        doc_content += f"""
Topic {topic_id}
{'-' * 40}
- Document count: {size:,}
- Percentage: {percentage:.2f}%
- Top keywords: {', '.join(words) if isinstance(words, list) else words}

"""

    # Add topic distribution by destination
    doc_content += """
================================================================================
4. TOPIC DISTRIBUTION BY DESTINATION
================================================================================

"""

    # Calculate topic distribution per destination
    topic_dest = df_results.groupby(['destination', 'topic']).size().unstack(fill_value=0)

    for dest in topic_dest.index[:10]:  # Top 10 destinations
        doc_content += f"\n{dest}:\n"
        dest_topics = topic_dest.loc[dest]
        for topic_id in dest_topics[dest_topics > 0].index:
            if topic_id != -1:
                count = dest_topics[topic_id]
                doc_content += f"  - Topic {topic_id}: {count} reviews\n"

    # Add rating distribution per topic
    doc_content += """
================================================================================
5. TOPIC-RATING RELATIONSHIP
================================================================================

Average rating per topic:

"""

    topic_ratings = df_results.groupby('topic')['stars'].agg(['mean', 'count'])
    for topic_id in sorted(topic_ratings.index):
        if topic_id == -1:
            label = "Outliers"
        else:
            label = f"Topic {topic_id}"
        mean_rating = topic_ratings.loc[topic_id, 'mean']
        count = topic_ratings.loc[topic_id, 'count']
        doc_content += f"{label}: {mean_rating:.2f} stars (n={count})\n"

    # Add sample documents per topic
    doc_content += """
================================================================================
6. SAMPLE DOCUMENTS PER TOPIC
================================================================================

"""

    for topic_id in sorted(results.topic_words.keys()):
        if topic_id == -1:
            continue

        doc_content += f"\nTopic {topic_id} - Sample Reviews:\n"
        doc_content += "-" * 40 + "\n"

        # Get sample documents for this topic
        topic_docs = df_results[df_results['topic'] == topic_id].head(3)
        for idx, row in topic_docs.iterrows():
            text = row['cleaned_text'][:200] + "..." if len(str(row['cleaned_text'])) > 200 else row['cleaned_text']
            doc_content += f"- [{row['destination']}] {row['stars']} stars: {text}\n\n"

    # Add interpretation guide
    doc_content += """
================================================================================
7. INTERPRETATION GUIDE
================================================================================

How to Interpret Topics:
------------------------
1. Each topic represents a cluster of semantically similar reviews
2. Topic keywords indicate the main themes discussed in that cluster
3. Higher document counts indicate more prevalent themes
4. Outliers (-1) are reviews that don't fit any clear topic pattern

Possible Topic Interpretations:
-------------------------------
Based on the keywords, topics may represent:
- Facilities and amenities (parkir, toilet, fasilitas)
- Natural beauty and scenery (pantai, pemandangan, indah)
- Historical/cultural aspects (candi, sejarah, budaya)
- Experience and atmosphere (suasana, nyaman, bagus)
- Food and services (makanan, harga, murah)
- Activities (foto, jalan, naik)

Using the Results:
------------------
1. Review 'topic_modeling_results.csv' for individual review topics
2. Check 'topics_summary.csv' for topic overview
3. Open HTML visualizations for interactive exploration
4. Use the saved model for predicting topics on new reviews

================================================================================
8. VISUALIZATIONS
================================================================================

Generated Visualizations:
-------------------------
1. topic_visualization_intertopic_distance.html
   - Interactive 2D map showing topic relationships
   - Distance between topics indicates similarity

2. topic_visualization_barchart.html
   - Top words per topic with their importance scores

3. topic_visualization_hierarchy.html
   - Hierarchical clustering of topics
   - Shows how topics relate to each other

4. topic_visualization_heatmap.html
   - Topic similarity matrix
   - Darker colors indicate higher similarity

================================================================================
9. TECHNICAL DETAILS
================================================================================

Libraries Used:
- bertopic: Topic modeling
- sentence-transformers: Document embeddings
- umap-learn: Dimensionality reduction
- hdbscan: Clustering
- scikit-learn: CountVectorizer, metrics
- pandas: Data manipulation
- numpy: Numerical operations

Model Persistence:
- Model saved to: bertopic_model/
- Can be loaded with: BERTopic.load("bertopic_model")

Reproducing Results:
- Random state set to {RANDOM_STATE} for reproducibility
- Same results can be obtained by running with identical parameters

================================================================================
END OF DOCUMENTATION
================================================================================
"""

    # Write documentation
    with open(DOCUMENTATION_FILE, 'w', encoding='utf-8') as f:
        f.write(doc_content)

    print(f"   Documentation saved to {DOCUMENTATION_FILE}")


def main():
    """Main function"""
    print("=" * 60)
    print("TOPIC MODELING WITH BERTOPIC")
    print("Yogyakarta Tourism Reviews Analysis")
    print("=" * 60)

    results = TopicModelingResults()

    # Store model parameters
    results.model_params = {
        'embedding_model': EMBEDDING_MODEL,
        'min_topic_size': MIN_TOPIC_SIZE,
        'n_neighbors': N_NEIGHBORS,
        'n_components': N_COMPONENTS,
        'min_cluster_size': MIN_CLUSTER_SIZE,
        'random_state': RANDOM_STATE
    }

    # 1. Load data
    df = load_data(INPUT_FILE)
    if df is None:
        return

    # Get documents (cleaned text)
    documents = df['cleaned_text'].fillna('').tolist()

    # Filter out empty documents
    valid_indices = [i for i, doc in enumerate(documents) if doc.strip()]
    documents = [documents[i] for i in valid_indices]
    df = df.iloc[valid_indices].reset_index(drop=True)

    print(f"   Valid documents for modeling: {len(documents)}")

    # 2. Create BERTopic model
    topic_model = create_bertopic_model()

    # 3. Run topic modeling
    topics, probabilities, processing_time = run_topic_modeling(topic_model, documents)
    results.processing_time = processing_time

    # 4. Analyze results
    results = analyze_results(topic_model, topics, documents, results)

    # 5. Save results
    df_results = save_results(df, topics, probabilities, topic_model, results)

    # 6. Generate documentation
    generate_documentation(results, df_results, topic_model)

    # Print final summary
    print("\n" + "=" * 60)
    print("TOPIC MODELING COMPLETE")
    print("=" * 60)
    print(f"Total documents: {results.total_documents:,}")
    print(f"Topics discovered: {results.total_topics}")
    print(f"Outliers: {results.outlier_count:,} ({results.outlier_count/results.total_documents*100:.1f}%)")
    print(f"Processing time: {results.processing_time:.2f} seconds")
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_RESULTS_FILE}")
    print(f"  - {OUTPUT_TOPICS_FILE}")
    print(f"  - {DOCUMENTATION_FILE}")
    print(f"  - topic_visualization_*.html")
    print("=" * 60)


if __name__ == "__main__":
    main()
