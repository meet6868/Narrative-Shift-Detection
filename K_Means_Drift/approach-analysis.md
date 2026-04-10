# Narrative Drift Detection using K-Means (Single User Input, Topic-Wise Inference)

## 1. Objective

This pipeline detects narrative drift over time from one user input CSV by:

1. splitting articles into sentences,
2. generating contextual SBERT embeddings,
3. assigning each sentence to one dominant topic,
4. training a topic-wise K-Means model from filtered user sentences,
5. measuring date-to-date distribution drift,
6. and printing sentence-level before-vs-after narrative changes.

The five configured topics are:

1. War
2. Health
3. Technology
4. Climate
5. Economics

------------------------------------------------------------------------

## 2. Current Input Contract (What the Code Actually Uses)

### Input file

K_Means_Drift/user_article2.csv

### Required columns

1. date
2. article

### Path handling

The notebook resolves the input path robustly using multiple candidate paths, so it works even if notebook working directory changes.

### Format handling

If CSV has trailing comma artifacts (extra unnamed columns), they are dropped automatically.

------------------------------------------------------------------------

## 3. High-Level End-to-End Flow

### Stage A: Input Setup and Validation

1. Resolve user CSV path.
2. Load CSV into DataFrame.
3. Remove unnamed columns from malformed schema variants.
4. Validate required columns date and article.
5. Initialize inference config and core constants.

### Stage B: User Inference Preprocessing (Exact Requested Chain)

1. split_articles_into_sentences
2. build_context_texts
3. generate_contextual_sbert_embeddings
4. soft_topic_label_sentences
5. filter_user_topic_sentences

### Stage C: Topic-Wise Clustering and Prediction

1. Train K-Means per topic using filtered user sentences.
2. Build per-topic test matrices.
3. Predict sentence cluster labels per topic.

### Stage D: Drift Computation and Interpretation

1. Build date-wise cluster distributions.
2. Compute Jensen-Shannon drift between consecutive dates.
3. Print topic-wise shift events.
4. Print sentence-level before-vs-after mapping with change score.
5. Print compact topic summary metrics.

------------------------------------------------------------------------

## 4. Detailed Flow (Cell-Level Logic)

## Step 1: Imports

Core libraries loaded:

1. pandas, numpy, ast
2. KMeans from scikit-learn
3. Jensen-Shannon distance from scipy
4. SentenceTransformer (later step)

Purpose:

Prepare data manipulation, clustering, distance computation, and embedding generation.

## Step 2: parse_embedding helper

A generic helper to parse embedding-like string fields into numpy vectors.

Note:

In the current single-input run, embeddings are generated from SBERT directly, so this helper is mostly retained as utility compatibility.

## Step 3: Single input configuration

What happens:

1. Define TOPICS and NUM_CLUSTERS.
2. Resolve USER_CSV_PATH from candidate paths.
3. Load user_input_df from CSV.
4. Drop malformed unnamed columns.
5. Print row count and columns.

Why this matters:

Prevents FileNotFoundError and schema instability across environments.

## Step 4: Schema validation

Checks input has date and article.

If missing, raises clear ValueError early.

## Step 5: Inference config

Current config includes:

1. topics
2. context_window = 3
3. inference_batch_size = 16
4. topic_threshold = 0.25
5. embedding_dim = 768

Note:

The threshold is still present in config, but topic routing now primarily relies on dominant_topic assignment.

## Step 6 and Step 7 notes

These cells document that training is now performed after topic labeling and filtering on user data, not from external Climate.csv.

## Step 8: User inference pipeline (Steps 1 to 3)

### 8.1 split_articles_into_sentences

Input: user_input_df with date and article.

Output sentence columns:

1. date
2. article_id
3. sentence_id
4. sentence_text
5. sentence_order

How:

1. Parse date with dayfirst=True.
2. Split article text by sentence-ending punctuation.
3. Create one row per sentence.

### 8.2 build_context_texts

For each sentence, constructs a local context string using window size 3 or 5.

For context_window = 3:

Each sentence embedding uses previous, current, next sentence context when available.

### 8.3 generate_contextual_sbert_embeddings

1. Encodes context_text using all-mpnet-base-v2.
2. Ensures output dimension is 768.
3. Stores vectors in sentence_embeddings.

Result of Step 8:

sentence_df ready for topic labeling.

## Step 9: Optional model save

Saves topic_models if they exist.

If models are not yet created, prints guidance message.

## Step 10: Soft topic labeling and filtering (Steps 4 and 5)

### 10.1 Build topic prototypes from topic labels

Instead of using Climate.csv embeddings, the code builds one prototype vector per topic by encoding topic names:

1. War
2. Health
3. Technology
4. Climate
5. Economics

### 10.2 Soft topic scoring

For each sentence embedding:

1. Compute similarity with all topic prototypes.
2. Apply softmax to get topic probabilities.
3. Assign dominant_topic by argmax probability.

### 10.3 Topic filtering (current behavior)

filter_user_topic_sentences now keeps rows where dominant_topic equals requested topic.

This creates mutually exclusive per-topic sentence sets.

### 10.4 Topic-wise model training

For each topic:

1. If filtered rows < NUM_CLUSTERS, skip training.
2. Else train KMeans(n_clusters=5) on sentence_embeddings.

Output objects:

1. filtered_by_topic
2. topic_models

Important practical implication:

If input file is mostly war-related, only War gets enough data and a trained model.

## Step 11: Topic-wise matrices

Build X_topic_test only for topics with:

1. trained model, and
2. non-empty filtered rows.

## Step 12: Cluster prediction

Predict cluster per sentence for each trained topic model.

Output:

inference_by_topic with cluster labels.

## Step 13: Date-wise narrative distributions

For each topic:

1. group by date,
2. count cluster frequencies,
3. normalize into probability distributions.

These distributions are the narrative signature per date.

## Step 14: Drift and sentence-level narrative change

### 14.1 Drift score

For consecutive dates, compute Jensen-Shannon distance:

$$
JS(P, Q) = \frac{1}{2} KL(P \parallel M) + \frac{1}{2} KL(Q \parallel M), \quad M = \frac{P+Q}{2}
$$

Shift rule:

drift > 0.3

### 14.2 Sentence-level before-vs-after mapping

For each detected shift pair of dates:

1. take sentence embeddings from date_before and date_after,
2. build cosine similarity matrix,
3. greedily match each before sentence to an unused after sentence,
4. compute:
   1. similarity
   2. change_score = 1 - similarity
5. print top changed pairs (highest change first).

This gives sentence-level narrative movement, not just topic-level aggregate shift.

### 14.3 Topic-wise summary

Final summary printed per topic includes:

1. number of date pairs,
2. number of shifts above threshold,
3. max drift,
4. mean drift,
5. all drift values.

------------------------------------------------------------------------

## 5. Why Only War May Show Rich Output

With current user_article2.csv, most sentences are war-related.

Therefore:

1. dominant_topic often becomes War,
2. War has enough rows to train K-Means,
3. other topics have too few rows and are skipped.

This is expected behavior, not a bug, under dominant-topic routing.

------------------------------------------------------------------------

## 6. Current Output Artifacts

Primary outputs are notebook prints and in-memory objects:

1. sentence_df
2. labeled_sentence_df
3. filtered_by_topic
4. topic_models
5. inference_by_topic
6. article_df_by_topic
7. topic_drift_summary

Optional saved file:

kmeans_narrative_models.pkl (if models exist when save cell is run)

------------------------------------------------------------------------

## 7. Current Configuration Snapshot

1. Input mode: single user file
2. Input path: resolved dynamically for user_article2.csv
3. Embedding model: sentence-transformers/all-mpnet-base-v2
4. Embedding dimension: 768
5. Topics: 5
6. Cluster count per topic: 5
7. Drift metric: Jensen-Shannon distance
8. Drift threshold: 0.3
9. Sentence pair display per shift: controlled by SENTENCE_PAIR_LIMIT

------------------------------------------------------------------------

## 8. What This Version Is Best At

1. End-to-end user-input-only processing without external training CSV dependency.
2. Topic-aware drift analysis with explicit per-topic model availability checks.
3. Sentence-level narrative transition evidence for each detected drift event.
4. Robust handling of path and CSV schema issues in real notebook runs.
