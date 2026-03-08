# Narrative Drift Detection using K-Means

## Objective

Detect narrative changes in news articles over time by analyzing how the
distribution of narrative clusters changes.

The system identifies narrative drift when the semantic structure of
articles changes significantly between time periods.

------------------------------------------------------------------------

# Dataset

Training dataset: Climate.csv (\~1.26GB)

Columns:

  Column          Description
  --------------- ---------------------------
  date            article timestamp
  sentence_id     sentence identifier
  main_sentence   sentence text
  w3_embedding    768-dimensional embedding
  War             topic probability
  Health          topic probability
  Technology      topic probability
  Climate         topic probability
  Economics       topic probability

Important: Embeddings already exist in the dataset → No embedding model
needed.

------------------------------------------------------------------------

# Approach Overview

Pipeline:

Articles\
↓\
Sentence Embeddings\
↓\
K-Means Clustering\
↓\
Cluster Distribution over Time\
↓\
Jensen-Shannon Divergence\
↓\
Narrative Drift Detection

------------------------------------------------------------------------

# Step 1 --- Parse Embeddings

Embeddings are stored as strings such as: "\[0.021, -0.15, 0.004 ...\]"

Convert to numpy vectors using:

ast.literal_eval() → numpy array

Embedding dimension: 768

------------------------------------------------------------------------

# Step 2 --- Topic Filtering

Restrict dataset to a specific topic.

Example:

TOPIC = Climate

Filtering condition:

train_df = train_df\[train_df\["Climate"\] \> 0.3\]

Purpose: Ensure all sentences belong to the same topic.

------------------------------------------------------------------------

# Step 3 --- Train K-Means Model

Cluster sentence embeddings to discover latent narrative patterns.

Parameter: K = 5 clusters

Model:

kmeans = KMeans(n_clusters=5) kmeans.fit(X_train)

Example clusters may represent:

Cluster 0 → scientific discussion\
Cluster 1 → climate policy\
Cluster 2 → environmental damage\
Cluster 3 → economic impacts\
Cluster 4 → international agreements

------------------------------------------------------------------------

# Step 4 --- Assign Cluster Labels

Each sentence is mapped to the nearest cluster centroid.

Example:

Sentence A → cluster 2\
Sentence B → cluster 2\
Sentence C → cluster 4\
Sentence D → cluster 1

------------------------------------------------------------------------

# Step 5 --- Narrative Distribution per Date

Group sentences by date and compute cluster distribution.

Example:

Date 1: \[0.6, 0.2, 0.1, 0.1, 0.0\]

Date 2: \[0.1, 0.1, 0.6, 0.1, 0.1\]

These vectors represent the narrative structure for each time period.

------------------------------------------------------------------------

# Step 6 --- Drift Detection

Use Jensen-Shannon divergence to measure change between distributions.

JS(P \|\| Q) = ½ KL(P \|\| M) + ½ KL(Q \|\| M)

Where: M = (P + Q) / 2

Range: 0 → identical distributions\
1 → completely different distributions

------------------------------------------------------------------------

# Step 7 --- Drift Threshold

Narrative shift is detected when:

JS divergence \> 0.3

Example:

JS = 0.12 → no drift\
JS = 0.83 → strong drift

------------------------------------------------------------------------

# Step 8 --- Output Interpretation

Example result:

Narrative Shift Detected From: 2015-04-21 To: 2015-09-16 Drift Score:
0.83

Meaning: Narrative structure changed significantly between these dates.

------------------------------------------------------------------------

# Final Model Summary

Model type: Unsupervised Narrative Drift Detection

Configuration:

Embedding dimension: 768\
Clustering algorithm: K-Means\
Number of clusters: 5\
Distance metric: Euclidean\
Drift metric: Jensen-Shannon divergence\
Drift threshold: 0.3

Pipeline:

Sentence embeddings\
→ K-Means clustering\
→ Cluster distribution per time\
→ Jensen-Shannon divergence\
→ Narrative drift detection

------------------------------------------------------------------------

# What the Model Detects

The model detects:

• Narrative shift\
• Topic evolution\
• Discourse change\
• Semantic distribution drift

Note: The model does not directly detect fine-grained framing changes
but captures major narrative structure changes over time.
