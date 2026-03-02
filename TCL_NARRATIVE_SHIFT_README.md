# 📘 README

# Generalized Topic-Conditioned Narrative Shift Detection using Temporal Contrastive Learning (TCL)

---

## 1. Problem Statement

Narratives in news media evolve over time.
However, this evolution is not always smooth. At certain moments, framing, emphasis, or thematic direction shifts significantly. Detecting such **narrative shifts** is important for:

* Media analysis
* Political framing studies
* Event segmentation modeling
* Longitudinal discourse tracking

This project proposes a **generalized, topic-conditioned Temporal Contrastive Learning (TCL) framework** to detect narrative shifts across multiple topics using sentence embeddings and topic probabilities.

---

## 2. Data Description

Each sentence in the dataset contains:

* `date`
* `w3_embedding` → 768-dimensional sentence embedding (window size = 3)
* `topic_probability_vector` → 5-dimensional soft topic distribution:

$$
T_i = [p_{war}, p_{health}, p_{econ}, p_{tech}, p_{climate}]
$$

No hard topic assignment is used.
Soft topic distributions are preserved.

---

## 3. Design Principles

The system is built on four key principles:

1. **Soft Topic Conditioning** (not one-hot)
2. **Weighted Semantic Aggregation**
3. **Irregular Temporal Modeling**
4. **Multi-scale Shift Detection (macro + micro)**

The model is designed to:

* Work across multiple topics
* Avoid catastrophic forgetting
* Generalize to unseen narrative trajectories
* Handle irregular publication gaps

---

## 4. Complete Pipeline

---

### 🔵 Stage 1 — Sentence-Level Representation

Each sentence:

$$
E_i \in \mathbb{R}^{768}
$$

$$
T_i \in \mathbb{R}^{5}
$$

No filtering is applied at this stage.

---

### 🔵 Stage 2 — Topic-Specific Daily Aggregation

We construct daily representations for each topic $k$.

#### Step 2.1 — Define Topic Weight

$$
w_i = T_i[k]
$$

Each sentence contributes proportionally to how strongly it belongs to topic $k$.

---

#### Step 2.2 — Weighted Semantic Mean

$$
Z_d^{(k)} = \frac{\sum w_i E_i}{\sum w_i}
$$

This produces the **daily semantic center**.

---

#### Step 2.3 — Weighted Topic Mean

$$
T_d^{(k)} = \frac{\sum w_i T_i}{\sum w_i}
$$

This preserves daily topic mixture.

---

#### Step 2.4 — Topic Presence Filtering

If:

$$
T_d^{(k)}[k] < 0.1
$$

The day is removed for topic $k$.

This ensures:

* Only meaningful topic days are modeled
* Noise days are excluded

---

### 🔵 Stage 3 — Temporal Gap Modeling

Since publication dates are irregular:

$$
\Delta t_d = date_d - date_{d-1}
$$

We encode:

$$
\tau_d = \log(1 + \Delta t_d)
$$

This allows the model to learn that larger time gaps naturally permit larger drift.

---

### 🔵 Stage 4 — Final Daily Representation

$$
X_d^{(k)} = [Z_d^{(k)} ; \tau_d ; T_d^{(k)}]
$$

Final dimension:

$$
768 + 1 + 5 = 774
$$

Each topic produces a chronological daily sequence.

---

### 🔵 Stage 5 — Window Construction

Using fixed window size:

$$
W = 30
$$

Sliding windows (stride = 1):

$$
W_t^{(k)} = [X_t^{(k)}, ..., X_{t+29}^{(k)}]
$$

Each window:

$$
(30, 774)
$$

Windows are created for all topics.

---

### 🔵 Stage 6 — Dataset Merging

All topic windows are merged into one dataset:

$$
\mathcal{D} = \bigcup_{k=1}^{5} \{W_t^{(k)}\}
$$

This enables training a **single generalized model**.

Sequential topic training is avoided to prevent catastrophic forgetting.

---

### 🔵 Stage 7 — Temporal Contrastive Learning Model

#### Architecture

**Input:** $(30, 774)$

1. Linear projection $(774 \rightarrow 256)$
2. Positional encoding
3. 2–3 Transformer encoder layers
4. Mean pooling over time
5. Projection head (MLP $256 \rightarrow 128$)
6. L2 normalization

**Output:**

$$
z_t \in \mathbb{R}^{128}
$$

---

#### Contrastive Objective

**Positive pairs:**

$$
(W_t, W_{t+1})
$$

**Negative pairs:**

All other windows in batch (including cross-topic).

**Loss:**

NT-Xent contrastive loss.

This forces:

$$
f(W_t) \approx f(W_{t+1})
$$

unless narrative structure shifts.

---

### 🔵 Stage 8 — Macro Drift Detection

For each topic:

$$
D_t = 1 - \cos(z_t, z_{t-1})
$$

Then normalize:

$$
Z_t = \frac{D_t - \mu}{\sigma}
$$

Major shifts detected via:

* Z-score > 2
  **OR**
* Top 5% percentile

These represent **macro narrative shifts**.

---

### 🔵 Stage 9 — Micro Pivot Detection

Within each macro shift region:

1. Extract original sentences
2. Compute:

$$
s_i = 1 - \cos(E_i, E_{i-1})
$$

3. Adjust for temporal spacing:

$$
s_i^{adj} = \frac{s_i}{\log(1+\Delta t)}
$$

The maximum $s_i^{adj}$ identifies the **pivot sentence** responsible for narrative break.

---

## 5. Why This Framework Is Strong

### ✔ Soft Topic Conditioning

Preserves multi-topic structure.
Enables smooth generalization.

### ✔ Weighted Pooling

Reduces noise.
Improves semantic stability.

### ✔ Irregular Time Handling

Prevents artificial drift inflation.

### ✔ Unified Model

Single model across topics.
No catastrophic forgetting.

### ✔ Multi-scale Detection

* Macro (window-level)
* Micro (sentence-level)

---

## 6. Advantages Over Traditional Methods

| Traditional Approach   | Our Approach              |
| ---------------------- | ------------------------- |
| Hard topic labeling    | Soft topic conditioning   |
| Static embeddings      | Temporal modeling         |
| Fixed time intervals   | Irregular time handling   |
| Single-scale detection | Multi-scale detection     |
| Per-topic models       | Generalized unified model |

---

## 7. Final Output

For each topic:

* Drift timeline
* Major shift dates
* Shift intensity
* Pivot sentence pair
* Local semantic break score

---

## 8. Summary

This project presents a scalable, topic-aware narrative shift detection system based on Temporal Contrastive Learning.

The framework:

* Learns smooth narrative trajectories
* Detects structural breaks
* Identifies linguistic pivot points
* Generalizes across topics

It combines probabilistic topic modeling with deep temporal representation learning in a principled way.

---

## 9. Implementation Details

### Dataset Structure

The system processes 5 topic-specific CSV files:
- `War.csv`
- `Health.csv`
- `Technology.csv`
- `Climate.csv`
- `Economics.csv`

Each file contains:
- `date`: Publication timestamp
- `w3_embedding`: 768-dimensional sentence embedding (comma-separated string)
- `main_sentence`: Original text
- Topic probability columns: `War`, `Health`, `Technology`, `Climate`, `Economics`

### Key Hyperparameters

- **Window size**: 30 days
- **Window stride**: 1 day
- **Topic threshold**: 0.1 (minimum topic probability for daily inclusion)
- **Model hidden size**: 256
- **Embedding projection**: 128
- **Transformer layers**: 2-3
- **Contrastive temperature**: 0.07
- **Training epochs**: 10-20
- **Batch size**: 32
- **Shift detection threshold**: Z-score > 2 or top 5%

### Environment Requirements

- Python 3.8+
- PyTorch 2.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- transformers (for SBERT embeddings)

### Kaggle Compatibility

The pipeline is designed to run on:
- **Kaggle GPU**: T4x2 accelerator
- **Local development**: CPU/GPU environments
- **Data path**: Auto-detection between local and Kaggle paths

---

## 10. Future Extensions

Potential improvements:

1. **Multi-head attention** for topic-specific feature extraction
2. **Graph neural networks** for cross-topic relationship modeling
3. **Adaptive window sizing** based on narrative velocity
4. **Hierarchical shift detection** (daily → weekly → monthly)
5. **Causality analysis** between topics during shift events
6. **Real-time streaming** adaptation for live news monitoring

---

## 11. References

This methodology draws inspiration from:

- Temporal Contrastive Learning (Chen et al., 2020)
- Sentence-BERT embeddings (Reimers & Gurevych, 2019)
- Soft topic modeling approaches
- Time series anomaly detection
- Narrative framing analysis in NLP

---

## 12. Contact & Attribution

**Project**: Generalized Topic-Conditioned Narrative Shift Detection using TCL  
**Course**: Information and Language Processing (INLP)  
**Institution**: [Your Institution]  
**Date**: March 2026  

For questions or collaboration inquiries, please refer to the course instructor.

---

**End of README**
