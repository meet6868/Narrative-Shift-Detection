# Approach 4 with Balanced Data: User Inference & Narrative Shift Detection

## Overview

This document describes the **user inference stage** of Approach 4, where the trained Approach 4 TCL model — retrained on **balanced topic data** — is applied to 5 user-submitted articles to detect narrative shifts. This is the production-style testing of Approach 4 and represents the first real-world validation of the TCL pipeline on unseen input.

**Notebook:** `TCL_Pipeline_4.ipynb` (inference section)  
**Status:** ✅ Completed — inference run on Kaggle  
**Input:** `Output/Model_Testing/Approch_4/user_article2.csv`  
**Output:** `Output/Model_Testing/Approch_4/user_results_*.json` (per topic)

---

## 1. Data Balancing (Pre-Training Step)

Before retraining Approach 4, the training data was rebalanced using `Pre_Processing/Data_balancing.ipynb`.

### Why Balancing Was Needed

The original preprocessed topic files (from `Processed_Data/`) had significantly unequal topic sizes:

| Topic | Original Sentences |
|-------|--------------------|
| War | 490,123 |
| Economics | 277,886 |
| Technology | 190,543 |
| Health | 188,593 |
| Climate | 188,013 |

War had ~2.6× more data than Climate/Health/Technology. This imbalance risks the model overfitting on War narratives and underperforming on smaller topics.

### Balancing Pipeline (10 Stages)

The balancing notebook (`Data_balancing.ipynb`) implements the following pipeline:

```
Stage 1  → Load all CSV files from topic-labelled-w5 folder
Stage 2  → Combine into one dataframe; remove duplicates (on date + w5_embedding + main_sentence)
Stage 3  → Filter: max_topic_weight ≥ 0.35 AND topic_gap (top1 − top2) ≥ 0.20
Stage 4  → Detect dominant topic (idxmax of 5 topic columns)
Stage 5  → Extract year, month, year_month from date column
Stage 6  → Per-month balancing: iterative removal of weakest sentences from dominant topic
             until max_topic_count − min_topic_count ≤ MONTH_THRESHOLD (600)
Stage 7  → Per-year balancing: same iterative logic with YEAR_THRESHOLD (3000)
Stage 8  → Plot balanced topic distribution (bar chart)
Stage 9  → Plot balanced topic distribution by year (stacked bar)
Stage 10 → Export 5 topic CSV files (War.csv, Economics.csv, etc.)
```

### Balancing Algorithm Detail

```python
MONTH_THRESHOLD = 600   # max allowed difference between any two topics per month
YEAR_THRESHOLD  = 3000  # max allowed difference between any two topics per year

def iterative_balance(df, threshold):
    while True:
        topic_counts = df["dominant_topic"].value_counts()
        min_count  = topic_counts.min()
        max_topic  = topic_counts.idxmax()
        max_count  = topic_counts.max()
        if (max_count - min_count) <= threshold:
            break
        remove_n   = (max_count - min_count) - threshold
        # Remove the weakest (lowest topic weight) sentences from the dominant topic
        remove_rows = (df[df["dominant_topic"] == max_topic]
                       .sort_values(max_topic, ascending=True)
                       .head(remove_n).index)
        df = df.drop(remove_rows)
    return df
```

**Key design choice:** Sentences are removed from the overrepresented topic starting with the *weakest* topic-assignment sentences (lowest `max_topic_weight`), preserving the most topic-confident examples.

### Filtering Criteria Applied

| Criterion | Value | Purpose |
|-----------|-------|---------|
| `max_topic_weight` ≥ | 0.35 | Remove ambiguous sentences |
| `topic_gap` ≥ | 0.20 | Ensure dominant topic is clearly dominant |
| Duplicate removal | date + embedding + text | Remove redundant data |

---

## 2. Approach 4 Architecture (Retrained on Balanced Data)

The model architecture is unchanged from the original Approach 4. It is retrained from scratch on the balanced dataset.

| Parameter | Value |
|-----------|-------|
| Input dimension | 774 (768 SBERT W5 + 5 topic scores + 1 time gap) |
| Architecture | Temporal Transformer |
| Layers | 4 |
| Attention heads | 8 |
| Hidden dim | 512 |
| Projection dim | 512 |
| Temperature τ | 0.05 |
| Loss | NT-Xent + Topic Separation + Hard Negative Mining |
| Segmentation | Ruptures PELT + RBF kernel (penalty=0.1, min_size=2) |

**Loss function:**

$$\mathcal{L}_{\text{total}} = 1.5 \cdot \mathcal{L}_{\text{NT-Xent}} + 0.5 \cdot \mathcal{L}_{\text{topic-sep}} + 0.3 \cdot \mathcal{L}_{\text{hard-neg}}$$

---

## 3. User Inference Input

**File:** `Output/Model_Testing/Approch_4/user_article2.csv`

5 articles covering the **Russia-Ukraine war (Feb–Apr 2022)**, representing a clear real-world narrative evolution:

| Article | Date | Narrative Phase |
|---------|------|-----------------|
| a0 | 2022-02-15 | Military buildup / pre-invasion tensions |
| a1 | 2022-02-25 | Russian invasion launched — full-scale war begins |
| a2 | 2022-03-15 | Humanitarian crisis — refugees, civilian displacement |
| a3 | 2022-04-01 | Diplomatic talks — ceasefire negotiations in Istanbul |
| a4 | 2022-04-20 | Reconstruction / post-war recovery focus |

The 5 articles span **64 days**, 5 distinct narrative phases, and represent a controlled test of multi-phase shift detection.

---

## 4. Inference Pipeline

The model processes user articles as follows:

```
User CSV (date + article text)
    ↓
Sentence segmentation (NLTK sent_tokenize)
    ↓
W5 SBERT embedding (all-mpnet-base-v2)
    ↓
Topic scoring against ideal article embeddings
    ↓
TCL projection (774-dim → 512-dim)
    ↓
Per-topic narrative shift detection
    ↓
Sentence-level shift localization
    ↓
JSON output (per topic)
```

**Parameters used during inference:**

| Parameter | Value |
|-----------|-------|
| Shift threshold | 0.10 |
| Adaptive threshold | false (fixed) |
| Retrieval | Linear (sequential comparison) |
| Sentence context window | 3 sentences |

---

## 5. Inference Results

### 5.1 War Topic

**File:** `Output/Model_Testing/Approch_4/user_results_War (1).json`

| Field | Value |
|-------|-------|
| Articles processed | 5 |
| Sentences extracted | 28 |
| Unique days | 5 |
| Shifts detected | **2** |
| Threshold | 0.10 |

**Detected Shifts:**

| Shift | From Date | To Date | Shift Score | Similarity Score |
|-------|-----------|---------|-------------|-----------------|
| 1 | 2022-02-15 | 2022-03-15 | **0.4781** | 0.5392 |
| 2 | 2022-02-25 | 2022-04-01 | **1.0** | 0.2462 |

**Shift 1 — Pre-invasion tensions → Humanitarian crisis:**
- Sentence A (a0): *"International observers expressed concern that the growing tensions could lead to a broader conflict in the region."*
- Sentence B (a2): *"Emergency aid groups are struggling to provide food, shelter and medical assistance to displaced populations."*
- Narrative change: from concern/warning → active humanitarian emergency

**Shift 2 — Invasion launched → Diplomatic negotiations (strongest shift):**
- Sentence A (a1): *"Russian forces launched a large-scale military invasion of Ukraine early Thursday, attacking multiple cities with missile strikes and ground troops."*
- Sentence B (a3): *"International leaders encouraged continued dialogue, arguing that diplomacy offers the best path to ending the war and preventing further humanitarian suffering."*
- Narrative change: from military escalation → diplomatic resolution attempts
- **Similarity score 0.246** — very low, confirming maximal semantic divergence

---

### 5.2 Economics Topic

**File:** `Output/Model_Testing/Approch_4/user_results_Economics.json`

| Field | Value |
|-------|-------|
| Articles processed | 5 |
| Sentences extracted | 26 |
| Unique days | 5 |
| Shifts detected | **2** |
| Threshold | 0.10 |

**Detected Shifts:**

| Shift | From Date | To Date | Shift Score | Similarity Score |
|-------|-----------|---------|-------------|-----------------|
| 1 | 2022-02-15 | 2022-03-15 | **0.1226** | 0.5392 |
| 2 | 2022-03-15 | 2022-04-20 | **1.0** | 0.5308 |

**Shift 1 — Tensions → Humanitarian crisis (weak shift):**
- Sentence A (a0): *"International observers expressed concern that the growing tensions could lead to a broader conflict in the region."*
- Sentence B (a2): *"Emergency aid groups are struggling to provide food, shelter and medical assistance to displaced populations."*
- Shift score 0.123 — borderline, just above threshold (0.10)

**Shift 2 — Active conflict → Reconstruction/recovery (strongest shift):**
- Sentence A (a2): *"Meanwhile, Ukrainian forces continued to resist Russian advances in several cities, slowing the pace of the invasion."*
- Sentence B (a4): *"Global attention has increasingly shifted toward rebuilding Ukraine and supporting long-term recovery after months of conflict."*
- Narrative change: from active combat → post-conflict economic rebuilding narrative
- **Shift score 1.0** — maximum; model sees this as the dominant economic narrative shift

---

### 5.3 Health Topic

**File:** `Output/Model_Testing/Approch_4/user_results_Health.json`

| Field | Value |
|-------|-------|
| Articles processed | 5 |
| Sentences extracted | 18 |
| Unique days | 4 |
| Shifts detected | **1** |
| Threshold | 0.10 |

**Detected Shift:**

| Shift | From Date | To Date | Shift Score | Similarity Score |
|-------|-----------|---------|-------------|-----------------|
| 1 | 2022-03-15 | 2022-04-20 | **1.0** | 0.5308 |

**Shift — Humanitarian aid → Reconstruction:**
- Sentence A (a2): *"Meanwhile, Ukrainian forces continued to resist Russian advances in several cities, slowing the pace of the invasion."*
- Sentence B (a4): *"Global attention has increasingly shifted toward rebuilding Ukraine and supporting long-term recovery after months of conflict."*
- Fewer sentences extracted for Health topic (18 vs 28 for War) because articles are War-focused — Health relevance is lower
- Model detects a single strong shift (score 1.0) at the conflict→recovery transition

---

## 6. Cross-Topic Comparison

| Metric | War | Economics | Health |
|--------|-----|-----------|--------|
| Sentences extracted | 28 | 26 | 18 |
| Unique days | 5 | 5 | 4 |
| Shifts detected | 2 | 2 | 1 |
| Max shift score | 1.0 | 1.0 | 1.0 |
| Min shift score | 0.478 | 0.123 | 1.0 |
| Strongest shift pair | a1→a3 (invasion→diplomacy) | a2→a4 (combat→rebuild) | a2→a4 (combat→rebuild) |

**Key observations:**

1. **War topic detects earliest shift (Feb 15 → Mar 15):** The War topic model is most sensitive to the pre-invasion → humanitarian transition, which aligns with the training data distribution (War has the most sentences in balanced data too).

2. **Economics shift 1 is very weak (0.123):** Just above threshold. The model sees only a minor change between tensions and humanitarian phases from an economics lens — both involve conflict and instability without clear economic framing.

3. **Health topic sees fewer sentences:** Articles are not primarily Health-framed, so fewer sentences pass the Health topic filter. This is correct behavior — the model correctly applies topic-relevance filtering.

4. **Shift pair a2→a4 appears across Economics and Health:** Both topics agree that the most significant narrative shift is the transition from active conflict (Mar 15) to reconstruction/recovery (Apr 20), which involves economic aid pledges and humanitarian recovery — relevant to both topics.

5. **Shift score 1.0 appears in all three topics:** This is the normalized maximum — the model is maximally confident about the combat→rebuild transition being a genuine narrative shift regardless of topic lens.

---

## 7. Output File Structure

```
Output/Model_Testing/Approch_4/
├── user_article2.csv              ← 5 input articles (date, article text)
├── user_results_War (1).json      ← War topic inference output
├── user_results_Economics.json    ← Economics topic inference output
└── user_results_Health.json       ← Health topic inference output
```

**JSON schema (per topic):**
```json
{
  "topic": "War",
  "csv_path": "...",
  "num_articles": 5,
  "num_sentences": 28,
  "num_days": 5,
  "num_shifts": 2,
  "threshold": 0.1,
  "is_adaptive": false,
  "shifts": [
    {
      "shift_id": 1,
      "date_1": "YYYY-MM-DD",
      "date_2": "YYYY-MM-DD",
      "shift_score": 0.0–1.0,
      "distance": float
    }
  ],
  "sentence_shifts": [
    {
      "shift_id": 1,
      "date_1": "...", "date_2": "...",
      "article_id_1": "a0", "sentence_id_1": "a0_s5",
      "sentence_1": "...",
      "context_1": "... [±2 sentence window] ...",
      "article_id_2": "a2", "sentence_id_2": "a2_s3",
      "sentence_2": "...",
      "context_2": "...",
      "similarity_score": 0.0–1.0,
      "shift_score": 0.0–1.0
    }
  ]
}
```

---

## 8. Interpretation of Shift Scores

| Score Range | Interpretation |
|-------------|----------------|
| 0.10–0.30 | Weak shift — minor narrative change, borderline |
| 0.30–0.60 | Moderate shift — meaningful topic evolution |
| 0.60–0.90 | Strong shift — significant narrative transition |
| 0.90–1.00 | Maximum shift — fundamental narrative change |

The `distance` field is the raw TCL embedding distance (not normalized). The `shift_score` is the normalized version mapped to [0, 1] for interpretability.

---

## 9. Comparison: Original A4 vs A4 with Balanced Data

| Aspect | Original A4 (unbalanced) | A4 with Balanced Data |
|--------|--------------------------|----------------------|
| Training data | Raw topic CSVs (War: 490K, others: ~188K) | Balanced (max diff ≤ 3000/year, ≤ 600/month) |
| Bias risk | High (War-biased) | Reduced |
| Inference tested | No (training metrics only) | ✅ Yes — 5 articles, 3 topics |
| Shift detection | Evaluated on internal metrics | Real user article validation |
| Sentences (War) | 356 dynamic segments | 28 sentences → 2 shifts |
| Metrics available | Intra-sim, inter-sim, temporal | Shift scores + similarity scores per sentence pair |

---

## 10. Limitations of This Inference Run

1. **Only 3 of 5 topics tested:** Climate and Technology results not generated for this input (articles are War-focused; those topics likely had near-zero relevant sentences).

2. **Fixed threshold (0.10):** `is_adaptive: false` — threshold is not tuned per topic. A lower threshold was used (0.10 vs typical 0.3–0.4) to ensure detection in a 5-article test with limited data.

3. **Small input size:** 5 articles is a demonstration run. Production usage would process hundreds of articles over months.

4. **Sentence count discrepancy:** War extracts 28 sentences, Health only 18 from the same 5 articles — this reflects topic filtering, not a bug.

5. **No ground truth:** There are no human-annotated shift labels for these 5 articles, so precision/recall cannot be computed. Evaluation is qualitative.

---

## References

- Base approach: [Approach 4 - Article-Level Sentence Shift Detection](approach_4.md)
- Next step: [Approach 5 - Multi-Modal NER + Sentiment + FAISS](approach_5.md)
- Balancing notebook: `Pre_Processing/Data_balancing.ipynb`
- Inference inputs: `Output/Model_Testing/Approch_4/`
- Comprehensive report: `Compre_report.md` (Section: Approach 4 with Balanced Data)
