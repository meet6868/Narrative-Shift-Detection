# Semantic Narrative Drift Detection Pipeline — Midsem Project

A full end-to-end NLP pipeline that detects **when and how** the narrative around a topic shifts over time in a corpus of news articles. Built on top of **SBERT (`all-mpnet-base-v2`, 768-dim).**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Stage-by-Stage Breakdown](#4-stage-by-stage-breakdown)
   - [Stage 1 — Ideal Embeddings](#stage-1--ideal-embeddings-ideal_embeddingpy)
   - [Stage 2 — Topic Thresholds](#stage-2--topic-thresholds-topic_thresholdpy)
   - [Stage 3 — Drift Thresholds](#stage-3--drift-thresholds-drift_thresholdpy)
   - [Stage 4 — Drift Detection](#stage-4--drift-detection-detect_driftpy)
5. [Context-Window Models](#5-context-window-models)
6. [Calibration Outputs](#6-calibration-outputs)
7. [Article File Format](#7-article-file-format)
8. [How to Run](#8-how-to-run)
9. [Output Format](#9-output-format)

---

## 1. Project Overview

The pipeline answers the question:

> *"Has the way news articles talk about **[topic]** changed significantly between two time periods?"*

It does this by:

1. Building a **prototype vector** per topic from hand-curated "ideal" articles that represent the topic's core narrative.
2. Filtering any incoming article's sentences to keep only those **relevant to the topic** (cosine similarity ≥ topic threshold).
3. Grouping the filtered sentences into **5-day temporal windows** and mean-pooling each window into a single vector.
4. Computing the **semantic drift** between consecutive windows as `1 − cosine_similarity`.
5. Flagging a window transition as a **drift event** if the drift exceeds a calibrated threshold.
6. Identifying the **sentences most responsible** for the detected drift.

### Topics Supported

| Topic                | Subtopics                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------ |
| **Climate**    | Climate Policy, Environmental Disasters, Global Warming, Renewable Energy, Sustainability  |
| **Economics**  | Corporate Economy, Financial Markets, Fiscal Policy, Global Trade, Macroeconomics          |
| **Health**     | Chronic Disease, Healthcare Policy, Medical Research, Mental Health, Public Health         |
| **Technology** | AI & Automation, Consumer Tech, Digital Infrastructure, Tech Labor Impact, Tech Regulation |
| **War**        | Armed Conflict, Defense Technology, Geopolitics, Humanitarian Crisis, Peace Process        |

---

## 2. Directory Structure

```
SBERT_semantic_drift
│
├── ideal_embedding.py          ← Stage 1 : build topic prototype vectors
├── topic_threshold.py          ← Stage 2 : compute relevance thresholds
├── drift_threshold.py          ← Stage 3 : calibrate drift thresholds
├── detect_drift.py             ← Stage 4 : interactive drift detection (main script)
│
├── topic_prototypes.json       ← output of Stage 1 — topic → 768-dim centroid vector
├── topic_thresholds.json       ← output of Stage 2 — topic → cosine-sim filter threshold
├── drift_thresholds.json       ← output of Stage 3 — topic × model → drift threshold
└── drift_results.json          ← output of Stage 4 — latest detection run results
```

The `ideal_article/` folder (used by Stages 1 & 2) sits one level up:

```
ideal_article/
├── Climate/
│   ├── Climate_Policy/
│   ├── Environmental_Disasters/
│   ├── Global_Warming/
│   ├── Renewable_Energy/
│   └── Sustainability/
├── Economics/
│   ├── Corporate_Economy/
│   ├── Financial_Markets/
│   ├── Fiscal_Policy/
│   ├── Global_Trade/
│   └── Macroeconomics/
├── Health/
│   ├── Chronic_Disease/
│   ├── Healthcare_Policy/
│   ├── Medical_Research/
│   ├── Mental_Health/
│   └── Public_Health/
├── Technology/
│   ├── AI_Automation/
│   ├── Consumer_Tech/
│   ├── Digital_Infrastructure/
│   ├── Tech_Labor_Impact/
│   └── Tech_Regulation/
└── War/
    ├── Armed_Conflict/
    ├── Defense_Technology/
    ├── Geopolitics/
    ├── Humanitarian_Crisis/
    └── Peace_Process/
```

Each subtopic folder contains hand-curated `.txt` reference articles that define what "ideal" content for that topic looks like.

---

## 3. Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      CALIBRATION  (run once)                         │
│                                                                      │
│  ideal_article/              ALL_Combined_Data.csv                   │
│       │                              │                               │
│       ▼                              │                               │
│  ideal_embedding.py                  │                               │
│  → topic_prototypes.json             │                               │
│       │                              │                               │
│       ▼                              │                               │
│  topic_threshold.py                  │                               │
│  → topic_thresholds.json             │                               │
│       │                              │                               │
│       └──────────────┬───────────────┘                               │
│                      ▼                                               │
│              drift_threshold.py                                      │
│              → drift_thresholds.json                                 │
└──────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      DETECTION  (run anytime)                        │
│                                                                      │
│   User provides:  topic name  +  .txt article files                  │
│                       │                                              │
│                       ▼                                              │
│              detect_drift.py                                         │
│                                                                      │
│   Stage 1  →  Sentence segmentation  (NLTK)                          │
│   Stage 2  →  Build w1 / w3 / w5 context representations            │
│   Stage 3  →  SBERT encode  (all-mpnet-base-v2, GPU)                 │
│   Stage 4  →  Topic-relevance filter  (cosine-sim ≥ threshold)       │
│   Stage 5  →  Group into 5-day temporal windows                      │
│   Stage 6  →  Mean-pool window embeddings  (L2-normalised)           │
│   Stage 7  →  Compute drift between adjacent windows                 │
│   Stage 8  →  Identify drift-responsible sentences                   │
│                       │                                              │
│                       ▼                                              │
│              drift_results.json  +  console summary                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Stage-by-Stage Breakdown

### Stage 1 — Ideal Embeddings (`ideal_embedding.py`)

**Purpose:** Build one 768-dim prototype vector per topic that represents its "canonical" narrative.

**How it works:**

1. Recursively loads all `.txt` files from `ideal_article/<Topic>/**/*.txt`
2. Splits each article into sentences using NLTK `punkt_tab`
3. Filters out sentences shorter than 20 characters
4. Encodes all sentences in batches using SBERT (`all-mpnet-base-v2`)
5. Mean-pools all sentence embeddings per topic → one vector per topic
6. L2-normalises the prototype vector

**Why diverse subtopics?** Each topic has 5 subtopics with multiple articles each. This diversity ensures the prototype captures the *breadth* of the topic rather than a narrow slice, making the relevance filter more robust.

**Outputs:**

- `topic_prototypes.json` — `{"Climate": [768 floats], ...}`
- `topic_prototypes.pt` — PyTorch tensor dict for fast loading

---

### Stage 2 — Topic Thresholds (`topic_threshold.py`)

**Purpose:** For each topic, compute the minimum cosine similarity a sentence must have against the prototype to be considered "on-topic".

**Formula:**

$$
\text{threshold}_{\text{topic}} = \mu_{\text{sim}} - \alpha \cdot \sigma_{\text{sim}}
$$

where $\mu$ and $\sigma$ are the mean and standard deviation of cosine similarities of all ideal-article sentences against their topic prototype, and $\alpha = 0.5$ (tunable via `THRESHOLD_ALPHA`).

**Rationale:** Setting the threshold at `mean − 0.5·std` accepts sentences that are clearly on-topic while rejecting noise, without over-filtering borderline-relevant sentences.

**Output — `topic_thresholds.json`:**

| Topic      | Threshold |
| ---------- | --------- |
| Climate    | 0.4368    |
| Economics  | 0.4739    |
| Health     | 0.4407    |
| Technology | 0.4370    |
| War        | 0.4123    |

---

### Stage 3 — Drift Thresholds (`drift_threshold.py`)

**Purpose:** Calibrate what level of between-window semantic shift counts as "drift", for each topic × context-window model, using the full 32,570-article corpus.

**Pipeline per topic × model:**

1. Filter sentences by topic relevance (Stage 2 threshold)
2. Group filtered sentences into 5-day temporal windows
3. Mean-pool each window → window embedding (L2-normalised)
4. Compute drift between every pair of adjacent windows: `drift = 1 − cosine(w_t, w_{t−1})`
5. Collect all drift values across the entire corpus → distribution

**Formula:**

$$
\text{drift\_threshold}_{\text{topic,\,model}} = \mu_{\text{drift}} + \sigma_{\text{drift}}
$$

Flags only transitions one standard deviation above the typical corpus drift — roughly the top 16% of observed drift values.

**Output — `drift_thresholds.json`:**

| Topic      | w1     | w3     | w5     |
| ---------- | ------ | ------ | ------ |
| Climate    | 0.3208 | 0.3900 | 0.3953 |
| Economics  | 0.2772 | 0.3802 | 0.4189 |
| Health     | 0.2561 | 0.3885 | 0.4387 |
| Technology | 0.2553 | 0.3879 | 0.4086 |
| War        | 0.3183 | 0.4261 | 0.4804 |

---

### Stage 4 — Drift Detection (`detect_drift.py`)

**Purpose:** Given a user-chosen topic and a set of `.txt` news articles, run the full detection pipeline and report which 5-day windows show significant narrative drift.

**Interactive CLI flow:**

```
1. User selects topic by name or number  (1–5)
2. User enters paths to .txt article files  (min 5, one per line or space-separated)
3. Pipeline runs automatically for all 3 context-window models (w1 / w3 / w5)
4. Results printed to console + saved to drift_results.json
```

**8 internal stages:**

| Stage | Function                        | What it does                                                           |
| ----- | ------------------------------- | ---------------------------------------------------------------------- |
| 1     | `segment_sentences()`         | NLTK sentence split; deduplicates identical sentences on the same date |
| 2     | `build_context_windows()`     | Concatenates neighbouring sentences per model (w1/w3/w5)               |
| 3     | `encode_sentences()`          | SBERT encode → (N, 768) float32 L2-normalised embeddings              |
| 4     | `filter_topic_sentences()`    | Keeps only sentences with cosine-sim ≥ topic threshold                |
| 5     | `group_temporal_windows()`    | Bins sentences into 5-day time buckets                                 |
| 6     | `compute_window_embeddings()` | Mean-pools sentence embeddings per window, then L2-normalises          |
| 7     | `compute_drift()`             | `drift = 1 − cosine(w_t, w_{t−1})` for every adjacent window pair  |
| 8     | `find_drift_sentences()`      | Scores sentences by impact; returns top-N responsible for the shift    |

---

## 5. Context-Window Models

Three models run in parallel on every detection job:

| Model        | Half-width | Context text for sentence$S_i$                |
| ------------ | ---------- | ----------------------------------------------- |
| **w1** | 0          | $S_i$ alone                                   |
| **w3** | 1          | $S_{i-1} + S_i + S_{i+1}$                     |
| **w5** | 2          | $S_{i-2} + S_{i-1} + S_i + S_{i+1} + S_{i+2}$ |

- Context is **clipped at article boundaries** — never bleeds into the next article.
- All three models share the same sentence segmentation; only the text fed to SBERT differs.
- Wider context (w3/w5) → smoother embeddings → higher drift thresholds needed.

---

## 6. Calibration Outputs

### `topic_prototypes.json`

```json
{
  "Climate":    [0.021, -0.014, "...", 0.003],
  "Economics":  ["..."],
  "Health":     ["..."],
  "Technology": ["..."],
  "War":        ["..."]
}
```

Each value is a list of 768 floats, L2-normalised (norm ≈ 1.0).

### `topic_thresholds.json`

```json
{
  "Climate":    0.436828,
  "Economics":  0.473888,
  "Health":     0.440662,
  "Technology": 0.436980,
  "War":        0.412256
}
```

### `drift_thresholds.json`

```json
{
  "Technology": {
    "window_days": 5,
    "models": {
      "w1": {"mean": 0.163899, "std": 0.091357, "threshold": 0.255256},
      "w3": {"mean": 0.255919, "std": 0.131964, "threshold": 0.387883},
      "w5": {"mean": 0.270446, "std": 0.138158, "threshold": 0.408604}
    }
  }
}
```

---

## 7. Article File Format

Each `.txt` article file passed to `detect_drift.py` must follow this layout:

```
Date: 2024-03-15
Article body starts here. Any number of sentences on any number of lines.
Both bare format (2024-03-15) and the labelled format (Date: 2024-03-15) are accepted.
```

**Rules:**

- **Line 1:** date string (required)
- **Lines 2+:** article body text (free text, any length)
- At least **5 files** required per run to form ≥ 2 temporal windows
- Files with identical `(date, sentence)` pairs are automatically deduplicated with a log warning

---

## 8. How to Run

All scripts live in SBERT_semantic_drift. Run them from inside that folder:

```bash
cd SBERT_semantic_drift

```

### Step 1 — Build topic prototypes  *(run once)*

```bash
python ideal_embedding.py
# Reads  : ../ideal_article/<Topic>/**/*.txt
# Output : topic_prototypes.json
# Runtime: ~2 min  (RTX 3050, BATCH_SIZE=32, FP16)
```

### Step 2 — Compute topic thresholds  *(run once)*

```bash
python topic_threshold.py
# Reads  : topic_prototypes.json  +  ../ideal_article/
# Output : topic_thresholds.json
# Runtime: ~3 min  (RTX 3050)
```

### Step 3 — Calibrate drift thresholds  *(run once)*

```bash
python drift_threshold.py
# Reads  : ALL_Combined_Data.csv  +  topic_prototypes.json  +  topic_thresholds.json
# Output : drift_thresholds.json
# Runtime: ~18 min  (RTX 3050, 32,570 articles)
```

### Step 4 — Detect drift  *(run anytime)*

```bash
python detect_drift.py
```

Example interactive session:

```
═══════════════════════════════════════════════════════
  Narrative Drift Detector
═══════════════════════════════════════════════════════
  Available topics:
    [1] Climate
    [2] Economics
    [3] Health
    [4] Technology
    [5] War
───────────────────────────────────────────────────────
  Enter topic name or number: 5
  → Topic selected: War

  Enter paths to .txt article files.
  • You can paste multiple paths separated by spaces,
    or press Enter after each path.
  • Press Enter on a blank line when done.
───────────────────────────────────────────────────────
  Path(s): articles/jan_01.txt articles/jan_07.txt
  Path(s): articles/jan_15.txt articles/jan_22.txt
  Path(s): articles/feb_03.txt
  Path(s):

  ✓  5 article file(s) accepted.
```

---

## 9. Output Format

### Console summary (printed live per model)

```
══════════════════════════════════════════════════════════════════════
  Topic: War   │   Model: w1   │   Drift threshold: 0.3183
──────────────────────────────────────────────────────────────────────
  2024-01-03 → 2024-01-18   drift=0.4888   🔴 DRIFT DETECTED
    ↳ Top drift-driving sentences (current window):
      [1.956] At the same time, analysts note that geopolitical tensions…
  2024-01-18 → 2024-01-23   drift=0.7030   🔴 DRIFT DETECTED
    ↳ Top drift-driving sentences (current window):
      [6.701] Military officials announced that autonomous surveillance…
      [3.523] Meanwhile, international organizations are calling for…
  2024-01-23 → 2024-02-27   drift=0.6518   🔴 DRIFT DETECTED
    ↳ Top drift-driving sentences (current window):
      [3.883] Peace negotiations between rival nations resumed this week…
══════════════════════════════════════════════════════════════════════
```

### `drift_results.json` structure

```json
{
  "topic": "War",
  "n_articles": 10,
  "window_days": 5,
  "elapsed_s": 0.61,
  "w1": {
    "drift_threshold": 0.318281,
    "n_windows": 4,
    "n_drifts": 3,
    "results": [
      {
        "window_previous": "2024-01-03",
        "window_current":  "2024-01-18",
        "drift": 0.488800,
        "drift_threshold": 0.318281,
        "drift_detected": true,
        "previous_window_sentences": [
          {"sentence": "...", "sim_score": 0.91}
        ],
        "current_window_drift_sentences": [
          {"sentence": "...", "impact_score": 1.956},
          {"sentence": "...", "impact_score": 1.732}
        ]
      }
    ]
  },
  "w3": {"..."},
  "w5": {"..."}
}
```

**Key fields:**

| Field                              | Meaning                                                                                        |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| `drift`                          | `1 − cosine(window_t, window_{t−1})` — 0 = identical, 1 = completely different            |
| `drift_detected`                 | `true` if `drift > drift_threshold`                                                        |
| `impact_score`                   | `cos(sent, curr_window) / cos(sent, prev_window)` — higher = more responsible for the shift |
| `sim_score`                      | cosine similarity of sentence to its window embedding (no-drift case)                          |
| `previous_window_sentences`      | top-N anchor sentences representing the old narrative                                          |
| `current_window_drift_sentences` | top-N sentences most responsible for the narrative shift                                       |

---

*Built with SBERT · PyTorch · NLTK · pandas*
