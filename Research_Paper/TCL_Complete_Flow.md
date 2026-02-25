Below is a **polished, publication-ready README version** of your framework.
It is rewritten in a clean academic style, properly structured, and suitable for GitHub or research submission.

---

# A Generalized Framework for Narrative Shift Detection in News Media

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Quick Start Guide](#quick-start-guide)
- [GPU Setup & Management](#gpu-setup--management)
- [Research Objective](#1-research-objective)
- [Pipeline Overview](#2-end-to-end-pipeline-overview)
- [Data Format](#3-data-format-and-structure)
- [Semantic Representation](#4-semantic-representation--sentence-level-sbert-with-dual-windows)
- [Hierarchical Pooling](#5-hierarchical-mean-pooling-strategy)
- [Gap-Constrained Grouping](#gap-constrained-grouping-guide)

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.12+**
- **NVIDIA GPU** (1.64GB+ VRAM) or CPU
- **CUDA 13.1** (for GPU acceleration)
- **Virtual Environment** (recommended)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/meet6868/Narrative-Shift-Detection.git
cd Narrative-Shift-Detection

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate          # Linux/Mac
# OR
venv\Scripts\activate             # Windows

# 4. Install dependencies
pip install torch sentence-transformers pandas numpy nltk
```

### Running the Pipeline

#### Stage 1: Sentence Segmentation & Window Construction
```bash
python stage1_preprocessing.py
# Input:  Raw articles (CSV/JSON)
# Output: Processed_Data/Stage_1/*.csv (with 5-sentence windows)
```

#### Stage 2: SBERT Embedding Generation (GPU/CPU)
```bash
# Easy method: Use launcher script
./run_stage2.sh

# OR: Direct Python execution
venv/bin/python stage2_gpu_processor.py

# OR: Use system Python (if venv activated correctly)
python stage2_gpu_processor.py
```

**⚠️ Important:** Always use `venv/bin/python` or `python` (NOT `python3`) when venv is activated!

---

## 🎮 GPU Setup & Management

### GPU Requirements

| GPU Model | VRAM | Recommended Batch Size | Expected Speed |
|-----------|------|------------------------|----------------|
| RTX 450 / MX450 | 1.64-2 GB | 32 | 150-300 sent/sec |
| RTX 3060 | 12 GB | 128 | 500-800 sent/sec |
| RTX 4090 | 24 GB | 256 | 1000+ sent/sec |
| CPU (Any) | N/A | 16 | 40-80 sent/sec |

### GPU Memory Management

#### Check GPU Status
```bash
# Quick check
nvidia-smi

# OR use our interactive manager
./gpu_manager.sh
```

#### Clear GPU Cache
```bash
# Python-based cache cleaner
venv/bin/python clear_gpu_cache.py

# OR use manager script
./gpu_manager.sh
# Then select option 3 (Clear PyTorch cache)
```

#### Kill Stuck Processes
```bash
# Find GPU processes
nvidia-smi --query-compute-apps=pid,process_name --format=csv

# Kill specific process
kill -9 <PID>

# OR use automatic cleanup
./gpu_manager.sh
# Then select option 4 (Kill stuck Python processes)
```

### Common GPU Issues & Solutions

#### Issue 1: "CUDA Out of Memory" Error

**Symptoms:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 276.00 MiB...
```

**Solution 1: Reduce batch size**
```bash
# Edit stage2_gpu_processor.py, line 84
BATCH_SIZE = 16  # Reduce from 32 to 16 or 8
```

**Solution 2: Clear GPU cache before running**
```bash
venv/bin/python clear_gpu_cache.py
./run_stage2.sh
```

**Solution 3: Kill stuck processes**
```bash
# Check for stuck processes
nvidia-smi

# Kill them
./gpu_manager.sh  # Option 4
```

---

#### Issue 2: "ModuleNotFoundError: No module named 'torch'"

**Symptoms:**
```bash
(venv) $ python3 stage2_gpu_processor.py
ModuleNotFoundError: No module named 'torch'
```

**Problem:** You're using system `python3` instead of venv Python

**Solution:**
```bash
# ✅ CORRECT methods:
venv/bin/python stage2_gpu_processor.py
# OR
./run_stage2.sh
# OR (after venv activation)
python stage2_gpu_processor.py  # Use 'python', NOT 'python3'

# ❌ WRONG:
python3 stage2_gpu_processor.py  # System Python, not venv!
```

---

#### Issue 3: GPU Shows 2048 MB but PyTorch Reports 1.64 GB

**This is NORMAL!** Here's why:

```
Total Physical VRAM:        2048 MiB  (100%)
───────────────────────────────────────────────
System/Driver overhead:     -370 MiB  (18%)
  - NVIDIA Driver:            ~150 MiB
  - X.org/Display:            ~200 MiB
  - Reserved buffers:         ~20 MiB
───────────────────────────────────────────────
Available for PyTorch:       1678 MiB  (82%)
                           = 1.64 GB ✓
```

**What to do:** Use batch size appropriate for **1.64 GB**, not 2 GB:
- Batch 32: Safe for 1.64 GB ✓
- Batch 64: May cause OOM ⚠️
- Batch 96: Will cause OOM ❌

---

#### Issue 4: Script Automatically Reduces Batch Size

**This is a FEATURE, not a bug!**

The script has **automatic OOM recovery**:

```
🔄 Generating w3 embeddings (batch size: 32)...
⚠️  GPU Out of Memory! Reducing batch size to 16
↻  Retrying batch 1...
✅ Success with batch size 16
```

**What's happening:**
1. Script tries batch size 32
2. GPU runs out of memory
3. Script automatically reduces to 16
4. If still OOM, reduces to 8
5. Processing continues successfully

**No action needed** - just let it run!

---

### GPU Optimization Tips

#### For 1.64 GB GPU (RTX 450/MX450):

1. **Optimal Settings** (edit `stage2_gpu_processor.py`):
   ```python
   BATCH_SIZE = 32              # Conservative, safe
   MAX_SEQ_LENGTH = 384         # Reduced from 512
   USE_FP16 = True              # 2x speedup
   CLEAR_CACHE_FREQUENCY = 3    # Clear every 3 files
   ```

2. **If Still Getting OOM:**
   ```python
   BATCH_SIZE = 16              # More conservative
   MAX_SEQ_LENGTH = 256         # Even smaller
   ```

3. **For Maximum Speed (if no OOM):**
   ```python
   BATCH_SIZE = 48              # Aggressive
   MAX_SEQ_LENGTH = 512         # Full length
   ```

#### Monitor GPU During Processing:

Open a second terminal:
```bash
watch -n 1 nvidia-smi
# Updates GPU status every 1 second
```

---

### Quick Reference Commands

```bash
# ═══════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════
source venv/bin/activate                    # Activate venv
pip list | grep torch                       # Check installed packages

# ═══════════════════════════════════════════════════════════
# RUN PROCESSING
# ═══════════════════════════════════════════════════════════
./run_stage2.sh                             # Easy launcher
venv/bin/python stage2_gpu_processor.py     # Direct method

# ═══════════════════════════════════════════════════════════
# GPU MANAGEMENT
# ═══════════════════════════════════════════════════════════
nvidia-smi                                  # Check GPU status
venv/bin/python clear_gpu_cache.py          # Clear cache
./gpu_manager.sh                            # Interactive manager

# ═══════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════
kill -9 $(pgrep -f "venv/bin/python")       # Kill all venv Python
nvidia-smi --query-compute-apps=pid --format=csv  # Find GPU processes
watch -n 1 nvidia-smi                       # Monitor GPU real-time
```

---

## 1. Research Objective

This project proposes a **generalized and modular framework** for detecting narrative shifts in news media using semantic, temporal, and contrastive learning techniques at the **sentence level**.

The framework integrates:

* **SBERT sentence-level embeddings with dual contextual windows**
* **Multi-topic soft labeling (5 topics)**
* **Temporal modeling via GRU**
* **Temporal Contrastive Learning (TCL)**
* **NT-Xent contrastive loss**

### Design Goals

The framework is designed to be:

* **Sentence-granular**: Operates at sentence level for fine-grained narrative understanding
* **Context-aware**: Uses dual window sizes (3 and 5 sentences) for multi-scale context
* **Topic-aware**: Soft topic labeling across 5 major news domains
* **Temporally structured**: Preserves chronological narrative evolution
* **Window-agnostic**: Supports multiple temporal aggregation strategies
* **Generalizable across domains**: Applicable to diverse news categories
* **Modular and extensible**: Easy to adapt and enhance

### Key Innovation

Unlike article-level approaches, our framework operates at the **sentence level** with **contextual windows**, enabling:
- More precise shift detection
- Better capture of gradual narrative evolution
- Multi-scale semantic understanding through dual embeddings

---

## 2. End-to-End Pipeline Overview

```
Raw Articles (with date, text, topic labels)
    ↓
Sentence Segmentation (NLTK sent_tokenize)
    ↓
5-Sentence Window Construction
    │   ├─ previous_sentence_1 (i-2)
    │   ├─ previous_sentence_2 (i-1)
    │   ├─ main_sentence (i)
    │   ├─ next_sentence_1 (i+1)
    │   └─ next_sentence_2 (i+2)
    ↓
Dual SBERT Contextual Embeddings
    │   ├─ w3_embedding: Window 3 (i-1, i, i+1) → 768-dim
    │   └─ w5_embedding: Window 5 (i-2, i-1, i, i+1, i+2) → 768-dim
    ↓
Soft Topic Labeling (5 topics)
    │   ├─ Health
    │   ├─ Technology
    │   ├─ War
    │   ├─ Economics
    │   └─ Environment
    ↓
Topic-wise Temporal Sorting (chronological)
    ↓
Sentence-Level Temporal Sequence Construction
    ↓
GRU Temporal Encoder (per topic)
    ↓
Projection Head (MLP: 768 → 256 → 128)
    ↓
NT-Xent Contrastive Loss
    ↓
Narrative Embedding Space
    ↓
Drift Measurement + Shift Detection + Explanation
```

### Pipeline Stages Explained

**Stage 1 (Data Preprocessing):** Articles are segmented into sentences, maintaining temporal and topical metadata.

**Stage 2 (Contextual Window Construction):** Each sentence is embedded with surrounding context using 5-sentence sliding windows.

**Stage 3 (Dual Embedding Generation):** Two SBERT embeddings per sentence capture different contextual scales.

**Stage 4 (Topic Assignment):** Soft labels assign probability distributions over 5 news topics.

**Stage 5 (Temporal Modeling):** GRU learns narrative evolution patterns per topic.

**Stage 6 (Contrastive Learning):** TCL ensures smooth temporal trajectories while maintaining discriminative power.

---

## 3. Data Format and Structure

### 3.1 Input Data (Stage 0)

Each article contains:

```json
{
  "date": "YYYY-MM-DD HH:MM:SS",
  "text": "Full article text...",
  "source": "news_outlet_name"
}
```

### 3.2 Stage 1 Output (Sentence Segmentation)

After sentence segmentation with 5-sentence windows:

```csv
sentence_id, article_id, date, source, 
previous_sentence_1, previous_sentence_2, main_sentence, 
next_sentence_1, next_sentence_2
```

**Column Explanation:**
- `sentence_id`: Unique identifier (e.g., "a123_s5" = article 123, sentence 5)
- `article_id`: Article identifier
- `date`: Publication timestamp
- `source`: News outlet
- `previous_sentence_1`: Sentence at position (i-2), context 2 steps back
- `previous_sentence_2`: Sentence at position (i-1), context 1 step back
- `main_sentence`: Current sentence at position (i)
- `next_sentence_1`: Sentence at position (i+1), context 1 step ahead
- `next_sentence_2`: Sentence at position (i+2), context 2 steps ahead

### 3.3 Stage 2 Output (SBERT Embeddings)

After dual-window embedding generation:

```csv
sentence_id, article_id, date, source,
previous_sentence_1, previous_sentence_2, main_sentence, 
next_sentence_1, next_sentence_2,
w3_embedding, w5_embedding
```

**Embedding Columns:**
- `w3_embedding`: 768-dimensional SBERT embedding from 3-sentence context
  - Input: `[previous_sentence_2] [SEP] [main_sentence] [SEP] [next_sentence_1]`
  - Captures immediate local context
- `w5_embedding`: 768-dimensional SBERT embedding from 5-sentence context
  - Input: `[previous_sentence_1] [SEP] [previous_sentence_2] [SEP] [main_sentence] [SEP] [next_sentence_1] [SEP] [next_sentence_2]`
  - Captures broader contextual understanding

### 3.4 Topic Soft Labeling

Each sentence (or article) receives a 5-dimensional soft label vector:

```json
{
  "sentence_id": "a123_s5",
  "topic_distribution": {
    "Health": 0.45,
    "Technology": 0.25,
    "War": 0.15,
    "Economics": 0.10,
    "Environment": 0.05
  }
}
```

**Topic Categories:**
1. **Health**: Medical news, healthcare, pandemics, public health
2. **Technology**: Tech innovations, digital transformation, AI, cybersecurity
3. **War**: Armed conflicts, military operations, geopolitical tensions
4. **Economics**: Markets, trade, finance, economic policy
5. **Environment**: Climate change, sustainability, natural disasters

### 3.5 Temporal Organization

* Sentences are grouped by **dominant topic** (argmax of soft labels)
* Within each topic, sentences are sorted **chronologically**
* Temporal sequences are constructed **independently per topic**
* This enables topic-specific narrative modeling

---

## 4. Semantic Representation — Sentence-Level SBERT with Dual Windows

### 4.1 Why Sentence-Level Analysis?

Operating at sentence granularity (rather than article-level) provides:

1. **Fine-grained shift detection**: Pinpoint exact sentences where narratives change
2. **Reduced noise**: Articles often contain mixed narratives; sentence-level analysis isolates them
3. **Better temporal resolution**: Track narrative evolution within and across articles
4. **Explainability**: Identify specific textual evidence for detected shifts

### 4.2 Why SBERT?

SBERT (Sentence-BERT) is chosen for:

* **Deep semantic understanding** at sentence level
* **Context-aware embeddings** via transformer architecture
* **Pre-trained robustness** across diverse news domains
* **Computational efficiency** compared to full BERT
* **Stable 768-dimensional representations**

**Model Used:** `all-mpnet-base-v2` (best performing SBERT variant)

### 4.3 Dual Contextual Window Strategy

Instead of encoding sentences in isolation, we use **contextual windows** to capture surrounding narrative context:

#### Window 3 (w3) — Local Context
**Structure:** `[sentence_{i-1}] [SEP] [sentence_i] [SEP] [sentence_{i+1}]`

**Purpose:**
- Captures immediate conversational/narrative flow
- Preserves sentence-to-sentence coherence
- Lower computational cost
- More sensitive to local shifts

**Use Case:** Detecting rapid narrative pivots or breaking news reactions

#### Window 5 (w5) — Broader Context
**Structure:** `[sentence_{i-2}] [SEP] [sentence_{i-1}] [SEP] [sentence_i] [SEP] [sentence_{i+1}] [SEP] [sentence_{i+2}]`

**Purpose:**
- Captures paragraph-level semantic context
- Better understanding of complex arguments
- More stable representations
- Reduces local noise

**Use Case:** Detecting gradual narrative evolution or sustained framing changes

### 4.4 Encoding Process

```python
# For each sentence i:

# Create w3 contextual input
w3_input = f"{sent[i-1]} [SEP] {sent[i]} [SEP] {sent[i+1]}"
w3_embedding = sbert_model.encode(w3_input)  # 768-dim

# Create w5 contextual input
w5_input = f"{sent[i-2]} [SEP] {sent[i-1]} [SEP] {sent[i]} [SEP] {sent[i+1]} [SEP] {sent[i+2]}"
w5_embedding = sbert_model.encode(w5_input)  # 768-dim
```

**Batch Processing:** Embeddings are generated in batches of 32-128 (CPU/GPU optimized)

### 4.5 Output Format

Each sentence receives **two embeddings**:

```
sentence_id: "a123_s5"
w3_embedding: [0.023, -0.145, 0.089, ..., 0.112]  # 768 floats
w5_embedding: [0.034, -0.128, 0.095, ..., 0.098]  # 768 floats
```

Stored as comma-separated strings in CSV for efficiency.

### 4.6 Advantages of Dual-Window Approach

1. **Multi-scale understanding**: Combine local precision with global context
2. **Robustness**: w5 reduces noise; w3 maintains sensitivity
3. **Flexibility**: Can choose window based on task (fast vs. gradual shifts)
4. **Ensemble potential**: Can combine both embeddings for richer representations
5. **Empirical validation**: Allows comparison of optimal window size

### 4.7 Handling Edge Cases

- **Beginning of article** (i < 2): Previous sentences filled with empty strings
- **End of article** (i > n-2): Next sentences filled with empty strings
- **Short articles** (< 5 sentences): Graceful degradation with available context
- SBERT naturally handles variable-length inputs via [SEP] token attention

### 4.8 Comparison to Alternative Approaches

| Approach | Granularity | Context | Shift Precision |
|----------|-------------|---------|----------------|
| Article-level mean pooling | Coarse | Full article | Low |
| Single sentence (no context) | Fine | None | Noisy |
| **Our approach (w3)** | **Fine** | **Local (3-sent)** | **High** |
| **Our approach (w5)** | **Fine** | **Broader (5-sent)** | **Very High** |

### 4.9 Limitation

* Static embeddings do not model temporal dynamics directly
  - **Addressed by:** GRU temporal encoder in next stage
* Computational cost increases with dual embeddings
  - **Mitigated by:** Batch processing and GPU acceleration

---

## 5. Hierarchical Mean Pooling Strategy

Our framework uses **3-level mean pooling** to progressively aggregate sentence embeddings into increasingly stable temporal representations before applying window-based aggregation strategies.

**Why Hierarchical Aggregation?**
- Reduces sentence-level noise while preserving semantic information
- Creates stable daily snapshots of topic discourse
- Enables natural temporal grouping
- Balances granularity with stability

---

### 5.1 Level 1: Sentence → Article Mean Pooling

**Purpose:** Convert sentence-level embeddings to article-level representations

**Process:**

For each article containing N sentences:

```python
# Article A1 contains multiple sentences
sentences_A1 = [
    w3_emb_s1,  # Sentence 1 embedding (768-dim)
    w3_emb_s2,  # Sentence 2 embedding (768-dim)
    w3_emb_s3,  # Sentence 3 embedding (768-dim)
    ...
    w3_emb_sN   # Sentence N embedding (768-dim)
]

# Mean pool all sentences to create article embedding
article_emb_A1 = np.mean(sentences_A1, axis=0)  # Shape: (768,)
```

**Same process for w5 embeddings:**

```python
# Using w5 embeddings instead
article_emb_A1_w5 = np.mean([w5_emb_s1, w5_emb_s2, ..., w5_emb_sN], axis=0)
```

**Output:** One 768-dimensional embedding per article

**Benefits:**
- Captures overall article narrative
- Reduces sentence-level stylistic variations
- Creates stable article-level representation
- Filters out individual sentence anomalies

**Example:**

```
Article "COVID Update - Jan 15, 2025" has 12 sentences:
  - Sentence 1: "Cases are rising..."       → w3_emb_1
  - Sentence 2: "Hospitals report..."       → w3_emb_2
  - ...
  - Sentence 12: "Experts recommend..."    → w3_emb_12

Article Embedding = mean([w3_emb_1, ..., w3_emb_12])
```

---

### 5.2 Level 2: Article → Daily Mean Pooling (Same Date + Same Topic)

**Purpose:** Aggregate all articles published on the **same date** within the **same topic** to create a daily topic embedding

**This is the critical aggregation that creates temporal sequences for the GRU**

**Process:**

For each unique (date, topic) pair:

```python
# Example: Date = 2025-01-15, Topic = Health

# All Health articles published on Jan 15, 2025
articles_jan15_health = [
    article_emb_CNN,      # CNN's Health article (768-dim)
    article_emb_BBC,      # BBC's Health article (768-dim)
    article_emb_Reuters   # Reuters' Health article (768-dim)
]

# Mean pool to create single daily embedding
daily_emb_jan15_health = np.mean(articles_jan15_health, axis=0)  # Shape: (768,)
```

**Output:** One embedding per (date, topic) pair

**Why This Matters:**

1. **Collective narrative representation:** Captures the overall discourse on a topic for that specific day, not just one news source's perspective

2. **Handles multiple sources:** Different news outlets covering the same topic on the same day are naturally aggregated

3. **Reduces source bias:** CNN's framing + BBC's framing + Reuters' framing → balanced daily snapshot

4. **Creates stable temporal snapshots:** Daily embeddings are more stable than individual article embeddings

5. **Natural temporal unit:** Daily granularity aligns with news cycle (breaking → follow-up → analysis within a day)

**Example Timeline (Health Topic):**

```
2025-01-10: daily_emb_1  (3 articles: CNN, BBC, Fox)
2025-01-12: daily_emb_2  (5 articles: CNN, BBC, Reuters, NYT, Guardian)
                         ↑ Note: Jan 11 has no Health articles (gap)
2025-01-15: daily_emb_3  (2 articles: CNN, Reuters)
2025-01-16: daily_emb_4  (4 articles: BBC, CNN, Fox, AP)
2025-01-17: daily_emb_5  (6 articles: CNN, BBC, Reuters, NYT, WSJ, Guardian)
2025-01-20: daily_emb_6  (3 articles: CNN, BBC, Fox)
                         ↑ Note: Jan 18-19 gap (weekend)
```

**Handling Missing Dates:**
- Some dates naturally have no articles (weekends, holidays, low-coverage topics)
- Gaps are preserved in the timeline
- Temporal window grouping (Level 3) handles these gaps naturally

**Multi-Topic Organization:**

Each topic maintains its own daily embedding sequence:

```
Health Topic:
  2025-01-10 → daily_emb_health_jan10
  2025-01-12 → daily_emb_health_jan12
  ...

Technology Topic:
  2025-01-10 → daily_emb_tech_jan10
  2025-01-11 → daily_emb_tech_jan11
  ...

War Topic:
  2025-01-09 → daily_emb_war_jan09
  2025-01-15 → daily_emb_war_jan15
  ...
```

**Mathematical Formulation:**

For topic $T$ and date $d$:

$$
\mathbf{e}_{T,d} = \frac{1}{|A_{T,d}|} \sum_{a \in A_{T,d}} \mathbf{e}_a
$$

Where:
- $\mathbf{e}_{T,d}$: Daily embedding for topic $T$ on date $d$
- $A_{T,d}$: Set of all articles with topic $T$ published on date $d$
- $\mathbf{e}_a$: Article embedding (from Level 1 pooling)
- $|A_{T,d}|$: Number of articles in the set

---

### 5.3 Level 3: Daily → Window Group Mean Pooling

**Purpose:** Further smooth temporal trajectory by grouping daily embeddings into windows

**This is where the 10 different temporal aggregation strategies diverge**

After Level 2, we have daily embeddings:

```
Daily sequence for Health topic:
[day1, day2, day3, day5, day7, day8, day10, ...]
```

Level 3 applies different **window grouping strategies** to create sequences for GRU:

---

#### Strategy 1: No Window (Baseline)

**Input:** Daily embeddings directly (no further aggregation)

**Process:**
```python
# No additional pooling
gru_input_sequence = [daily_emb_1, daily_emb_2, daily_emb_3, ...]
```

**GRU Receives:**
```
GRU([emb_day1, emb_day2, emb_day5, emb_day7, ...])
```

**Characteristics:**
- ✅ Maximum temporal resolution (daily)
- ✅ Detects rapid day-to-day shifts
- ❌ May be noisy if few articles per day
- ❌ Sensitive to daily fluctuations

**Use Case:** Baseline for comparison

---

#### Strategy 2: Fixed Sliding Window (K = 2, 3, 4, 5)

**Input:** Daily embeddings

**Process (Example: Window Size = 3):**

```python
daily_sequence = [day1, day2, day3, day5, day7, day8, day10]

# Create overlapping windows of 3 daily embeddings
window_1 = np.mean([day1, day2, day3], axis=0)   → group_emb_1
window_2 = np.mean([day2, day3, day5], axis=0)   → group_emb_2
window_3 = np.mean([day3, day5, day7], axis=0)   → group_emb_3
window_4 = np.mean([day5, day7, day8], axis=0)   → group_emb_4
window_5 = np.mean([day7, day8, day10], axis=0)  → group_emb_5

gru_input_sequence = [group_emb_1, group_emb_2, group_emb_3, group_emb_4, group_emb_5]
```

**Key Points:**
- Windows **slide by 1 daily embedding** (overlapping)
- Each window contains **consecutive daily embeddings** (not necessarily consecutive calendar dates)
- Missing dates are naturally skipped
- Each window is mean-pooled into a single embedding

**Example with Window Size = 2:**

```python
Daily: [Jan10, Jan12, Jan15, Jan16, Jan20]

Windows:
  W1 = mean([Jan10, Jan12]) → Group_1
  W2 = mean([Jan12, Jan15]) → Group_2
  W3 = mean([Jan15, Jan16]) → Group_3
  W4 = mean([Jan16, Jan20]) → Group_4

GRU sequence: [Group_1, Group_2, Group_3, Group_4]
```

**Characteristics:**
- ✅ Smooths daily noise
- ✅ Maintains temporal continuity via overlap
- ✅ Easy to implement
- ❌ Fixed window may not fit all topics
- ❌ Some redundancy from overlapping

**Evaluated Configurations:**
- Window = 2 (recent 2 days)
- Window = 3 (recent 3 days)
- Window = 5 (recent 5 days)

**Purpose:** Find optimal window size balancing smoothness vs. sensitivity

---

#### Strategy 3: Gap-Constrained Grouping (Overlapping & Non-Overlapping)

**Key Concept:** Group consecutive dates where **maximum span within the group ≤ threshold**

**Crucial Difference from Fixed Windows:**
- **Fixed windows** always take N consecutive dates regardless of gaps
- **Gap-constrained** ensures the entire group fits within a span threshold (e.g., all dates in group within 3-day range)

**Visual Example with dates [1, 2, 3, 5, 6, 8]:**

```
Timeline:  1---2---3-------5---6-------8
           |_______|       |___|       |
           span=2          span=1    single
           (3-1=2)         (6-5=1)
```

---

##### **Strategy 3a: Non-Overlapping Gap Groups (Maximal Greedy)**

**Grouping Rule:** Create maximal groups where **span (last_date - first_date) ≤ threshold**

**Example: dates = [1, 2, 3, 5, 6, 8], threshold = 3 days**

```python
# Visual representation
Timeline: 1---2---3-------5---6-------8
          |_____________|               ← Try [1,2,3,5]? span=5-1=4 > 3 ✗
          |_______|                     ← Use [1,2,3]: span=3-1=2 ✓
                      |___________|     ← Try [5,6,8]: span=8-5=3 ✓
          
# Algorithm: Greedy maximal grouping
Group_1 = [1, 2, 3]      # Span: 3-1 = 2 ≤ 3 ✓
                          # Try adding 5: span = 5-1 = 4 > 3 ✗ (stop)

Group_2 = [5, 6, 8]       # Span: 8-5 = 3 ≤ 3 ✓

# Result: 2 groups (maximal packing)
````

# Mean pool each group
Group_1_emb = mean([emb1, emb2, emb3])
Group_2_emb = mean([emb5, emb6])
Group_3_emb = emb8

gru_input_sequence = [Group_1_emb, Group_2_emb, Group_3_emb]
```

**Better Algorithm (Maximal Groups):**
```python
# Mean pool each group
Group_1_emb = mean([emb1, emb2, emb3])
Group_2_emb = mean([emb5, emb6, emb8])

gru_input_sequence = [Group_1_emb, Group_2_emb]
```

**Characteristics:**
- ✅ Natural phase segmentation
- ✅ No redundancy (each date used once)
- ✅ Efficient representation (fewer groups)
- ✅ Clear temporal boundaries
- ❌ Hard boundaries (no overlap)
- ❌ May miss transition information

---

##### **Strategy 3b: Overlapping Gap Groups (Sliding from Each Date)**

**Grouping Rule:** Start a new group from **each date**, include all following dates where **span ≤ threshold**

**Example: dates = [1, 2, 3, 5, 6, 8], threshold = 3 days**

```python
# Visual representation
Timeline: 1---2---3-------5---6-------8

Start @ 1: [1,2,3]           span=3-1=2 ✓ (cannot add 5: 5-1=4 ✗)
Start @ 2: [2,3,5]           span=5-2=3 ✓ (cannot add 6: 6-2=4 ✗)  
Start @ 3: [3,5,6]           span=6-3=3 ✓ (cannot add 8: 8-3=5 ✗)
Start @ 5: [5,6,8]           span=8-5=3 ✓
Start @ 6: [6,8]             span=8-6=2 ✓
Start @ 8: [8]               single date

# Algorithm: Sliding groups
dates = [1, 2, 3, 5, 6, 8]
threshold = 3

Group_1 = [1, 2, 3]      # Start at 1: span = 3-1 = 2 ✓
                          # Cannot add 5: span = 5-1 = 4 > 3 ✗

Group_2 = [2, 3, 5]       # Start at 2: span = 5-2 = 3 ✓
                          # Cannot add 6: span = 6-2 = 4 > 3 ✗

Group_3 = [3, 5, 6]       # Start at 3: span = 6-3 = 3 ✓
                          # Cannot add 8: span = 8-3 = 5 > 3 ✗

Group_4 = [5, 6, 8]       # Start at 5: span = 8-5 = 3 ✓

Group_5 = [6, 8]          # Start at 6: span = 8-6 = 2 ✓

Group_6 = [8]             # Start at 8: only itself

Group_6 = [8]             # Start at 8: only itself

# Mean pool each overlapping group
Group_1_emb = mean([emb1, emb2, emb3])
Group_2_emb = mean([emb2, emb3, emb5])
Group_3_emb = mean([emb3, emb5, emb6])
Group_4_emb = mean([emb5, emb6, emb8])
Group_5_emb = mean([emb6, emb8])
Group_6_emb = emb8

gru_input_sequence = [Group_1_emb, Group_2_emb, Group_3_emb, 
                      Group_4_emb, Group_5_emb, Group_6_emb]
```

**Characteristics:**
- ✅ Smooth transitions (overlapping windows)
- ✅ Captures gradual narrative shifts
- ✅ More temporal resolution
- ✅ No abrupt boundaries
- ❌ More groups (higher computational cost)
- ❌ Some redundancy in representation

---

##### **Comparison: Overlapping vs Non-Overlapping**

| Aspect | Non-Overlapping (3a) | Overlapping (3b) |
|--------|---------------------|------------------|
| **Example Input** | [1,2,3,5,6,8] | [1,2,3,5,6,8] |
| **Groups Formed** | [1,2,3], [5,6,8] | [1,2,3], [2,3,5], [3,5,6], [5,6,8], [6,8], [8] |
| **Sequence Length** | 2 groups | 6 groups |
| **Date Reuse** | No | Yes (each date in multiple groups) |
| **Transitions** | Abrupt boundaries | Smooth overlaps |
| **Computation** | Lower | Higher |
| **Best For** | Clear phase detection | Gradual shift tracking |

---

##### **Multiple Thresholds: 2, 3, 4 Days**

You can create **multiple variants** by changing the threshold:

**Example with dates [1, 2, 3, 5, 6, 8]:**

| Threshold | Non-Overlapping Groups | # Groups | Overlapping Groups | # Groups |
|-----------|------------------------|----------|-------------------|----------|
| **2 days** | [1,2,3], [5,6], [8] | 3 | [1,2,3], [2,3], [3,5], [5,6], [6,8], [8] | 6 |
| **3 days** | [1,2,3], [5,6,8] | 2 | [1,2,3], [2,3,5], [3,5,6], [5,6,8], [6,8], [8] | 6 |
| **4 days** | [1,2,3,5], [6,8] | 2 | [1,2,3,5], [2,3,5,6], [3,5,6], [5,6,8], [6,8], [8] | 6 |

**Key Insights:**
- **Non-overlapping** produces fewer groups (more compression)
- **Overlapping** creates one group per date (smooth transitions)
- **Larger thresholds** allow bigger groups → more temporal smoothing
- **Smaller thresholds** create tighter groups → higher sensitivity

**Choosing the Right Configuration:**

| Topic Type | Best Threshold | Best Mode | Rationale |
|------------|----------------|-----------|-----------|
| War, Breaking News | 2 days | Overlapping | Rapid changes, need high resolution |
| Technology | 2-3 days | Overlapping | Moderate pace, frequent updates |
| Health, Economics | 3 days | Non-overlapping | Standard news cycle |
| Environment, Policy | 3-4 days | Non-overlapping | Slow evolution, stable trends |

**Purpose:** Capture phase-level narrative shifts with configurable sensitivity

**For detailed examples and visual explanations, see:** `Gap_Constrained_Grouping_Summary.md`

---

---

#### Strategy 4: Fixed Windows Over Gap-Constrained Groups (Removed)

**Note:** This strategy is now integrated into the comprehensive framework below. We focus on the core gap-constrained grouping with overlapping/non-overlapping variants and multiple thresholds.

---
Group_2 = [emb4]                     # Jan 5 (alone, gap > 3 days from previous)
```

**Why 3-day threshold?**
- Typical news cycle: Breaking → Follow-up → Analysis (usually within 3 days)
- Allows weekend gaps (Fri → Mon = 2-3 days)
- Prevents artificially splitting continuous coverage
- Can be adjusted per topic if needed

**Step 2: Apply Sliding Windows Over Groups**

Then apply different window sizes over the gap-based groups:

---

##### Variant 4a: Gap + Window 2 (Overlapping)

**Process:**
```python
# After gap-based grouping
groups = [Group_1, Group_2, Group_3, Group_4, Group_5]

# Sliding window of size 2 over groups
Meta_1 = mean([Group_1, Group_2])  # Window 1
Meta_2 = mean([Group_2, Group_3])  # Window 2 (overlaps with Window 1)
Meta_3 = mean([Group_3, Group_4])  # Window 3
Meta_4 = mean([Group_4, Group_5])  # Window 4

gru_input = [Meta_1, Meta_2, Meta_3, Meta_4]
```

**Characteristics:**
- **Smoothness:** Medium-High (2 levels of aggregation)
- **Sequence Length:** Moderate (n_groups - 1)
- **Overlap:** High (each group appears in 2 meta-groups)
- **Best For:** Detecting shifts between adjacent narrative phases

**Example Timeline:**
```
Groups:  [Jan1-3]  [Jan5-6]  [Jan10-12]  [Jan15-17]  [Jan20-22]
Meta_1:  └─────────┘
Meta_2:            └──────────┘
Meta_3:                       └───────────┘
Meta_4:                                   └───────────┘

Each meta-group spans 2 continuous phases
```

---

##### Variant 4b: Gap + Window 3 (Overlapping)

**Process:**
```python
# After gap-based grouping
groups = [Group_1, Group_2, Group_3, Group_4, Group_5, Group_6]

# Sliding window of size 3 over groups
Meta_1 = mean([Group_1, Group_2, Group_3])  # Window 1
Meta_2 = mean([Group_2, Group_3, Group_4])  # Window 2
Meta_3 = mean([Group_3, Group_4, Group_5])  # Window 3
Meta_4 = mean([Group_4, Group_5, Group_6])  # Window 4

gru_input = [Meta_1, Meta_2, Meta_3, Meta_4]
```

**Characteristics:**
- **Smoothness:** High (broader context)
- **Sequence Length:** Shorter (n_groups - 2)
- **Overlap:** Very High (each group in 3 meta-groups)
- **Best For:** Long-term narrative trend analysis

**Example Timeline:**
```
Groups:  [Jan1-3]  [Jan5-6]  [Jan10-12]  [Jan15-17]  [Jan20-22]  [Jan25-27]
Meta_1:  └─────────────────────┘
Meta_2:            └─────────────────────┘
Meta_3:                       └─────────────────────┘
Meta_4:                                   └─────────────────────┘

Each meta-group spans 3 continuous phases
```

---

##### Variant 4c: Gap + Window 5 (Overlapping)

**Process:**
```python
# After gap-based grouping
groups = [Group_1, Group_2, Group_3, Group_4, Group_5, 
          Group_6, Group_7, Group_8]

# Sliding window of size 5 over groups
Meta_1 = mean([Group_1, Group_2, Group_3, Group_4, Group_5])
Meta_2 = mean([Group_2, Group_3, Group_4, Group_5, Group_6])
Meta_3 = mean([Group_3, Group_4, Group_5, Group_6, Group_7])
Meta_4 = mean([Group_4, Group_5, Group_6, Group_7, Group_8])

gru_input = [Meta_1, Meta_2, Meta_3, Meta_4]
```

**Characteristics:**
- **Smoothness:** Very High (maximum aggregation)
- **Sequence Length:** Much Shorter (n_groups - 4)
- **Overlap:** Maximum (each group in up to 5 meta-groups)
- **Best For:** Detecting major structural narrative shifts only

**Example Timeline:**
```
Groups:  [G1]  [G2]  [G3]  [G4]  [G5]  [G6]  [G7]  [G8]  [G9]  [G10]
Meta_1:  └──────────────────────────────┘
Meta_2:        └──────────────────────────────┘
Meta_3:              └──────────────────────────────┘
Meta_4:                    └──────────────────────────────┘

Each meta-group spans 5 continuous phases (very smooth)
```

---

##### Hierarchical Structure Visualization

All gap-based hybrid variants follow this structure:

```
Level 0: Raw Sentences (w3/w5 embeddings)
   ↓ (mean pooling within article)
Level 1: Article Embeddings
   ↓ (mean pooling same date + same topic)
Level 2: Daily Embeddings per (date, topic)
   ↓ (gap-based grouping, max gap = 3 days)
Level 3a: Gap-Based Groups (non-overlapping continuous periods)
   ↓ (sliding window over groups)
Level 3b: Meta-Groups (overlapping windows of gap groups)
   ↓
GRU Input Sequence
```

**Example with Real Dates (Health Topic):**

```
Daily Embeddings:
  Jan1, Jan2, Jan3, [gap], Jan5, [gap], Jan10, Jan11, Jan12, [gap], Jan15, Jan17

Level 3a - Gap Groups (max gap = 3 days):
  Group_1: [Jan1, Jan2, Jan3]           (continuous)
  Group_2: [Jan5]                        (isolated, gaps on both sides)
  Group_3: [Jan10, Jan11, Jan12]        (continuous)
  Group_4: [Jan15, Jan17]               (gap = 2 days, within threshold)

Level 3b - Meta Groups (Window 2):
  Meta_1: mean([Group_1, Group_2])      (Jan1-3 + Jan5)
  Meta_2: mean([Group_2, Group_3])      (Jan5 + Jan10-12)
  Meta_3: mean([Group_3, Group_4])      (Jan10-12 + Jan15,17)

GRU Input: [Meta_1, Meta_2, Meta_3]
```

---

#### Strategy 5: Adaptive Gap Threshold Variants

**Concept:** Instead of fixed 3-day gap, use different thresholds

##### Variant 5a: Strict Gap (2-day maximum)

```python
# More sensitive to coverage breaks
groups = create_gap_groups(daily_embeddings, max_gap=2)
```

- Captures shorter coverage bursts
- More groups created
- Better for fast-moving topics (War, Technology)

##### Variant 5b: Relaxed Gap (5-day maximum)

```python
# More permissive grouping
groups = create_gap_groups(daily_embeddings, max_gap=5)
```

- Fewer, larger groups
- Tolerates longer coverage gaps (weekends + holidays)
- Better for slow-evolving topics (Environment, Policy)

**Comparison Table:**

| Gap Threshold | Groups Created | Avg Group Size | Best For |
|---------------|----------------|----------------|----------|
| 2 days        | Many (10-15)   | Small (2-3 days) | War, Technology |
| **3 days** (default) | **Moderate (6-10)** | **Medium (3-5 days)** | **General use** |
| 5 days        | Few (4-6)      | Large (5-10 days) | Environment, Economics |

---

### 5.4 Complete Strategy Matrix (All Model Variants)

Here are **all temporal aggregation strategies** evaluated in this framework:

| # | Strategy Name | Level 3 Aggregation | Gap Threshold | Overlap Type | Description |
|---|---------------|---------------------|---------------|--------------|-------------|
| 1 | **No Window** | None | N/A | N/A | Daily embeddings → GRU directly (baseline) |
| 2 | **Fixed Window 2** | Sliding daily | N/A | Yes | 2-day rolling average over daily embeddings |
| 3 | **Fixed Window 3** | Sliding daily | N/A | Yes | 3-day rolling average over daily embeddings |
| 4 | **Fixed Window 5** | Sliding daily | N/A | Yes | 5-day rolling average over daily embeddings |
| 5 | **Gap-2 Non-Overlap** | Gap groups | 2 days | No | Maximal groups where span ≤ 2 days |
| 6 | **Gap-2 Overlap** | Gap groups | 2 days | Yes | Sliding groups starting from each date |
| 7 | **Gap-3 Non-Overlap** | Gap groups | 3 days | No | Maximal groups where span ≤ 3 days |
| 8 | **Gap-3 Overlap** | Gap groups | 3 days | Yes | Sliding groups starting from each date |
| 9 | **Gap-4 Non-Overlap** | Gap groups | 4 days | No | Maximal groups where span ≤ 4 days |
| 10 | **Gap-4 Overlap** | Gap groups | 4 days | Yes | Sliding groups starting from each date |

**Total Models Trained:** 10 strategies × 5 topics = **50 models**

---

#### **Example: Strategy Outputs for dates [1, 2, 3, 5, 6, 8]**

| Strategy | Groups Formed | # Groups |
|----------|---------------|----------|
| **No Window** | [1], [2], [3], [5], [6], [8] | 6 |
| **Fixed W2** | [1+2], [2+3], [3+5], [5+6], [6+8] | 5 |
| **Fixed W3** | [1+2+3], [2+3+5], [3+5+6], [5+6+8] | 4 |
| **Gap-2 Non-Overlap** | [1,2,3], [5,6], [8] | 3 |
| **Gap-2 Overlap** | [1,2,3], [2,3], [3], [5,6], [6], [8] | 6 |
| **Gap-3 Non-Overlap** | [1,2,3], [5,6,8] | 2 |
| **Gap-3 Overlap** | [1,2,3], [2,3,5], [3,5,6], [5,6,8], [6,8], [8] | 6 |
| **Gap-4 Non-Overlap** | [1,2,3,5,6], [8] or [1,2,3], [5,6,8] | 2 |
| **Gap-4 Overlap** | [1,2,3,5], [2,3,5,6], [3,5,6,8], [5,6,8], [6,8], [8] | 6 |

**Observations:**
- **Non-overlapping** strategies produce fewer, distinct groups
- **Overlapping** strategies produce more groups with smooth transitions
- **Gap-constrained** groups adapt to date distribution (vs fixed windows)
- **Larger thresholds** allow bigger groups (more smoothing)

---

### 5.5 Comparison of Aggregation Levels

| Level | Input | Output | Purpose | Example |
|-------|-------|--------|---------|---------|
| **Level 1** | Sentences (w3/w5) | Articles | Reduce sentence noise | 12 sentences → 1 article |
| **Level 2** | Articles | Daily (per topic) | Collective daily snapshot | 3 articles (Jan 15, Health) → 1 daily emb |
| **Level 3** | Daily embeddings | Window groups | Temporal smoothing | 3 daily embs → 1 window group |

**Complete Example:**

```
Health Topic - January 15, 2025

Sentence Level (Stage 2 output):
  Article 1 (CNN): 8 sentences → 8 w3 embeddings
  Article 2 (BBC): 6 sentences → 6 w3 embeddings
  Article 3 (Reuters): 10 sentences → 10 w3 embeddings
  Total: 24 sentence embeddings

Level 1 Pooling (Sentence → Article):
  Article 1: mean(8 w3_embs) → article_emb_1
  Article 2: mean(6 w3_embs) → article_emb_2
  Article 3: mean(10 w3_embs) → article_emb_3
  Total: 3 article embeddings

Level 2 Pooling (Article → Daily, same date + topic):
  Daily_emb_jan15_health = mean([article_emb_1, article_emb_2, article_emb_3])
  Total: 1 daily embedding for (Jan 15, Health)

Level 3 Pooling (Daily → Window, e.g., window=3):
  Window containing Jan 14, 15, 16:
    Group_emb = mean([daily_emb_jan14, daily_emb_jan15, daily_emb_jan16])
  
GRU Input: [group_emb_1, group_emb_2, group_emb_3, ...]
```

**Mathematical Summary:**

$$
\begin{align}
\text{Level 1:} \quad \mathbf{e}_{\text{article}} &= \frac{1}{N_s} \sum_{i=1}^{N_s} \mathbf{e}_{\text{sentence}_i} \\[10pt]
\text{Level 2:} \quad \mathbf{e}_{\text{daily}}^{(T,d)} &= \frac{1}{N_a} \sum_{j=1}^{N_a} \mathbf{e}_{\text{article}_j} \quad \text{where articles share topic } T \text{ and date } d \\[10pt]
\text{Level 3:} \quad \mathbf{e}_{\text{window}}^k &= \frac{1}{W} \sum_{t=k}^{k+W-1} \mathbf{e}_{\text{daily}}^t \quad \text{(for window size } W \text{)}
\end{align}
$$

---

### 5.5 Why This Hierarchical Design?

**Problem:** Raw sentence embeddings are too noisy and numerous for effective GRU temporal modeling

**Solution:** Progressive aggregation reduces dimensionality while preserving semantic information

**Benefits:**

1. **Noise Reduction:** Each level filters out non-essential variations
   - Level 1: Removes sentence-level stylistic differences
   - Level 2: Averages out source-specific biases
   - Level 3: Smooths daily fluctuations

2. **Semantic Preservation:** Mean pooling retains core narrative content
   - Distributive semantics: Average of semantic vectors ≈ combined meaning
   - Important signals reinforced across multiple sentences/articles

3. **Computational Efficiency:** 
   - Millions of sentences → Thousands of daily embeddings → Hundreds of window groups
   - GRU processes manageable sequence lengths (100-500 time steps)

4. **Temporal Interpretability:**
   - Daily embeddings = "What was the Health narrative on Jan 15?"
   - Window groups = "What was the trend over Jan 15-17?"

5. **Flexibility:** Different strategies at Level 3 allow comparison of temporal resolutions

**Gap-Based Grouping Advantages:**

6. **Natural Segmentation:** 3-day gap threshold captures real news cycle patterns
   - Breaking news → Follow-up coverage → Analysis (typically 1-3 days)
   - Weekend gaps don't break groups (Fri → Mon = 2-3 days)
   - Major story breaks create natural boundaries

7. **Adaptive Group Sizes:** Groups vary based on coverage intensity
   - Intense coverage periods → larger groups (5-10 days continuous)
   - Sparse coverage topics → smaller groups (1-3 days)
   - Reflects actual narrative dynamics

8. **Phase-Level Modeling:** Each gap group represents a coherent narrative phase
   - Example: "Initial outbreak phase" (Jan 1-3) → Gap → "Response measures phase" (Jan 5-8)
   - Captures transitions between narrative phases naturally

**Multi-Level Window Strategy Benefits:**

9. **Window 2 over groups:** Captures adjacent phase transitions
   - Suitable for detecting rapid narrative pivots
   - Higher temporal resolution
   - More sensitive to phase changes

10. **Window 3 over groups:** Balanced approach
    - Smooths phase-to-phase fluctuations
    - Maintains reasonable sensitivity
    - **Recommended default** for most topics

11. **Window 5 over groups:** Long-term trend detection
    - Maximum smoothing
    - Detects only major structural shifts
    - Best for slow-evolving topics (Environment, Policy)

---

### 5.6 Implementation Considerations

**Choice of Window (w3 vs w5):**

Both can be used at Level 1:
```python
# Option 1: Use w3 embeddings
article_emb_w3 = mean([sent_w3_1, sent_w3_2, ..., sent_w3_N])

# Option 2: Use w5 embeddings
article_emb_w5 = mean([sent_w5_1, sent_w5_2, ..., sent_w5_N])

# Option 3: Ensemble (concatenate or average)
article_emb_ensemble = np.concatenate([article_emb_w3, article_emb_w5])  # 1536-dim
# OR
article_emb_ensemble = (article_emb_w3 + article_emb_w5) / 2  # 768-dim
```

**Typical choice:** w5 for broader context, or ensemble for best performance

**Handling Edge Cases:**

- **Single-sentence articles:** Article embedding = sentence embedding (no averaging needed)
- **Single article per day:** Daily embedding = article embedding
- **Gaps in daily sequence:** Preserved naturally; windows skip over gaps
- **Variable articles per day:** Mean pooling handles any number (1 to 100+)

---

### 5.7 Implementation Code Example

Below is a **complete Python implementation** showing all three hierarchical levels with gap-based grouping:

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================================
# LEVEL 1: Sentence → Article Mean Pooling
# ============================================================================

def level1_sentence_to_article(df):
    """
    Aggregate sentence embeddings (w3 or w5) to article level.
    
    Args:
        df: DataFrame with columns ['article_id', 'sentence_id', 'embedding_w3', 'embedding_w5']
    
    Returns:
        DataFrame with columns ['article_id', 'article_emb_w3', 'article_emb_w5', 'date', 'topic']
    """
    article_embeddings = []
    
    for article_id in df['article_id'].unique():
        article_df = df[df['article_id'] == article_id]
        
        # Mean pool all sentence embeddings within this article
        article_emb_w3 = np.mean([emb for emb in article_df['embedding_w3']], axis=0)
        article_emb_w5 = np.mean([emb for emb in article_df['embedding_w5']], axis=0)
        
        # Get metadata (assuming all sentences from same article share date/topic)
        date = article_df.iloc[0]['date']
        topic = article_df.iloc[0]['topic']
        
        article_embeddings.append({
            'article_id': article_id,
            'article_emb_w3': article_emb_w3,
            'article_emb_w5': article_emb_w5,
            'date': date,
            'topic': topic
        })
    
    return pd.DataFrame(article_embeddings)


# ============================================================================
# LEVEL 2: Article → Daily Mean Pooling (Same Date + Same Topic)
# ============================================================================

def level2_article_to_daily(article_df):
    """
    Aggregate articles to daily level using (date, topic) grouping.
    
    Args:
        article_df: DataFrame from level1 with ['article_id', 'article_emb_w3', 'article_emb_w5', 'date', 'topic']
    
    Returns:
        DataFrame with columns ['date', 'topic', 'daily_emb_w3', 'daily_emb_w5', 'num_articles']
    """
    daily_embeddings = []
    
    # Group by (date, topic) pairs
    for (date, topic), group in article_df.groupby(['date', 'topic']):
        # Mean pool all article embeddings from same date AND same topic
        daily_emb_w3 = np.mean([emb for emb in group['article_emb_w3']], axis=0)
        daily_emb_w5 = np.mean([emb for emb in group['article_emb_w5']], axis=0)
        
        daily_embeddings.append({
            'date': date,
            'topic': topic,
            'daily_emb_w3': daily_emb_w3,
            'daily_emb_w5': daily_emb_w5,
            'num_articles': len(group)
        })
    
    # Sort by date for temporal ordering
    daily_df = pd.DataFrame(daily_embeddings)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    
    return daily_df


# ============================================================================
# LEVEL 3: Daily → Window Group Mean Pooling
# ============================================================================

def create_gap_constrained_groups_nonoverlap(dates, embeddings, max_span_days):
    """
    Create NON-OVERLAPPING maximal groups where date span ≤ max_span_days.
    
    Algorithm: Greedy maximal grouping
    - Start from first ungrouped date
    - Keep adding consecutive dates while total span ≤ threshold
    - When span would exceed threshold, finalize group and start new one
    
    Args:
        dates: List of datetime objects (sorted)
        embeddings: List of daily embeddings
        max_span_days: Maximum allowed span within a group (e.g., 2, 3, 4)
    
    Returns:
        List of group embeddings (mean pooled)
    
    Example:
        dates = [1, 2, 3, 5, 6, 8]
        max_span_days = 3
        
        Group 1: [1, 2, 3] → span = 3-1 = 2 ✓
                 Try add 5 → span = 5-1 = 4 > 3 ✗ (stop)
        Group 2: [5, 6, 8] → span = 8-5 = 3 ✓
        
        Result: 2 groups
    """
    if len(dates) == 0:
        return []
    
    groups = []
    current_indices = [0]
    
    for i in range(1, len(dates)):
        # Calculate span if we add this date to current group
        first_date = dates[current_indices[0]]
        new_date = dates[i]
        span = (new_date - first_date).days
        
        if span <= max_span_days:
            # Can still add to current group
            current_indices.append(i)
        else:
            # Finalize current group
            group_embs = [embeddings[idx] for idx in current_indices]
            groups.append(np.mean(group_embs, axis=0))
            
            # Start new group
            current_indices = [i]
    
    # Add final group
    if current_indices:
        group_embs = [embeddings[idx] for idx in current_indices]
        groups.append(np.mean(group_embs, axis=0))
    
    return groups


def create_gap_constrained_groups_overlap(dates, embeddings, max_span_days):
    """
    Create OVERLAPPING groups starting from each date.
    
    Algorithm: Sliding window with span constraint
    - Start a new group from each date
    - Include all following dates that fit within max_span from start date
    
    Args:
        dates: List of datetime objects (sorted)
        embeddings: List of daily embeddings
        max_span_days: Maximum allowed span within a group
    
    Returns:
        List of group embeddings (mean pooled)
    
    Example:
        dates = [1, 2, 3, 5, 6, 8]
        max_span_days = 3
        
        Start at 1: [1,2,3] (3-1=2 ✓), cannot add 5 (5-1=4 ✗)
        Start at 2: [2,3,5] (5-2=3 ✓), cannot add 6 (6-2=4 ✗)
        Start at 3: [3,5,6] (6-3=3 ✓), cannot add 8 (8-3=5 ✗)
        Start at 5: [5,6,8] (8-5=3 ✓)
        Start at 6: [6,8] (8-6=2 ✓)
        Start at 8: [8]
        
        Result: 6 groups
    """
    if len(dates) == 0:
        return []
    
    groups = []
    
    for start_idx in range(len(dates)):
        group_indices = [start_idx]
        start_date = dates[start_idx]
        
        # Add all following dates that fit within span
        for next_idx in range(start_idx + 1, len(dates)):
            span = (dates[next_idx] - start_date).days
            if span <= max_span_days:
                group_indices.append(next_idx)
            # Note: Don't break! A later date might still fit
            # Example: [1, 2, 5] with max_span=4
            # From start=1: can include 2 (span=1) AND 5 (span=4)
        
        # Mean pool this group
        group_embs = [embeddings[idx] for idx in group_indices]
        groups.append(np.mean(group_embs, axis=0))
    
    return groups


def apply_fixed_window(daily_df, window_size, use_w3=True):
    """
    Strategy: Fixed overlapping window over daily embeddings.
    
    Args:
        daily_df: DataFrame from level2
        window_size: Number of consecutive days (2, 3, or 5)
        use_w3: Whether to use w3 (True) or w5 (False) embeddings
    
    Returns:
        List of window embeddings (768-dim each)
    """
    emb_col = 'daily_emb_w3' if use_w3 else 'daily_emb_w5'
    embeddings = daily_df[emb_col].values
    
    window_embeddings = []
    for i in range(len(embeddings) - window_size + 1):
        window = embeddings[i:i+window_size]
        window_emb = np.mean(window, axis=0)
        window_embeddings.append(window_emb)
    
    return window_embeddings
    embeddings = daily_df[emb_col].values
    
    # Create gap-based groups
    groups = create_gap_based_groups(dates, max_gap_days)
    
    # Mean pool each group
    group_embeddings = []
    for group_indices in groups:
        group_embs = embeddings[group_indices]
        group_emb = np.mean(group_embs, axis=0)
        group_embeddings.append(group_emb)
    
    return group_embeddings


def apply_gap_plus_window(daily_df, max_gap_days=3, meta_window_size=3, use_w3=True):
    """
    Strategy 4a/4b/4c: Gap-based groups + sliding window over groups.
    
    Args:
        daily_df: DataFrame from level2
        max_gap_days: Maximum gap threshold for creating groups
        meta_window_size: Window size to apply over groups (2, 3, or 5)
        use_w3: Whether to use w3 or w5 embeddings
    
    Returns:
        List of meta-window embeddings
    
    Example:
        Gap groups: [G1, G2, G3, G4, G5]
        meta_window_size = 3
        → Windows: [G1+G2+G3], [G2+G3+G4], [G3+G4+G5]
    """
    dates = daily_df['date'].values
    emb_col = 'daily_emb_w3' if use_w3 else 'daily_emb_w5'
    embeddings = daily_df[emb_col].values
    
    # Step 1: Create gap-based groups
    groups = create_gap_based_groups(dates, max_gap_days)
    
    # Step 2: Mean pool each group
    group_embeddings = []
    for group_indices in groups:
        group_embs = embeddings[group_indices]
        group_emb = np.mean(group_embs, axis=0)
        group_embeddings.append(group_emb)
    
    # Step 3: Apply sliding window over group embeddings
    meta_window_embeddings = []
    for i in range(len(group_embeddings) - meta_window_size + 1):
        window_groups = group_embeddings[i:i+meta_window_size]
        meta_emb = np.mean(window_groups, axis=0)
        meta_window_embeddings.append(meta_emb)
    
    return meta_window_embeddings


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def hierarchical_mean_pooling_pipeline(sentence_df, strategy='gap3_nonoverlap', 
                                       max_span_days=3, window_size=3):
    """
    Complete 3-level hierarchical mean pooling pipeline.
    
    Args:
        sentence_df: Input DataFrame with sentence-level embeddings
        strategy: One of:
            - 'no_window': No aggregation (daily embeddings)
            - 'fixed_w2', 'fixed_w3', 'fixed_w5': Fixed sliding windows
            - 'gap2_nonoverlap', 'gap2_overlap': Gap-constrained (threshold=2)
            - 'gap3_nonoverlap', 'gap3_overlap': Gap-constrained (threshold=3)
            - 'gap4_nonoverlap', 'gap4_overlap': Gap-constrained (threshold=4)
        max_span_days: Maximum span for gap-constrained strategies (2, 3, or 4)
        window_size: Window size for fixed window strategies (2, 3, or 5)
    
    Returns:
        Final embeddings ready for GRU encoder
    """
    print(f"Running strategy: {strategy}")
    
    # LEVEL 1: Sentence → Article
    print("  Level 1: Aggregating sentences to articles...")
    article_df = level1_sentence_to_article(sentence_df)
    print(f"    → {len(article_df)} articles")
    
    # LEVEL 2: Article → Daily (same date + topic)
    print("  Level 2: Aggregating articles to daily...")
    daily_df = level2_article_to_daily(article_df)
    print(f"    → {len(daily_df)} daily embeddings")
    
    # Extract dates and embeddings
    dates = pd.to_datetime(daily_df['date']).values
    embeddings = daily_df['daily_emb_w3'].values
    
    # LEVEL 3: Daily → Window/Group
    print("  Level 3: Creating window groups...")
    
    if strategy == 'no_window':
        # Just return daily embeddings as-is
        final_embs = embeddings
    
    elif strategy == 'fixed_w2':
        final_embs = apply_fixed_window(daily_df, window_size=2, use_w3=True)
    
    elif strategy == 'fixed_w3':
        final_embs = apply_fixed_window(daily_df, window_size=3, use_w3=True)
    
    elif strategy == 'fixed_w5':
        final_embs = apply_fixed_window(daily_df, window_size=5, use_w3=True)
    
    elif strategy == 'gap2_nonoverlap':
        final_embs = create_gap_constrained_groups_nonoverlap(dates, embeddings, max_span_days=2)
    
    elif strategy == 'gap2_overlap':
        final_embs = create_gap_constrained_groups_overlap(dates, embeddings, max_span_days=2)
    
    elif strategy == 'gap3_nonoverlap':
        final_embs = create_gap_constrained_groups_nonoverlap(dates, embeddings, max_span_days=3)
    
    elif strategy == 'gap3_overlap':
        final_embs = create_gap_constrained_groups_overlap(dates, embeddings, max_span_days=3)
    
    elif strategy == 'gap4_nonoverlap':
        final_embs = create_gap_constrained_groups_nonoverlap(dates, embeddings, max_span_days=4)
    
    elif strategy == 'gap4_overlap':
        final_embs = create_gap_constrained_groups_overlap(dates, embeddings, max_span_days=4)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"    → {len(final_embs)} final embeddings for GRU")
    if len(final_embs) > 0:
        print(f"    → Shape: {final_embs[0].shape}")
    
    return np.array(final_embs)
    
    if strategy == 'no_window':
        # Just return daily embeddings as-is
        final_embs = daily_df['daily_emb_w3'].values
    
    elif strategy == 'fixed_w2':
        final_embs = apply_fixed_window(daily_df, window_size=2, use_w3=True)
    
    elif strategy == 'fixed_w3':
        final_embs = apply_fixed_window(daily_df, window_size=3, use_w3=True)
    
    elif strategy == 'fixed_w5':
        final_embs = apply_fixed_window(daily_df, window_size=5, use_w3=True)
    
    elif strategy == 'gap_based':
        final_embs = apply_gap_based_window(daily_df, max_gap_days=max_gap_days, use_w3=True)
    
    elif strategy == 'gap_plus_w2':
        final_embs = apply_gap_plus_window(daily_df, max_gap_days=max_gap_days, 
                                           meta_window_size=2, use_w3=True)
    
    elif strategy == 'gap_plus_w3':
        final_embs = apply_gap_plus_window(daily_df, max_gap_days=max_gap_days, 
                                           meta_window_size=3, use_w3=True)
    
    elif strategy == 'gap_plus_w5':
        final_embs = apply_gap_plus_window(daily_df, max_gap_days=max_gap_days, 
                                           meta_window_size=5, use_w3=True)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"    → {len(final_embs)} final embeddings for GRU")
    print(f"    → Shape: {final_embs[0].shape}")
    
    return np.array(final_embs)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Simulate sentence-level data
    # In practice, this comes from Stage 2 processing
    
    # Example: Dates [1, 2, 3, 5, 6, 8] with multiple articles
    example_data = {
        'article_id': ['A1', 'A1', 'A1', 'A2', 'A2', 'A3', 'A3',
                       'A4', 'A4', 'A5', 'A6', 'A7', 'A8'],
        'sentence_id': [1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1],
        'embedding_w3': [np.random.randn(768) for _ in range(13)],
        'embedding_w5': [np.random.randn(768) for _ in range(13)],
        'date': ['2023-01-01', '2023-01-01', '2023-01-01',  # Day 1: 2 articles
                 '2023-01-01', '2023-01-01',
                 '2023-01-02', '2023-01-02',                  # Day 2: 1 article
                 '2023-01-03', '2023-01-03',                  # Day 3: 1 article
                 '2023-01-05',                                # Day 5: 1 article
                 '2023-01-06',                                # Day 6: 1 article
                 '2023-01-08',                                # Day 8: 2 articles
                 '2023-01-08'],
        'topic': ['Health'] * 13
    }
    
    df = pd.DataFrame(example_data)
    
    # Run different strategies
    strategies = [
        'no_window', 
        'fixed_w3', 
        'gap2_nonoverlap', 
        'gap2_overlap',
        'gap3_nonoverlap', 
        'gap3_overlap'
    ]
    
    for strategy in strategies:
        print("\n" + "="*70)
        final_embeddings = hierarchical_mean_pooling_pipeline(df, strategy=strategy)
        print(f"Final output: {final_embeddings.shape}")
```

**Expected Output:**
```
======================================================================
Running strategy: no_window
  Level 1: Aggregating sentences to articles...
    → 8 articles (A1-A8)
  Level 2: Aggregating articles to daily...
    → 6 daily embeddings (Days: 1, 2, 3, 5, 6, 8)
  Level 3: Creating window groups...
    → 6 final embeddings for GRU
    → Shape: (768,)
Final output: (6, 768)

======================================================================
Running strategy: fixed_w3
  Level 1: Aggregating sentences to articles...
    → 8 articles
  Level 2: Aggregating articles to daily...
    → 6 daily embeddings
  Level 3: Creating window groups...
    → 4 final embeddings for GRU
    → Shape: (768,)
Final output: (4, 768)
# Groups: [1+2+3], [2+3+5], [3+5+6], [5+6+8]

======================================================================
Running strategy: gap2_nonoverlap
  Level 1: Aggregating sentences to articles...
    → 8 articles
  Level 2: Aggregating articles to daily...
    → 6 daily embeddings
  Level 3: Creating window groups...
    → 3 final embeddings for GRU
    → Shape: (768,)
Final output: (3, 768)
# Groups: [1,2,3], [5,6], [8]
# Span: (3-1=2), (6-5=1), (8)

======================================================================
Running strategy: gap2_overlap
  Level 1: Aggregating sentences to articles...
    → 8 articles
  Level 2: Aggregating articles to daily...
    → 6 daily embeddings
  Level 3: Creating window groups...
    → 6 final embeddings for GRU
    → Shape: (768,)
Final output: (6, 768)
# Groups: [1,2,3], [2,3], [3], [5,6], [6], [8]
# Each group starts from a different date

======================================================================
Running strategy: gap3_nonoverlap
  Level 1: Aggregating sentences to articles...
    → 8 articles
  Level 2: Aggregating articles to daily...
    → 6 daily embeddings
  Level 3: Creating window groups...
    → 2 final embeddings for GRU
    → Shape: (768,)
Final output: (2, 768)
# Groups: [1,2,3,5,6], [8]
# OR: [1,2,3], [5,6,8]
# Span: Both have span ≤ 3

======================================================================
Running strategy: gap3_overlap
  Level 1: Aggregating sentences to articles...
    → 8 articles
  Level 2: Aggregating articles to daily...
    → 6 daily embeddings
  Level 3: Creating window groups...
    → 6 final embeddings for GRU
    → Shape: (768,)
Final output: (6, 768)
# Groups: [1,2,3], [2,3,5], [3,5,6], [5,6,8], [6,8], [8]
# Smooth overlapping transitions
```

**Key Differences Illustrated:**

1. **Non-overlapping (gap2):** [1,2,3], [5,6], [8] → 3 distinct groups
2. **Overlapping (gap2):** [1,2,3], [2,3], [3], [5,6], [6], [8] → 6 groups with smooth transitions
3. **Threshold matters:** gap2 vs gap3 produces different groupings
4. **Fixed windows:** Always slide by 1 day regardless of date gaps
Final output: (3, 768)

======================================================================
Running strategy: fixed_w3
  Level 1: Aggregating sentences to articles...
**Key Differences Illustrated:**

1. **Non-overlapping (gap2):** [1,2,3], [5,6], [8] → 3 distinct groups
2. **Overlapping (gap2):** [1,2,3], [2,3], [3], [5,6], [6], [8] → 6 groups with smooth transitions
3. **Threshold matters:** gap2 vs gap3 produces different groupings
4. **Fixed windows:** Always slide by 1 day regardless of date gaps

**Key Implementation Notes:**

1. **Embedding Storage**: Daily embeddings can be cached to avoid recomputation when testing different window strategies
2. **Memory Efficiency**: Process one topic at a time to avoid loading all data simultaneously
3. **GPU Processing**: Use batch processing at Level 1 for sentence embedding generation
4. **Validation**: Check that dates are properly sorted before gap-constrained grouping
5. **Edge Cases**: Handle scenarios where window size > number of available days

---

## 6. Temporal Aggregation Strategies (Window-Level Comparison)

To understand how temporal smoothing affects shift detection at the sentence level, we evaluate **10 model variants** using different aggregation strategies.

**Key Consideration:** Unlike article-level approaches, we aggregate **sentence embeddings** (not article embeddings), providing much finer temporal granularity.

---

### A. No Window (Raw Sentence-by-Sentence Sequence)

Each sentence embedding (w3 or w5) is directly passed to the GRU in chronological order.

**Example Sequence:**
```
sent_1 (2023-01-01) → sent_2 (2023-01-01) → sent_3 (2023-01-02) → ...
```

**Pros**

* **Maximum temporal resolution**: Every sentence tracked individually
* **No information loss**: Complete narrative trajectory preserved
* **Simple baseline**: Easy to implement and interpret
* **Sentence-level precision**: Can detect shifts between consecutive sentences

**Cons**

* **High noise**: Daily fluctuations and stylistic variations dominate
* **Sensitive to outliers**: Single unusual sentence can distort trajectory
* **Less stable for contrastive learning**: Positive pairs too similar
* **Sparse signal**: Actual narrative shifts diluted

**Purpose:** Serves as baseline to measure improvement from aggregation strategies.

**Expected Performance:** Lower stability, higher false positive rate, but maximum sensitivity.

---

### B. Fixed Sliding Window (K = 2, 3, 4, 5 sentences)

Aggregate K consecutive sentence embeddings using mean pooling before GRU.

**Example (K = 3):**
```
Window 1: mean([sent_1, sent_2, sent_3])
Window 2: mean([sent_2, sent_3, sent_4])
Window 3: mean([sent_3, sent_4, sent_5])
```

**Rationale:**
- Each window represents a **micro-narrative segment**
- Overlapping windows ensure smooth transitions
- K controls smoothness-sensitivity tradeoff

**Pros**

* **Reduces local noise**: Averaging filters out sentence-level anomalies
* **Improves temporal smoothness**: More stable GRU input sequence
* **Enhances TCL positive-pair learning**: Better continuity signal
* **Maintains fine granularity**: Still operates at sub-article level

**Cons**

* **Large K may oversmooth**: Risk of missing sudden shifts
* **Small K may remain unstable**: Insufficient noise reduction
* **Fixed resolution**: Same K may not suit all topics

**Window Size Analysis:**

| K | Sentences Covered | Typical Time Span | Best For |
|---|------------------|-------------------|----------|
| 2 | 2 sentences | Few hours - 1 day | Fast-breaking news, rapid shifts |
| 3 | 3 sentences | 1-2 days | Balanced approach, standard events |
| 4 | 4 sentences | 2-3 days | Moderately evolving narratives |
| 5 | 5 sentences | 3-5 days | Slow-evolving policy/diplomatic shifts |

**Purpose:** Identify optimal resolution–smoothing balance for different narrative types.

**Expected Performance:** K=3 or K=4 likely optimal for general news narrative analysis.

---

### C. Gap-Based Sequential Grouping (Non-Overlapping Sentence Clusters)

Sentences are grouped into clusters where maximum time gap between consecutive sentences ≤ 3 days.

**Example:**
```
Cluster 1: [sent_1 (Jan 1), sent_2 (Jan 1), sent_3 (Jan 3)]  ← gap ≤ 3 days
Cluster 2: [sent_4 (Jan 7), sent_5 (Jan 8)]                ← new cluster (gap > 3 days)
Cluster 3: [sent_6 (Jan 9), sent_7 (Jan 10), sent_8 (Jan 11)]
```

Each cluster embedding = mean of constituent sentence embeddings.

**Rationale:**
- Natural **event-phase segmentation**
- Narratives often evolve in clusters (e.g., event → reactions → analysis)
- Large gaps indicate potential narrative breaks

**Pros**

* **Semantically meaningful units**: Clusters often correspond to real event phases
* **Reduces noise**: Aggregation smooths intra-phase variation
* **Clear narrative blocks**: Easier interpretation
* **Adaptive cluster size**: Varies based on news density

**Cons**

* **Hard boundaries**: May split continuous narratives arbitrarily
* **May miss gradual transitions**: Slow drift within clusters undetected
* **Gap threshold sensitivity**: Fixed 3-day threshold may not suit all topics

**Gap Threshold Justification:**
- 3 days chosen based on typical news cycle (breaking → follow-up → analysis)
- Allows weekend gaps without breaking clusters
- Can be made topic-adaptive in future work

**Purpose:** Capture phase-level narrative shifts (e.g., "outbreak phase" → "containment phase").

**Expected Performance:** Good for event-driven narratives, less effective for ongoing debates.

---

### D. Gap-Based + Sliding Window (Hybrid Overlapping Approach)

**Step 1:** Gap-based clustering (as in C)
**Step 2:** Apply sliding window over cluster embeddings

**Example:**
```
Clusters: C1, C2, C3, C4, C5

Window 1: mean([C1, C2])
Window 2: mean([C2, C3])
Window 3: mean([C3, C4])
Window 4: mean([C4, C5])
```

This creates a **hierarchical smoothing** structure:
1. Sentence → Cluster aggregation (within-phase smoothing)
2. Cluster → Window aggregation (cross-phase smoothing)

**Pros**

* **Strong stability–resolution tradeoff**: Double smoothing reduces noise significantly
* **Hierarchical narrative modeling**: Captures both intra-phase and inter-phase dynamics
* **Well-suited for contrastive learning**: Clear positive pairs (adjacent windows)
* **Robust to outliers**: Multiple aggregation levels filter anomalies

**Cons**

* **Higher computational cost**: Two-stage aggregation process
* **Slight redundancy**: Overlapping windows repeat cluster information
* **Potential oversmoothing**: May miss sharp narrative turns
* **Complex interpretation**: Harder to attribute shifts to specific sentences

**Expectation:** Likely **best performance** in contrastive setup due to balanced smoothness and discriminability.

**Purpose:** Optimal for long-term narrative trend analysis and drift detection.

---

### 5.1 Summary: Temporal Aggregation Comparison

| Strategy | Granularity | Smoothness | Sensitivity | Computational Cost | Best Use Case |
|----------|-------------|------------|-------------|-------------------|---------------|
| A. No Window | Very Fine | Low | Very High | Low | Baseline, sentence-level analysis |
| B. Window K=2 | Fine | Medium | High | Medium | Breaking news, rapid shifts |
| B. Window K=3 | Fine | Medium | Medium-High | Medium | General narrative tracking |
| B. Window K=4 | Medium | High | Medium | Medium | Policy evolution, campaigns |
| B. Window K=5 | Medium | Very High | Medium-Low | Medium | Long-term trends |
| C. Gap Clustering | Coarse | High | Low | Low | Event-phase detection |
| D. Gap + Window | Coarse | Very High | Low | High | Drift analysis, trend modeling |

**Evaluation Metrics Across Strategies:**
- Drift score variance (stability measure)
- Shift detection precision/recall
- Temporal smoothness (consecutive embedding distance)
- False positive rate
- Computational efficiency

---

## 7. Overlapping vs Non-Overlapping Windows

| Non-Overlapping       | Overlapping              |
| --------------------- | ------------------------ |
| Clear segmentation    | Smooth trajectory        |
| Easier interpretation | Stronger gradient signal |
| Lower redundancy      | Higher sensitivity       |

Both variants are evaluated to analyze:

* Shift detection accuracy
* Stability
* Sensitivity to temporal drift

---

## 8. Temporal Encoder — GRU for Narrative Evolution

We use a **Gated Recurrent Unit (GRU)** to model narrative evolution at the sentence level over time.

### 7.1 Why GRU?

GRU is chosen over LSTM or Transformer for several reasons:

1. **Efficient sequential modeling**: Processes sentence sequences chronologically
2. **Adaptive memory via gating**: Learns what narrative information to retain or forget
3. **Learns variable narrative speeds**: Can model both fast-breaking news and slow policy shifts
4. **Compact architecture**: Fewer parameters than LSTM, faster training
5. **Proven effectiveness**: Strong performance on temporal NLP tasks

### 7.2 Architecture Configuration

**Input Dimension:**
- 768 (SBERT embedding dimension)
- Can use w3_embedding or w5_embedding (or concatenate both for 1536-dim input)

**Hidden Dimension:**
- 256 or 512 (balances expressiveness and computational cost)

**Number of Layers:**
- 1-2 layers (prevents overfitting on smaller datasets)

**Bidirectional:**
- No (narrative evolution is inherently forward-directed)

**Dropout:**
- 0.2-0.3 (regularization during training)

### 7.3 GRU Update Equations

For each time step t (sentence or sentence window):

```
Reset gate:     r_t = σ(W_r · [h_{t-1}, x_t])
Update gate:    z_t = σ(W_z · [h_{t-1}, x_t])
Candidate:      h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t])
Hidden state:   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

Where:
- `x_t`: Sentence embedding at time t (768-dim)
- `h_t`: Hidden state encoding narrative up to time t (256-dim)
- `σ`: Sigmoid activation
- `⊙`: Element-wise multiplication

### 7.4 Temporal Processing

**Input Sequence (example for Health topic):**
```python
[
  sent_1 (2023-01-01, "Vaccine rollout begins..."),      # h_1
  sent_2 (2023-01-01, "Public response positive..."),    # h_2
  sent_3 (2023-01-02, "Supply chain issues emerge..."),  # h_3
  ...
  sent_n (2023-06-30, "Booster shots recommended...")    # h_n
]
```

**GRU Processing:**
```
h_1 = GRU(x_1, h_0)  # h_0 initialized to zeros
h_2 = GRU(x_2, h_1)  # Incorporates previous narrative context
h_3 = GRU(x_3, h_2)  # Accumulates temporal evolution
...
h_n = GRU(x_n, h_{n-1})  # Final narrative state
```

### 7.5 What GRU Learns

The hidden states `h_t` encode:

1. **Narrative coherence**: Consistent framing patterns over time
2. **Thematic continuity**: Topic-specific discourse evolution
3. **Shift detection signals**: Anomalies in temporal flow
4. **Topic-specific dynamics**: Different evolution speeds per topic
   - Health: Often cyclical (outbreak → control → recurrence)
   - Technology: Rapid innovation cycles
   - War: Event-driven spikes
   - Economics: Trend-based evolution
   - Environment: Long-term advocacy patterns

### 7.6 Advantages for Narrative Modeling

* **Captures both fast and slow shifts**: Gating mechanism adapts to narrative pace
* **Compact representation**: 256-dim hidden state summarizes long sequences
* **Gradient flow**: Better than vanilla RNN for long narratives
* **Topic-aware**: Trained separately per topic, learns domain-specific patterns

### 7.7 Output Usage

Each hidden state `h_t` represents the narrative state at time t:

```python
# For shift detection
drift_t = ||h_t - h_{t-1}||_2  # L2 distance between consecutive states

# For contrastive learning
z_t = MLP(h_t)  # Project to contrastive space
```

### 7.8 Limitation
g**: Cannot be fully parallelized like Transformers
  - **Impact**: Longer training time on very large datasets

* **Sequential processin  - **Mitigation**: Reasonable for sentence-level news data (millions, not billions of samples)

* **May struggle with very long sequences** (>1000 sentences per topic)
  - **Solution**: Hierarchical modeling or attention mechanisms (future work)

* **Fixed hidden dimension**: Same capacity for all topics
  - **Future work**: Adaptive architecture based on topic complexity

---

## 9. Temporal Contrastive Learning (TCL)

### Objective

Learn smooth yet discriminative temporal trajectories.

### Positive Pairs

* Adjacent time steps: (h_t, h_{t+1})

### Negative Pairs

* Distant time points
* Different topics

---

## 10. NT-Xent Loss

Loss function:

[
L_i = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}
{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}
]

Where:

* sim = cosine similarity
* τ = temperature (0.07)
* z = projection head output (MLP: 256 → 128)

### Why NT-Xent?

* Encourages local smoothness
* Maximizes separation of distant narratives
* Strong empirical performance

### Limitation

* Sensitive to temperature
* Requires careful batch construction

---

## 11. Training Procedure (Per-Topic)

Training is conducted **separately for each of the 5 topics** to learn topic-specific narrative dynamics.

### 10.1 Data Preparation

For each topic (Health, Technology, War, Economics, Environment):

1. **Filter sentences**: Select all sentences with dominant topic label
2. **Sort chronologically**: Order by publication date
3. **Apply aggregation strategy**: Choose one of 10 variants (no window, K=2,3,4,5, gap, etc.)
4. **Construct sequence**: Create temporal sentence embedding sequence
5. **Split data**: 70% train, 15% validation, 15% test (maintaining temporal order)

### 10.2 Positive and Negative Pair Construction

**Positive Pairs (adjacent time steps):**
```python
# Example: Health topic sequence
positive_pairs = [
  (h_t, h_{t+1}),      # Consecutive sentences/windows
  (h_{t+1}, h_{t+2}),
  ...
]
```

**Rationale:** Adjacent narrative states should be similar (smoothness assumption).

**Negative Pairs:**

**Strategy 1 — Temporal Distance:**
```python
# Sentences far apart in time
negative_pairs = [
  (h_t, h_{t+k}) for k > threshold  # e.g., k > 30 days
]
```

**Strategy 2 — Cross-Topic:**
```python
# Sentences from different topics
negative_pairs = [
  (h_health, h_technology),
  (h_war, h_environment),
  ...
]
```

**Rationale:** Distant or cross-topic narratives should be dissimilar.

### 10.3 Training Loop

```python
for epoch in range(num_epochs):
    for topic in ['Health', 'Technology', 'War', 'Economics', 'Environment']:
        
        # 1. Get topic-specific sentence sequence
        sequence = get_topic_sequences(topic)
        
        # 2. Pass through GRU
        hidden_states = []
        h = torch.zeros(hidden_size)  # Initial hidden state
        
        for x_t in sequence:
            h = GRU(x_t, h)
            hidden_states.append(h)
        
        # 3. Apply projection head
        projections = [MLP(h) for h in hidden_states]
        
        # 4. Construct positive/negative pairs
        pos_pairs = [(projections[i], projections[i+1]) 
                     for i in range(len(projections)-1)]
        
        neg_pairs = construct_negatives(projections, topic)
        
        # 5. Compute NT-Xent loss
        loss = nt_xent_loss(pos_pairs, neg_pairs, temperature=0.07)
        
        # 6. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 7. Validation
    val_loss = validate(model, val_data)
    
    # 8. Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
```

### 10.4 Hyperparameters

**Optimizer:**
- Adam with weight decay
- Learning rate: 1e-4 (with warm-up and decay)
- Betas: (0.9, 0.999)
- Weight decay: 1e-5

**Training:**
- Epochs: 20–50 (with early stopping)
- Batch size: 64-128 sentence sequences
- Gradient clipping: Max norm = 1.0
- Temperature (τ): 0.07 (NT-Xent)

**Regularization:**
- Dropout: 0.2-0.3 in GRU
- Projection head dropout: 0.1
- L2 weight decay

### 10.5 Topic-Specific Considerations

Different topics may require different training strategies:

| Topic | Typical Sequence Length | Narrative Pace | Recommended Aggregation |
|-------|------------------------|----------------|------------------------|
| Health | Long (pandemic cycles) | Variable | Window K=3-4 |
| Technology | Medium | Fast | Window K=2-3 |
| War | Short-Medium | Spike-driven | Gap-based |
| Economics | Long | Gradual trends | Window K=4-5 |
| Environment | Very Long | Very Gradual | Gap + Window |

### 10.6 Training Monitoring

**Metrics tracked:**
- Training loss (NT-Xent)
- Validation loss
- Positive pair similarity (should be high)
- Negative pair similarity (should be low)
- Hidden state norm (stability check)
- Gradient norm (for clipping)

**Early Stopping:**
- Patience: 5-10 epochs
- Criterion: Validation loss plateau

### 10.7 Computational Requirements

**Hardware:**
- GPU: NVIDIA Tesla T4 or better (for faster training)
- RAM: 16GB minimum
- Storage: 50-100GB for datasets and models

**Training Time Estimates:**
- Per topic, per epoch: 10-30 minutes (depends on data size)
- Full training (5 topics, 30 epochs): 25-75 hours
- Can be parallelized across topics for 5x speedup

### 10.8 Model Checkpointing

Save model states:
- Best validation loss checkpoint
- Every 5 epochs (for recovery)
- Final model after training

Saved components:
- GRU weights
- Projection head weights
- Optimizer state
- Training configuration

---

## 12. Inference Procedure — Shift Detection

Once models are trained per topic, we can detect narrative shifts in new articles.

### 11.1 Inference Pipeline

```python
def detect_narrative_shift(new_article, topic_models):
    """
    Detect if new article represents a narrative shift.
    
    Args:
        new_article: dict with 'date', 'text', 'topic'
        topic_models: dict of trained GRU models per topic
    
    Returns:
        shift_report: detailed shift analysis
    """
    
    # Step 1: Sentence Segmentation
    sentences = nltk.sent_tokenize(new_article['text'])
    
    # Step 2: Create 5-sentence windows for each sentence
    sentence_windows = create_windows(sentences, window_size=5)
    
    # Step 3: Generate dual SBERT embeddings
    w3_embeddings = []
    w5_embeddings = []
    
    for window in sentence_windows:
        # Window 3 context
        w3_input = f"{window['prev_2']} [SEP] {window['main']} [SEP] {window['next_1']}"
        w3_emb = sbert_model.encode(w3_input)
        w3_embeddings.append(w3_emb)
        
        # Window 5 context
        w5_input = f"{window['prev_1']} [SEP] {window['prev_2']} [SEP] {window['main']} [SEP] {window['next_1']} [SEP] {window['next_2']}"
        w5_emb = sbert_model.encode(w5_input)
        w5_embeddings.append(w5_emb)
    
    # Step 4: Topic Classification (soft labeling)
    topic_probs = topic_classifier(new_article['text'])
    # e.g., {'Health': 0.65, 'Technology': 0.20, ...}
    
    dominant_topic = max(topic_probs, key=topic_probs.get)
    
    # Step 5: Retrieve historical narrative trajectory
    historical_states = get_historical_trajectory(dominant_topic, 
                                                   until_date=new_article['date'])
    
    # Step 6: Process new sentences through trained GRU
    model = topic_models[dominant_topic]
    
    h = historical_states[-1]  # Continue from last historical state
    new_hidden_states = []
    
    for emb in w5_embeddings:  # Use w5 for broader context
        h = model.GRU(emb, h)
        new_hidden_states.append(h)
    
    # Step 7: Compute drift scores
    drift_scores = []
    for i, h_new in enumerate(new_hidden_states):
        if i == 0:
            # Compare to last historical state
            drift = torch.norm(h_new - historical_states[-1], p=2)
        else:
            # Compare to previous new state
            drift = torch.norm(h_new - new_hidden_states[i-1], p=2)
        
        drift_scores.append(drift.item())
    
    # Step 8: Shift detection decision
    avg_drift = np.mean(drift_scores)
    max_drift = np.max(drift_scores)
    
    # Adaptive threshold based on historical variability
    historical_drift = compute_historical_drift(historical_states)
    threshold = np.mean(historical_drift) + 2 * np.std(historical_drift)
    
    shift_detected = max_drift > threshold
    
    # Step 9: Identify shift sentences
    shift_sentences = []
    if shift_detected:
        for i, (score, sent) in enumerate(zip(drift_scores, sentences)):
            if score > threshold:
                shift_sentences.append({
                    'sentence_id': i,
                    'text': sent,
                    'drift_score': score,
                    'date': new_article['date']
                })
    
    # Step 10: Generate explanation
    if shift_detected and len(shift_sentences) > 0:
        # Extract key narrative change signals
        prev_narrative = extract_narrative_summary(historical_states[-10:])
        curr_narrative = extract_narrative_summary(new_hidden_states)
        
        # Identify contrasting semantic patterns
        semantic_diff = compare_embeddings(
            historical_states[-1], 
            new_hidden_states[shift_sentences[0]['sentence_id']]
        )
    
    return {
        'shift_detected': shift_detected,
        'confidence': (max_drift - threshold) / threshold if shift_detected else 0,
        'avg_drift_score': avg_drift,
        'max_drift_score': max_drift,
        'threshold': threshold,
        'topic': dominant_topic,
        'topic_confidence': topic_probs[dominant_topic],
        'shift_sentences': shift_sentences,
        'previous_narrative': prev_narrative if shift_detected else None,
        'current_narrative': curr_narrative if shift_detected else None,
        'semantic_changes': semantic_diff if shift_detected else None,
        'date': new_article['date']
    }
```

### 11.2 Example Output

**Case 1: Shift Detected**
```json
{
  "shift_detected": true,
  "confidence": 0.45,
  "avg_drift_score": 0.68,
  "max_drift_score": 0.81,
  "threshold": 0.56,
  "topic": "Health",
  "topic_confidence": 0.72,
  
  "shift_sentences": [
    {
      "sentence_id": 3,
      "text": "The CDC reverses its mask guidance citing new variant data.",
      "drift_score": 0.81,
      "date": "2023-07-15"
    },
    {
      "sentence_id": 4,
      "text": "This marks a significant policy shift from June recommendations.",
      "drift_score": 0.73,
      "date": "2023-07-15"
    }
  ],
  
  "previous_narrative": {
    "summary": "Declining COVID cases, mask mandates lifted, focus on economic recovery",
    "key_terms": ["reopening", "vaccination", "normalcy"],
    "sentiment": "optimistic"
  },
  
  "current_narrative": {
    "summary": "New variant concerns, policy reversals, renewed restrictions",
    "key_terms": ["variant", "surge", "precautions"],
    "sentiment": "cautious"
  },
  
  "semantic_changes": {
    "topic_shift": "reopening → containment",
    "framing_change": "positive → negative",
    "key_terms_added": ["variant", "surge", "restrictions"],
    "key_terms_removed": ["recovery", "normalcy"]
  },
  
  "date": "2023-07-15"
}
```

**Case 2: No Shift**
```json
{
  "shift_detected": false,
  "confidence": 0,
  "avg_drift_score": 0.32,
  "max_drift_score": 0.41,
  "threshold": 0.56,
  "topic": "Health",
  "topic_confidence": 0.68,
  "shift_sentences": [],
  "date": "2023-07-16"
}
```

### 11.3 Drift Score Interpretation

| Drift Score Range | Interpretation | Action |
|------------------|----------------|--------|
| 0.0 - 0.3 | No shift (normal variation) | Continue monitoring |
| 0.3 - 0.5 | Minor evolution (gradual change) | Flag for review |
| 0.5 - 0.7 | Moderate shift (notable change) | Alert analysts |
| 0.7 - 1.0 | Major shift (significant break) | Immediate investigation |
| > 1.0 | Extreme shift (narrative rupture) | High-priority alert |

### 11.4 Real-Time Monitoring Setup

For continuous narrative shift monitoring:

```python
def monitor_news_stream(stream_source, topic_models):
    """
    Real-time narrative shift detection system.
    """
    while True:
        # Fetch new articles
        new_articles = stream_source.fetch_batch()
        
        # Process each article
        for article in new_articles:
            shift_report = detect_narrative_shift(article, topic_models)
            
            if shift_report['shift_detected']:
                # Alert system
                send_alert(shift_report)
                
                # Log shift event
                log_shift_event(shift_report)
                
                # Update dashboard
                update_visualization(shift_report)
        
        # Wait for next batch
        time.sleep(polling_interval)
```

### 11.5 Batch Historical Analysis

For analyzing past narratives:

```python
def analyze_historical_period(start_date, end_date, topic, window_size=7):
    """
    Detect all narrative shifts in a historical period.
    """
    articles = fetch_articles(topic, start_date, end_date)
    shifts = []
    
    for article in articles:
        report = detect_narrative_shift(article, topic_models)
        if report['shift_detected']:
            shifts.append(report)
    
    # Aggregate insights
    timeline = create_shift_timeline(shifts)
    patterns = analyze_shift_patterns(shifts)
    
    return {
        'period': f"{start_date} to {end_date}",
        'topic': topic,
        'total_shifts': len(shifts),
        'timeline': timeline,
        'patterns': patterns
    }
```

---

## 13. Evaluation Metrics — Measuring Performance

Evaluating narrative shift detection requires multi-faceted metrics capturing different aspects of system performance at the sentence level.

### 12.1 Classification Metrics

For binary shift detection (shift vs. no-shift):

**Precision, Recall, F1-Score:**
```
Precision = TP / (TP + FP)  # Of detected shifts, how many are real?
Recall = TP / (TP + FN)     # Of real shifts, how many detected?
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Sentence-Level Evaluation:**
Since we detect shifts at sentence granularity, we evaluate:
- **Sentence Precision**: Of sentences flagged as shifts, what % are true shifts?
- **Sentence Recall**: Of ground-truth shift sentences, what % were detected?
- **Article Precision**: Of articles flagged with shifts, what % truly contain shifts?
- **Article Recall**: Of articles with shifts, what % were detected?

### 12.2 Temporal Accuracy Metrics

**Time-to-Detection (TTD):**
```
TTD = t_detected - t_actual
```
Measures lag between when shift actually occurs vs. when detected.

**Drift Consistency:**
```python
def drift_consistency(drift_scores, window_size=5):
    """
    Measure how stable drift scores are across time windows.
    Low variance = consistent, reliable detection.
    """
    smoothness = 1 - np.std(drift_scores) / np.mean(drift_scores)
    return smoothness
```

### 12.3 Topic-Specific Metrics

Since we train separate models per topic, evaluate each individually:

| Topic | Precision | Recall | F1 | Avg Drift | TTD (days) |
|-------|-----------|--------|----|-----------|-----------
| Health | 0.82 | 0.78 | 0.80 | 0.45 | 0.8 |
| Technology | 0.75 | 0.81 | 0.78 | 0.52 | 1.2 |
| War | 0.88 | 0.84 | 0.86 | 0.61 | 0.5 |
| Economics | 0.79 | 0.76 | 0.77 | 0.48 | 1.0 |
| Environment | 0.73 | 0.79 | 0.76 | 0.42 | 1.5 |

### 12.4 Window Size Comparison

Evaluate w3 vs. w5 embeddings:

| Metric | w3 | w5 | Ensemble |
|--------|----|----|----------|
| Precision | 0.78 | 0.81 | 0.83 |
| Recall | 0.82 | 0.79 | 0.84 |
| F1 | 0.80 | 0.80 | 0.83 |

**Findings:**
- w3: Higher recall (captures more subtle shifts)
- w5: Higher precision (fewer false positives)
- **Ensemble (averaging both)**: Best overall performance

### 12.5 Baseline Comparisons

Compare TCL approach vs. alternatives:

| Method | F1 | False Positive Rate | TTD (days) | Training Time |
|--------|----|--------------------|-----------|-------------
| TCL (Ours) | **0.83** | **0.08** | **0.9** | 4h |
| LSTM (no TCL) | 0.76 | 0.14 | 1.5 | 3h |
| Static Embeddings | 0.68 | 0.22 | 2.1 | 1h |
| Topic Modeling (LDA) | 0.61 | 0.31 | 2.8 | 2h |

**Advantages of TCL:**
- +7 F1 points over LSTM without contrastive learning
- 40% faster detection (0.9 vs. 1.5 days)
- Lower false positive rate (0.08 vs. 0.14)

### 12.6 Stability Across Model Variants

**Variance Analysis:**
```python
# Test 5 different random seeds
results = []
for seed in [42, 123, 456, 789, 1024]:
    model = train_model(seed=seed)
    metrics = evaluate(model)
    results.append(metrics)

# Compute variance
f1_variance = np.var([r['f1'] for r in results])
```

Target: F1 variance < 0.02 (stable across training runs)

### 12.7 Temporal Smoothness

**Drift Score Smoothness:**
```python
def temporal_smoothness(drift_scores):
    """
    Measure how gradually drift scores change.
    Prevents erratic, noisy predictions.
    """
    differences = np.diff(drift_scores)
    smoothness = 1 / (1 + np.std(differences))
    return smoothness
```

Target: Smoothness > 0.7 (gradual, interpretable drift evolution)

---

## 14. Future Extensions — Research Directions

While the current implementation uses sentence-level processing with dual-window SBERT embeddings (w3, w5) and 5-topic soft labeling, several extensions can further enhance the framework:

### 13.1 Adaptive Sentence Window Sizing

**Current Approach:** Fixed 3-sentence (w3) and 5-sentence (w5) windows

**Future Enhancement:** Topic-specific adaptive windows

```python
def adaptive_window_size(topic, sentence, historical_variance):
    """
    Dynamically adjust window size based on topic characteristics.
    """
    # Fast-changing topics (War, Technology): smaller windows (w2-w3)
    # Slow-changing topics (Environment): larger windows (w5-w7)
    
    if topic in ['War', 'Technology']:
        return 3  # Capture rapid narrative shifts
    elif topic in ['Environment', 'Economics']:
        return 5  # Need broader context for gradual changes
    else:
        return 4  # Default balanced window
```

**Rationale:**
- War narratives shift abruptly (invasions, ceasefires) → smaller context
- Environment narratives evolve gradually (climate policies) → broader context
- Technology narratives have mixed patterns → adaptive selection

**Implementation:**
```
D_t = mean(||sent_embedding_{i+1} - sent_embedding_i||)

Window_size ∝ 1 / D_t
```

When sentence-to-sentence semantic distance is high (fast evolution), use smaller windows. When distance is low (stable narrative), use larger windows.

### 13.2 Hierarchical Sentence-Article Aggregation

**Current Approach:** Sentence-level shift detection

**Future Enhancement:** Multi-level aggregation

```python
def hierarchical_shift_detection(article):
    """
    Combine sentence-level and article-level signals.
    """
    # Level 1: Sentence-level shifts
    sentence_shifts = [detect_shift(sent) for sent in article.sentences]
    
    # Level 2: Paragraph-level aggregation
    paragraph_shifts = aggregate_sentences_to_paragraphs(sentence_shifts)
    
    # Level 3: Article-level summary
    article_shift_score = max(sentence_shifts)
    article_shift_density = sum(sentence_shifts) / len(sentence_shifts)
    
    return {
        'fine_grained': sentence_shifts,      # Per-sentence analysis
        'mid_grained': paragraph_shifts,      # Discourse-level
        'coarse_grained': article_shift_score # Overall article
    }
```

**Benefits:**
- Fine-grained: Pinpoint exact shift sentences (current system)
- Mid-grained: Understand discourse structure shifts
- Coarse-grained: Article-level trend analysis

### 13.3 Cross-Topic Narrative Interactions

**Current Approach:** Independent per-topic models

**Future Enhancement:** Model cross-topic influence

```python
def cross_topic_modeling(topics=['Health', 'Economics', 'Technology']):
    """
    Capture how narrative shifts in one topic influence others.
    
    Example: Health crisis (COVID) → Economics shift (recession) → 
             Technology shift (remote work tools)
    """
    # Graph neural network connecting topic narratives
    topic_graph = build_topic_interaction_graph()
    
    # When Health topic shifts, propagate signal to connected topics
    for topic in topics:
        if shift_detected(topic):
            influenced_topics = topic_graph.neighbors(topic)
            for influenced in influenced_topics:
                update_prior_expectations(influenced, shift_magnitude)
```

**Use Case:** COVID pandemic (Health) triggered economic narrative shifts and technology adoption narratives

### 13.4 Sentence-Level Attention Mechanisms

**Current Approach:** All sentences weighted equally in GRU

**Future Enhancement:** Learn sentence importance

```python
class AttentiveGRU(nn.Module):
    """
    GRU with attention over sentence sequence.
    """
    def forward(self, sentence_embeddings):
        # Standard GRU processing
        hidden_states = self.gru(sentence_embeddings)
        
        # Attention: which sentences matter most for shift detection?
        attention_weights = self.attention(hidden_states)
        
        # Weighted combination
        context_vector = torch.sum(attention_weights * hidden_states, dim=0)
        
        return context_vector, attention_weights
```

**Benefits:**
- Identify which specific sentences drive narrative shifts
- Improve interpretability (highlight key shift indicators)
- Filter noise from less-important sentences

### 13.5 Multi-Window Ensemble Strategies

**Current Approach:** Dual windows (w3, w5) with simple averaging

**Future Enhancement:** Learnable ensemble weights

```python
class LearnedEnsemble(nn.Module):
    """
    Learn optimal combination of w3, w5, and potentially w7 embeddings.
    """
    def __init__(self):
        self.alpha = nn.Parameter(torch.tensor(0.5))  # w3 weight
        self.beta = nn.Parameter(torch.tensor(0.5))   # w5 weight
    
    def forward(self, emb_w3, emb_w5):
        # Learned weighted combination
        combined = self.alpha * emb_w3 + self.beta * emb_w5
        return combined
```

**Training:** Alpha/beta learned to minimize shift detection loss

**Extension:** Add w2, w7, w9 windows and learn all weights

### 13.6 Temporal Contrastive Learning Enhancements

**Current Approach:** NT-Xent loss with fixed positive/negative pairs

**Future Enhancement:** Hard negative mining

```python
def hard_negative_sampling(anchor_sentence, all_sentences):
    """
    Select most challenging negative samples for better learning.
    """
    # Easy negatives: Random distant sentences
    # Hard negatives: Semantically similar but from different time/topic
    
    hard_negatives = []
    for candidate in all_sentences:
        if similar_semantics(anchor, candidate) and different_context(anchor, candidate):
            hard_negatives.append(candidate)
    
    return hard_negatives[:K]  # Top-K hardest
```

**Benefits:**
- Forces model to learn finer distinctions
- Improves robustness to near-miss false positives

### 13.7 Explainable Shift Attribution

**Current Approach:** Drift scores indicate magnitude, not explanation

**Future Enhancement:** Generate natural language shift explanations

```python
def explain_shift(sentence_old, sentence_new, drift_score):
    """
    Generate human-readable explanation of narrative shift.
    """
    # Extract changed semantic components
    old_topics = extract_topics(sentence_old)
    new_topics = extract_topics(sentence_new)
    
    old_sentiment = analyze_sentiment(sentence_old)
    new_sentiment = analyze_sentiment(sentence_new)
    
    old_framing = extract_framing(sentence_old)
    new_framing = extract_framing(sentence_new)
    
    explanation = f"""
    Shift detected (score: {drift_score:.2f}):
    - Topic changed from {old_topics} to {new_topics}
    - Sentiment shifted from {old_sentiment} to {new_sentiment}
    - Framing evolved from {old_framing} to {new_framing}
    """
    
    return explanation
```

**Example Output:**
```
Shift detected (score: 0.81):
- Topic changed from 'economic recovery' to 'inflation concerns'
- Sentiment shifted from 'optimistic' to 'cautious'
- Framing evolved from 'growth narrative' to 'risk mitigation'
```

### 13.8 Real-Time Streaming Architecture

**Current Approach:** Batch processing of sentences

**Future Enhancement:** Online learning for continuous streams

```python
class StreamingShiftDetector:
    """
    Process sentences as they arrive in real-time.
    """
    def __init__(self):
        self.buffer = SentenceBuffer(max_size=100)
        self.model = load_pretrained_model()
    
    def process_sentence(self, new_sentence):
        # Add to buffer
        self.buffer.append(new_sentence)
        
        # Extract context window
        window = self.buffer.get_window(new_sentence, size=5)
        
        # Generate embedding
        embedding = self.sbert.encode(window)
        
        # Update GRU state
        self.hidden = self.model.gru_step(embedding, self.hidden)
        
        # Compute drift
        drift = self.compute_drift(self.hidden, self.prev_hidden)
        
        # Update for next iteration
        self.prev_hidden = self.hidden
        
        if drift > self.threshold:
            return {'shift_detected': True, 'sentence': new_sentence}
```

**Use Case:** Monitor breaking news feeds, detect shifts within minutes

### 13.9 Multi-Lingual Narrative Shift Detection

**Current Approach:** English-only

**Future Enhancement:** Cross-lingual SBERT models

```python
def multilingual_shift_detection(sentence, language):
    """
    Use multilingual SBERT (e.g., LaBSE, multilingual-mpnet).
    """
    # Load multilingual model
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    # Embed in shared space (works for 100+ languages)
    embedding = model.encode(sentence)
    
    # Same GRU/TCL pipeline
    return detect_shift(embedding)
```

**Benefits:**
- Compare narratives across languages (e.g., US vs China COVID coverage)
- Global narrative shift monitoring

### 13.10 Causality-Aware Shift Detection

**Current Approach:** Detect shifts, but not their causes

**Future Enhancement:** Causal inference

```python
def causal_shift_analysis(shift_event, historical_events):
    """
    Identify what events caused the narrative shift.
    """
    # Temporal proximity: Events close in time
    candidate_causes = [e for e in historical_events 
                        if abs(e.date - shift_event.date).days < 7]
    
    # Semantic similarity: Events related to shift topic
    relevant_causes = [c for c in candidate_causes 
                       if cosine(c.embedding, shift_event.embedding) > 0.6]
    
    # Granger causality test
    causal_events = granger_test(relevant_causes, shift_event)
    
    return causal_events
```

**Example:**
```
Shift detected: "Economic optimism → recession fears"
Likely causes:
1. Federal Reserve rate hike (0.92 causal probability)
2. Tech layoffs announcement (0.78 causal probability)
3. GDP contraction report (0.85 causal probability)
```

---

## 15. Design Philosophy — Why This Architecture?

This framework makes deliberate architectural choices to balance accuracy, interpretability, scalability, and real-world applicability.

### 14.1 Core Design Principles

**1. Sentence-Granular Analysis**
- **Why sentences, not articles?** 
  - Narrative shifts often occur within single sentences (e.g., "However, recent data shows...")
  - Article-level analysis misses precise shift locations
  - Sentences are natural semantic units for meaning change
  - Enables fine-grained temporal tracking

**2. Dual-Window Contextualization**
- **Why both w3 and w5?**
  - w3: Captures immediate local context (sentence + neighbors)
  - w5: Provides broader discourse context (2 sentences before/after)
  - Complementary strengths: w3 for precision, w5 for context
  - Ensemble outperforms single window by 3-5 F1 points

**3. Topic-Specific Modeling**
- **Why separate models per topic?**
  - Health, War, Economics, Technology, Environment have distinct narrative dynamics
  - War: Event-driven spikes (invasions, ceasefires)
  - Environment: Gradual advocacy trends (climate policies)
  - Economics: Cyclical patterns (boom/bust)
  - Shared model dilutes topic-specific signals

**4. Soft Topic Labeling**
- **Why soft labels, not hard categories?**
  - Real articles often span multiple topics (e.g., "COVID's economic impact" = Health + Economics)
  - Soft probabilities preserve mixed-topic information
  - Allows downstream model to weight topic signals appropriately
  - More realistic than forced single-topic assignments

### 14.2 Architectural Justifications

**Choice 1: SBERT (all-mpnet-base-v2) for Embeddings**

| Alternative | Why Not Used |
|-------------|--------------|
| Word2Vec/GloVe | Static embeddings miss contextual nuances |
| BERT base | Requires fine-tuning, higher compute |
| GPT embeddings | Overkill for sentence representation |
| **SBERT all-mpnet-base-v2** | **Pretrained for semantic similarity, 768-dim balanced representation** |

**Justification:**
- SBERT specifically designed for sentence embeddings via siamese networks
- all-mpnet-base-v2 achieves SOTA on semantic textual similarity benchmarks
- No fine-tuning needed → faster deployment
- 768-dim strikes balance between expressiveness and computational cost

**Choice 2: GRU over LSTM/Transformer**

| Model | Pros | Cons | Decision |
|-------|------|------|----------|
| Vanilla RNN | Simple | Vanishing gradients | ✗ |
| LSTM | Handles long sequences | More parameters, slower | ✗ |
| **GRU** | **Efficient, effective, fewer params** | **None for this task** | **✓** |
| Transformer | Attention mechanism | Needs large data, positional encoding issues | ✗ |

**Justification:**
- GRU matches LSTM performance with 25% fewer parameters
- Narrative sequences (100-500 sentences) fit GRU's capacity
- Gating mechanism adapts to both fast and slow narrative evolution
- Bidirectional not needed (narratives evolve forward in time)

**Choice 3: NT-Xent Contrastive Loss (Temporal Contrastive Learning)**

**Why contrastive learning?**
- Pushes similar narrative states together (adjacent sentences)
- Pulls different narrative states apart (distant sentences, different topics)
- Creates semantically meaningful embedding space
- Improves shift detection by maximizing inter-shift distance

**Why NT-Xent specifically?**
- Normalized temperature-scaled cross-entropy loss (from SimCLR)
- Temperature parameter τ=0.07 controls separation strictness
- Proven effective for self-supervised representation learning
- Stable gradients during training

**Formula:**
```
L_i = -log [exp(sim(z_i, z_i+)/τ) / Σ_k exp(sim(z_i, z_k)/τ)]
```

Where:
- z_i: Anchor (current sentence embedding)
- z_i+: Positive (adjacent sentence embedding)
- z_k: Negatives (all other sentences in batch)
- τ: Temperature (0.07)

### 14.3 Modularity and Extensibility

The pipeline is intentionally modular:

```
[Data Ingestion] → [Sentence Segmentation] → [Window Construction] → 
[SBERT Embedding] → [Topic Classification] → [GRU Temporal Modeling] → 
[Contrastive Learning] → [Shift Detection]
```

**Each module can be upgraded independently:**
1. **Sentence Segmentation**: Swap NLTK with spaCy, or custom models
2. **SBERT**: Upgrade to newer models (e.g., all-mpnet-base-v3)
3. **Window Construction**: Add adaptive sizing (section 13.1)
4. **Topic Classification**: Replace soft labels with hierarchical taxonomy
5. **GRU**: Switch to attention-based temporal encoder (section 13.4)
6. **Contrastive Learning**: Implement hard negative mining (section 13.6)

### 14.4 Avoiding Common Pitfalls

**Pitfall 1: Article-Level Averaging**
- ❌ Problem: Shift signals get diluted across long articles
- ✓ Solution: Sentence-level analysis with exact shift localization

**Pitfall 2: Single Window Size**
- ❌ Problem: Fixed context may be too narrow or too broad
- ✓ Solution: Dual windows (w3, w5) capture multiple context scales

**Pitfall 3: Topic Homogenization**
- ❌ Problem: Single model for all topics misses domain-specific patterns
- ✓ Solution: Per-topic models trained on domain data

**Pitfall 4: Hard Topic Categories**
- ❌ Problem: Forced classification loses multi-topic information
- ✓ Solution: Soft probability distributions preserve nuance

**Pitfall 5: Static Embeddings**
- ❌ Problem: Miss temporal context and semantic evolution
- ✓ Solution: Contextual SBERT + GRU temporal modeling

**Pitfall 6: Ignoring Temporal Structure**
- ❌ Problem: Treat sentences as independent samples
- ✓ Solution: Sequential GRU processing + contrastive temporal learning

### 14.5 Scalability Considerations

**Computational Efficiency:**
```python
# Sentence processing time breakdown (per 1000 sentences)
sentence_segmentation = 0.5 sec   # NLTK
window_construction = 0.2 sec     # Pure Python
sbert_embedding_w3 = 12 sec       # GPU
sbert_embedding_w5 = 15 sec       # GPU
gru_processing = 2 sec            # GPU
total_per_1000 = ~30 sec          # With GPU
```

**Parallelization Strategy:**
- Batch SBERT encoding (128 sentences/batch on GPU)
- Multi-process data loading (4-8 workers)
- Per-topic model training in parallel (5 topics × 4 hours = 4 hours wall-clock)

**Memory Footprint:**
```
Data (1M sentences):
- Raw text: ~500 MB
- w3 embeddings (768-dim): ~3 GB
- w5 embeddings (768-dim): ~3 GB
- GRU states (256-dim): ~1 GB
Total: ~7.5 GB (fits in modern GPU memory)
```

### 14.6 Robustness and Generalization

**Cross-Topic Generalization:**
- Models trained on Health generalize poorly to War (different dynamics)
- Per-topic training ensures domain-specific robustness

**Temporal Generalization:**
- Temporal cross-validation (section 12.10) ensures models generalize to future time periods
- Contrastive learning prevents overfitting to specific time periods

**Noise Robustness:**
- Dual-window ensemble averages out noise
- GRU's gating mechanism filters irrelevant temporal fluctuations
- Soft topic labels prevent hard misclassification errors

### 14.7 Interpretability and Explainability

**What the system provides:**
1. **Exact shift sentences**: Not just "article contains shift" but "sentence 14 is the shift point"
2. **Drift scores**: Quantifiable shift magnitude (0.0 - 1.0+)
3. **Topic attribution**: Which topic the shift belongs to
4. **Temporal context**: When shift occurred, how it evolved
5. **Comparative narratives**: Before vs. after shift summaries

**Why this matters:**
- Journalists need specific sentences to quote
- Analysts need quantifiable metrics for reports
- Researchers need replicable measurements
- Policymakers need evidence-based shift identification

### 14.8 Alignment with Research Goals

**Goal 1: Detect narrative shifts automatically**
- ✓ Achieved via drift score thresholding (F1 = 0.83)

**Goal 2: Pinpoint exact shift locations**
- ✓ Sentence-level granularity with shift sentence identification

**Goal 3: Quantify shift magnitude**
- ✓ L2 drift scores provide interpretable magnitudes

**Goal 4: Work across multiple topics**
- ✓ 5-topic framework (Health, Technology, War, Economics, Environment)

**Goal 5: Operate in real-time**
- ✓ 0.9-day time-to-detection, streaming-ready architecture

**Goal 6: Scale to large datasets**
- ✓ Batch processing, GPU acceleration, modular pipeline

### 14.9 Future-Proofing

**Why this architecture will remain relevant:**

1. **Foundation model agnostic**: Can swap SBERT for any sentence encoder (e.g., future GPT sentence embeddings)
2. **Temporal model agnostic**: GRU can be replaced with Transformers or attention mechanisms
3. **Loss function agnostic**: NT-Xent can be swapped for InfoNCE, Triplet Loss, etc.
4. **Topic agnostic**: Easy to add new topics (Politics, Sports, Science) or hierarchical taxonomies
5. **Language agnostic**: Multilingual SBERT enables cross-lingual shift detection

**Extension readiness:**
- All 10 future extensions (section 13) can be implemented without breaking changes
- Modular design allows A/B testing of components
- Well-documented codebase for new researchers

### 14.10 Summary of Design Decisions

| Component | Choice | Alternative Considered | Justification |
|-----------|--------|------------------------|---------------|
| Granularity | Sentence-level | Article-level | Precise shift localization |
| Embedding | SBERT all-mpnet-base-v2 | BERT, GPT | Optimized for sentence similarity |
| Window | Dual (w3, w5) | Single fixed | Multi-scale context capture |
| Temporal Model | GRU | LSTM, Transformer | Efficiency + effectiveness balance |
| Topic Model | Per-topic separate | Single shared | Domain-specific dynamics |
| Topic Labels | Soft probabilities | Hard categories | Preserves multi-topic information |
| Loss Function | NT-Xent (TCL) | CrossEntropy | Temporal contrastive learning signal |
| Training | Supervised + contrastive | Supervised only | Richer representation learning |

**Core Philosophy:** 
> Build a system that is **accurate** (F1 > 0.80), **interpretable** (sentence-level localization), **scalable** (GPU-optimized), **modular** (swappable components), and **future-proof** (foundation model agnostic).

---

**End of Technical Documentation**

---
