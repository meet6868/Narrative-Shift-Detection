# Approach 1: Baseline Day-Level Windowing

## Overview

Approach 1 serves as the **baseline** for narrative shift detection using standard temporal windowing techniques. It implements a straightforward day-level aggregation strategy with fixed 3-day windows.

**Pipeline File:** `TCL_Pipeline_1.ipynb`  
**Status:** ✅ Baseline Implementation  
**Complexity:** Low  

---

## Methodology

### 1. Window Construction
- **Window Size:** 3 consecutive days
- **Aggregation:** All articles within each day are embedded and averaged
- **Two Variants:**
  - **With Overlap:** Windows shift by 1 day (Day 1-3, Day 2-4, Day 3-5...)
  - **No Overlap:** Windows shift by 3 days (Day 1-3, Day 4-6, Day 7-9...)

### 2. Embedding Strategy
Each day's representation is created by:
1. Collecting all article embeddings for that day
2. Computing the mean embedding: 
   $$\mathbf{e}_{day} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{a}_i$$
   where $\mathbf{a}_i$ are article embeddings

3. Window embedding is the concatenation or average of 3 day embeddings:
   $$\mathbf{w}_t = [\mathbf{e}_{d_1}, \mathbf{e}_{d_2}, \mathbf{e}_{d_3}]$$

### 3. Drift Detection
- **Metric:** Cosine distance between consecutive windows
- **Threshold:** μ + 2σ (mean + 2 standard deviations)
- **Shift Criterion:** $\text{distance}(w_t, w_{t+1}) > \mu + 2\sigma$

### 4. Temporal Contrastive Learning
- **Positive Pairs:** Consecutive windows $(w_t, w_{t+1})$
- **Negative Pairs:** Non-consecutive windows from same topic
- **Loss Function:** InfoNCE loss with temperature scaling

---

## Output Folders

### Variant 1: With Overlap
**Location:** `../Model_output/Pip_1_W5_Overlap/`

**Contents:**
```
Pip_1_W5_Overlap/
├── Climate/
│   ├── drift_scores.png          # Drift timeline
│   ├── similarity_matrix.png     # Window similarity heatmap
│   └── shift_events.json         # Detected shift timestamps
├── Economics/
├── Health/
├── Technology/
├── War/
└── summary_metrics.json          # Overall statistics
```

**Example Visualization:**
![Drift Scores Example](../Model_output/Pip_1_W5_Overlap/Climate/drift_scores.png)

### Variant 2: No Overlap
**Location:** `../Model_output/Pip_1_W5_NOOverlap/`

Same structure as Variant 1, but with non-overlapping windows.

---

## Key Features

### ✅ Strengths
1. **Simplicity:** Easy to understand and implement
2. **Interpretability:** Clear temporal boundaries (3-day windows)
3. **Low Computational Cost:** Fast processing, minimal memory
4. **Reproducibility:** Deterministic windowing strategy
5. **Baseline Comparison:** Standard reference for evaluating other approaches

### ⚠️ Drawbacks
1. **Fixed Window Size:**
   - Cannot adapt to variable-length narrative shifts
   - Some shifts may span less than 3 days (missed)
   - Others may span more than 3 days (fragmented)

2. **Overlap Redundancy:**
   - Overlapping windows create correlated drift scores
   - Can double-count the same shift event
   - Increases false positive rate

3. **Sparse Data Handling:**
   - Days with few articles have weak embeddings
   - Creates imbalanced windows (e.g., Day 1: 50 articles, Day 2: 2 articles)
   - Noise in low-data days affects entire window

4. **Temporal Granularity:**
   - Day-level aggregation loses intra-day shift information
   - Cannot detect shifts happening within a single day

5. **No Context Modeling:**
   - No historical context beyond 3-day window
   - No long-term trend awareness

---

## Implementation Details

### Configuration Parameters
```python
WINDOW_SIZE = 3              # Days per window
OVERLAP = True               # Enable/disable overlap
STRIDE = 1 if OVERLAP else 3 # Window shift amount
SIMILARITY_THRESHOLD = 2.0   # σ multiplier for shift detection
```

### Data Requirements
**Input File:** `../Processed_Data/ALL_Combined_Data.csv`

**Required Columns:**
- `embedding` - Article embedding (768-dim array)
- `Topic` - Topic label (Climate, Economics, Health, Technology, War)
- `Date` - Publication date (YYYY-MM-DD format)

**Example Row:**
```csv
embedding,Topic,Date
"[-0.023, 0.451, ..., 0.128]",Climate,2023-01-15
```

### Stages in Notebook

| Stage | Description | Output |
|-------|-------------|--------|
| 1 | Data Loading | Parsed embeddings, date conversion |
| 2 | Window Construction | List of 3-day windows per topic |
| 3 | Embedding Aggregation | Window-level embeddings |
| 4 | TCL Training | Contrastive model weights |
| 5 | Drift Calculation | Cosine distances between windows |
| 6 | Shift Detection | Thresholded drift events |
| 7 | Visualization | Plots and heatmaps |
| 8 | Export Results | JSON and PNG files |

---

## Performance Metrics

### Computational Efficiency
- **Processing Time:** ~5 minutes (5 topics, ~10K articles)
- **Memory Usage:** ~2GB RAM
- **GPU Required:** No (CPU sufficient)

### Detection Statistics (Average across topics)
- **Windows Created:** 150-200 per topic
- **Shifts Detected:** 12-18 per topic
- **False Positive Rate:** ~15% (estimated via manual review)
- **Temporal Coverage:** 100% (all days included)

### Example Output (Climate Topic)
```json
{
  "topic": "Climate",
  "total_windows": 187,
  "shifts_detected": 16,
  "avg_drift_score": 0.42,
  "max_drift_score": 0.89,
  "shift_dates": [
    "2023-02-14",
    "2023-03-21",
    "2023-05-08",
    ...
  ]
}
```

---

## Comparison with Other Approaches

| Aspect | Approach 1 | Approach 2 | Approach 4 |
|--------|-----------|-----------|-----------|
| **Window Type** | Fixed 3-day | Variable groups | Article-based |
| **Granularity** | Day-level | Group-level | Sentence-level |
| **Sparse Data** | ⚠️ Poor | ✅ Good | ✅ Excellent |
| **Interpretability** | ✅ High | Moderate | ⚠️ Low |
| **Processing Time** | ✅ Fast (5 min) | Medium (8 min) | Slow (20 min) |
| **Shift Precision** | Moderate | Good | ✅ Excellent |

---

## Usage Example

### Running the Pipeline
```bash
# Navigate to TCL folder
cd /home/hp/SEM2/INLP/Naretve_Shift/TCL/

# Launch Jupyter
jupyter notebook TCL_Pipeline_1.ipynb

# Run all cells or specific stages:
# - Stage 1-2: Data loading and windowing
# - Stage 3-4: Embedding and training
# - Stage 5-8: Drift detection and visualization
```

### Customizing Parameters
```python
# In Stage 2: Window Construction
def create_windows(df, window_size=3, overlap=True):
    stride = 1 if overlap else window_size
    # ... rest of function
```

### Analyzing Results
```python
# Load shift events
import json
with open('../Model_output/Pip_1_W5_Overlap/Climate/shift_events.json') as f:
    shifts = json.load(f)

# Print shift timeline
for shift in shifts['shift_dates']:
    print(f"Shift detected on: {shift}")
```

---

## Visualizations

### Drift Score Timeline
**Description:** Line plot showing cosine distance between consecutive windows over time.

**Interpretation:**
- **Orange Line:** μ + 2σ threshold
- **Red Markers:** Detected narrative shifts
- **Peaks:** High narrative divergence
- **Valleys:** Narrative stability

### Similarity Matrix Heatmap
**Description:** Matrix showing pairwise similarity between all windows.

**Interpretation:**
- **Diagonal:** Self-similarity (always 1.0)
- **Off-diagonal:** Cross-window similarity
- **Dark Blocks:** Coherent narrative periods
- **Light Regions:** Shift boundaries

---

## Known Limitations

1. **Edge Cases:**
   - First and last windows may be incomplete (< 3 days)
   - Holiday periods with no articles create gaps

2. **Topic-Specific Issues:**
   - High-volume topics (Economics) have more stable embeddings
   - Low-volume topics (War) exhibit higher drift noise

3. **Temporal Assumptions:**
   - Assumes narratives shift at day boundaries
   - Cannot model intra-day or sub-3-day shifts

---

## Future Improvements

While this approach has been succeeded by more advanced methods (Approaches 2, 4, 5), potential enhancements could include:

1. **Adaptive Windowing:** Dynamically adjust window size based on article density
2. **Weighted Aggregation:** Weight articles by importance/relevance
3. **Smoothing:** Apply moving average to reduce noise
4. **Multi-Scale Analysis:** Combine 1-day, 3-day, and 7-day windows

However, these improvements are better realized in **Approach 2** (flexible grouping) and **Approach 4** (article-level detection).

---

## References

- Main Documentation: [TCL Complete Flow](TCL_Complete_Flow.md)
- Comparison Report: [TCL vs Baselines](TCL_vs_Baselines_Narrative_Shift_Comparison.pdf)
- Next Approach: [Approach 2 - Group-Based Segmentation](approach_2.md)

---

**Note:** This approach is maintained as the baseline reference. For production deployments, consider **Approach 5** (optimized) or **Approach 2** (better sparse data handling).
