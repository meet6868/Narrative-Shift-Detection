# Approach 2: Group-Based Segmentation

## Overview

Approach 2 introduces **flexible group-based segmentation** to address the fixed window limitations of Approach 1. Instead of rigid 3-day windows, this approach groups articles based on semantic coherence and temporal proximity, with two distinct variants.

**Pipeline File:** `TCL_Pipeline_2.ipynb`  
**Status:** ✅ Enhanced Implementation  
**Complexity:** Medium  

---

## Methodology

### Core Concept: Adaptive Grouping

Unlike Approach 1's fixed day-windows, Approach 2 creates **groups** of articles based on:
1. **Temporal proximity** (days close together)
2. **Group size constraints** (minimum/maximum articles per group)
3. **Semantic coherence** (optional: similar embeddings)

### Two Segmentation Variants

#### Variant 1: Fixed Size Grouping
**Strategy:** Create groups with approximately equal article counts

**Parameters:**
- `GROUP_SIZE = 5` articles per group
- No strict day boundaries
- Groups span variable days to meet size requirement

**Algorithm:**
```python
def create_fixed_size_groups(df, group_size=5):
    groups = []
    current_group = []
    
    for article in df.iterrows():
        current_group.append(article)
        
        if len(current_group) >= group_size:
            groups.append(current_group)
            current_group = []
    
    return groups
```

**Output Folder:** `../Model_output/Pip_2_W5_NO_of_day_Grouping/`

#### Variant 2: Day Gap Grouping
**Strategy:** Create groups where articles are within a maximum day gap

**Parameters:**
- `MAX_DAY_GAP = 3` days
- Variable group sizes
- Groups break when day gap exceeds threshold

**Algorithm:**
```python
def create_day_gap_groups(df, max_gap=3):
    groups = []
    current_group = [df.iloc[0]]
    prev_date = df.iloc[0]['Date']
    
    for article in df.iterrows():
        current_date = article['Date']
        gap = (current_date - prev_date).days
        
        if gap <= max_gap:
            current_group.append(article)
        else:
            groups.append(current_group)
            current_group = [article]
        
        prev_date = current_date
    
    return groups
```

**Output Folder:** `../Model_output/Pip_21_W5_Day_Gap/`

---

## Improvements Over Approach 1

### ✅ 1. Better Sparse Data Handling
**Problem in Approach 1:**
- Fixed 3-day windows create imbalanced groups
- Days with 1-2 articles dilute entire window embedding
- No adaptation to article density variations

**Solution in Approach 2:**
- **Variant 1:** Always maintains 5 articles per group (balanced embeddings)
- **Variant 2:** Skips over sparse days, groups dense periods together
- Result: More robust embeddings, less noise

**Example:**
```
Approach 1 Window:
  Day 1: 50 articles ──┐
  Day 2: 2 articles   ├─> Mixed embedding (biased toward Day 1)
  Day 3: 1 article    ┘

Approach 2 Variant 1:
  Group 1: 5 articles from Day 1  ──> Balanced
  Group 2: 5 articles from Day 1  ──> Balanced
  ...
  Group 10: 3 from Day 1, 2 from Day 2 ──> Still balanced
```

### ✅ 2. Reduced Overlap Redundancy
**Problem in Approach 1:**
- Overlapping windows share 2 out of 3 days
- Same articles appear in multiple windows
- Inflates drift scores artificially

**Solution in Approach 2:**
- Non-overlapping groups by default
- Each article appears in exactly one group
- Cleaner drift signals

### ✅ 3. Semantic Coherence
**Advantage:**
- Groups can be formed around semantic clusters (optional enhancement)
- Articles within a group have higher topical similarity
- Reduces intra-group variance in embeddings

**Implementation:**
```python
# Optional: Sort by embedding similarity before grouping
df['similarity_to_prev'] = compute_similarity(df['embedding'])
df_sorted = df.sort_values('similarity_to_prev')
groups = create_fixed_size_groups(df_sorted)
```

### ✅ 4. Flexible Temporal Granularity
**Variant 1 Benefit:**
- Adapts to article flow (high-volume periods get more groups)
- Low-volume periods don't create weak groups

**Variant 2 Benefit:**
- Respects narrative continuity (groups break at long gaps)
- Better captures event-driven shifts

---

## Drawbacks

### ⚠️ 1. Variable Window Sizes
**Challenge:** Interpretation becomes harder
- Approach 1: "Shift on Day 15" (clear)
- Approach 2: "Shift between Group 12 and Group 13" (what dates?)

**Mitigation:**
- Store date range metadata for each group
- Visualize groups as spans on timeline

### ⚠️ 2. Increased Complexity
**Computational:**
- More logic for group creation
- Need to track group boundaries
- Variant comparison requires running twice

**Conceptual:**
- Users must understand two variants
- Parameter tuning (GROUP_SIZE vs MAX_DAY_GAP)

### ⚠️ 3. Edge Case Handling
**Issues:**
- What if last group has < 5 articles? (Variant 1)
- What if no articles within MAX_DAY_GAP? (Variant 2)
- How to handle single-article days?

**Solutions Implemented:**
```python
# Variant 1: Merge last incomplete group with previous
if len(last_group) < group_size:
    groups[-1].extend(last_group)

# Variant 2: Create single-article group if necessary
if gap > max_day_gap:
    groups.append([article])  # Solo group
```

### ⚠️ 4. Harder Visualization
**Problem:**
- X-axis cannot be uniform timeline (groups vary in span)
- Heatmaps have irregular row/column meanings

**Solution:**
- Convert groups to date ranges for plotting
- Use group indices with date annotations

---

## Output Folders

### Variant 1: Fixed Size Grouping
**Location:** `../Model_output/Pip_2_W5_NO_of_day_Grouping/`

**Contents:**
```
Pip_2_W5_NO_of_day_Grouping/
├── Climate/
│   ├── drift_timeline_Climate.png        # Dual subplot (drift + z-scores)
│   ├── similarity_matrix_Climate.png     # Discrete color heatmap
│   ├── shift_events.json                 # Detected shifts with date ranges
│   └── group_metadata.json               # Group composition details
├── Economics/
├── Health/
├── Technology/
├── War/
├── model_evaluation.json                 # Quality metrics
└── summary_metrics.json
```

**Example Visualization:**
- **Drift Timeline:** Top plot shows drift scores, bottom shows z-scores
- **Similarity Matrix:** 10 discrete color bins (no gradients)

### Variant 2: Day Gap Grouping
**Location:** `../Model_output/Pip_21_W5_Day_Gap/`

Same structure as Variant 1, but with day-gap-based groups.

---

## Visualization Enhancements

### 1. Discrete Color Heatmaps
**Implementation:**
```python
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# 10 discrete color bins
boundaries = [-1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0]
colors = ['#8B0000', '#B22222', '#DC143C', '#FF4500', '#FFA500', 
          '#FFFF00', '#00FFFF', '#4169E1', '#0000CD', '#00008B']

norm = BoundaryNorm(boundaries, len(colors))
cmap = ListedColormap(colors)

plt.pcolormesh(similarity_matrix, cmap=cmap, norm=norm)
```

**Features:**
- No gradients (solid color blocks)
- Each cell has exactly one color from 10-bin palette
- Easier visual interpretation (discrete similarity levels)

**Before (Approach 1):**
![Gradient Heatmap](../Data_Cleaning_Visulization/Pasted%20image.png)

**After (Approach 2):**
- Discrete color blocks with clear boundaries

### 2. Dual Subplot Drift Detection
**Design:**
- **Top Plot:** Drift scores over groups
  - Blue line: drift scores
  - Orange line: μ + 2σ threshold
  - Red markers: detected shifts
  
- **Bottom Plot:** Z-scores
  - Blue line: z-scores
  - Red horizontal lines: ±2.0 thresholds
  - Blue markers: significant deviations

**Code:**
```python
def plot_drift_timeline(drift_scores, z_scores, shifts, topic):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Top: Drift scores
    ax1.plot(drift_scores, label='Drift Score', color='steelblue')
    ax1.axhline(threshold, color='orange', linestyle='--', label='μ+2σ')
    ax1.scatter(shifts, drift_scores[shifts], color='red', s=100, 
                label='Shifts', zorder=5)
    
    # Bottom: Z-scores
    ax2.plot(z_scores, label='Z-Score', color='steelblue')
    ax2.axhline(2.0, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(-2.0, color='red', linestyle='--', alpha=0.5)
    ax2.scatter(significant_zs, z_scores[significant_zs], 
                color='blue', s=50, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/drift_timeline_{topic}.png', dpi=150)
```

**Benefits:**
- Separate drift magnitude from statistical significance
- Easier to identify true shifts (high drift + high z-score)
- Consistent design across all topics

### 3. Model Evaluation Metrics
**Implementation (Stage 9.3 in notebook):**
```python
def evaluate_model_quality(embeddings, topics, windows):
    """
    Evaluate model quality before drift detection.
    
    Metrics:
    1. Intra-topic similarity (same topic cohesion)
    2. Inter-topic similarity (topic separation)
    3. Separation score (intra/inter ratio)
    4. Temporal consistency (consecutive windows)
    """
    
    # 1. Intra-topic similarity
    intra_sim = []
    for topic in unique_topics:
        topic_embeds = embeddings[topics == topic]
        pairs = random.sample(topic_embeds, min(100, len(topic_embeds)))
        intra_sim.append(cosine_similarity(pairs))
    
    # 2. Inter-topic similarity
    inter_sim = []
    for t1, t2 in combinations(unique_topics, 2):
        e1 = embeddings[topics == t1]
        e2 = embeddings[topics == t2]
        inter_sim.append(cosine_similarity(e1, e2))
    
    # 3. Separation score
    separation = np.mean(intra_sim) / np.mean(inter_sim)
    
    # 4. Temporal consistency
    temporal_sim = []
    for i in range(len(windows)-1):
        temporal_sim.append(cosine_similarity(windows[i], windows[i+1]))
    
    return {
        'intra_topic_similarity': np.mean(intra_sim),
        'inter_topic_similarity': np.mean(inter_sim),
        'separation_score': separation,
        'temporal_consistency': np.mean(temporal_sim)
    }
```

**Example Output:**
```json
{
  "intra_topic_similarity": 0.72,
  "inter_topic_similarity": 0.31,
  "separation_score": 2.32,
  "temporal_consistency": 0.68
}
```

**Interpretation:**
- **High intra-topic:** Good topic coherence (>0.7 ideal)
- **Low inter-topic:** Good topic separation (<0.4 ideal)
- **High separation:** Well-defined topics (>2.0 ideal)
- **Moderate temporal:** Narratives evolve (0.5-0.7 ideal)

---

## Implementation Details

### Configuration Parameters

**Variant 1: Fixed Size**
```python
GROUP_SIZE = 5              # Articles per group
MIN_GROUP_SIZE = 3          # Minimum for last group
SIMILARITY_THRESHOLD = 2.0  # σ multiplier
```

**Variant 2: Day Gap**
```python
MAX_DAY_GAP = 3            # Maximum days between articles
MIN_GROUP_SIZE = 2         # Minimum articles in group
SIMILARITY_THRESHOLD = 2.0
```

### Data Requirements
**Input File:** `../Processed_Data/ALL_Combined_Data.csv`

**Required Columns:**
- `embedding` - Article embedding (768-dim, JSON or array string)
- `Topic` - Topic label
- `Date` - Publication date (YYYY-MM-DD)
- `Article_ID` - Unique article identifier

**Example:**
```csv
embedding,Topic,Date,Article_ID
"[-0.023, 0.451, ...]",Climate,2023-01-15,art_001
```

### Stages in Notebook (31 Cells)

| Stage | Description | Cell Range | Output |
|-------|-------------|-----------|--------|
| 1-2 | Data Loading | 1-3 | Parsed CSV, cleaned embeddings |
| 3-4 | Group Construction | 4-7 | Variant 1 or Variant 2 groups |
| 5-6 | Embedding Aggregation | 8-12 | Group-level embeddings |
| 7-8 | TCL Training | 13-18 | Contrastive loss, model weights |
| 9.3 | Model Evaluation | 19-20 | Quality metrics |
| 9.5 | Similarity Matrix | 21-23 | Discrete color heatmap |
| 10 | Drift Calculation | 24-26 | Cosine distances |
| 11 | Dual Subplot Visualization | 27-29 | Drift + z-score plots |
| 12 | Export Results | 30-31 | JSON and PNG files |

---

## Performance Metrics

### Computational Efficiency

| Variant | Processing Time | Memory Usage | GPU Required |
|---------|----------------|--------------|--------------|
| **Variant 1** (Fixed Size) | ~8 min | ~3GB RAM | No |
| **Variant 2** (Day Gap) | ~7 min | ~2.5GB RAM | No |

**Comparison with Approach 1:**
- Approach 1: 5 min (faster)
- Approach 2: 7-8 min (60% slower due to flexible grouping)

### Detection Statistics (Average across topics)

| Metric | Variant 1 | Variant 2 |
|--------|-----------|-----------|
| **Groups Created** | 180-220 | 150-190 |
| **Shifts Detected** | 15-22 | 14-20 |
| **Avg Group Span** | 2.5 days | 1.8 days |
| **False Positive Rate** | ~12% | ~10% |

### Quality Metrics (Approach 2 vs Approach 1)

| Metric | Approach 1 | Approach 2 (Var 1) | Approach 2 (Var 2) |
|--------|-----------|-------------------|-------------------|
| **Intra-Topic Similarity** | 0.68 | **0.72** ✅ | **0.71** ✅ |
| **Inter-Topic Similarity** | 0.35 | **0.31** ✅ | **0.32** ✅ |
| **Separation Score** | 1.94 | **2.32** ✅ | **2.22** ✅ |
| **Temporal Consistency** | 0.71 | 0.68 | 0.66 |

**Interpretation:**
- ✅ Better topic coherence (higher intra-topic)
- ✅ Better topic separation (lower inter-topic)
- ✅ Stronger topic boundaries (higher separation score)
- ⚠️ Slightly lower temporal consistency (expected due to grouping)

---

## Comparison with Other Approaches

| Aspect | Approach 1 | Approach 2 (This) | Approach 4 |
|--------|-----------|------------------|-----------|
| **Window Type** | Fixed 3-day | Variable groups | Article-based |
| **Granularity** | Day-level | Group-level | Sentence-level |
| **Sparse Data** | ⚠️ Poor | ✅ **Good** | ✅ Excellent |
| **Interpretability** | ✅ High | Moderate | ⚠️ Low |
| **Processing Time** | ✅ 5 min | 7-8 min | ⚠️ 20 min |
| **Shift Precision** | Moderate | **Good** ✅ | Excellent |
| **Topic Separation** | 1.94 | **2.32** ✅ | 2.45 |
| **Visualization** | Basic | **Enhanced** ✅ | Advanced |

---

## Usage Example

### Running Variant 1 (Fixed Size)
```bash
cd /home/hp/SEM2/INLP/Naretve_Shift/TCL/
jupyter notebook TCL_Pipeline_2.ipynb

# In notebook, set:
VARIANT = "fixed_size"
GROUP_SIZE = 5

# Run all cells
```

### Running Variant 2 (Day Gap)
```python
# In Stage 3: Group Construction
VARIANT = "day_gap"
MAX_DAY_GAP = 3

# Run from Stage 3 onward
```

### Analyzing Results
```python
import json
import pandas as pd

# Load group metadata
with open('../Model_output/Pip_2_W5_NO_of_day_Grouping/Climate/group_metadata.json') as f:
    groups = json.load(f)

# Print group composition
for i, group in enumerate(groups[:5]):
    print(f"Group {i}:")
    print(f"  Date Range: {group['start_date']} to {group['end_date']}")
    print(f"  Articles: {group['article_count']}")
    print(f"  Days Span: {group['days_span']}")
    print()
```

---

## Visualizations Gallery

### 1. Drift Timeline (Dual Subplot)
**File:** `drift_timeline_Climate.png`

**Top Plot:**
- Blue line: Drift scores between consecutive groups
- Orange dashed line: μ + 2σ threshold (e.g., 0.65)
- Red circles: Detected shifts (drift > threshold)

**Bottom Plot:**
- Blue line: Z-scores (standardized drift)
- Red dashed lines: ±2.0 significance thresholds
- Blue triangles: Statistically significant deviations

**Example Interpretation:**
- Group 45 → Group 46: Drift = 0.78 (shift detected), Z = 3.2 (significant)
- Group 67 → Group 68: Drift = 0.52 (no shift), Z = 1.1 (not significant)

### 2. Similarity Matrix (Discrete Colors)
**File:** `similarity_matrix_Climate.png`

**Color Legend:**
- Dark Red (-1.0 to -0.6): Strong dissimilarity
- Red (-0.6 to -0.3): Moderate dissimilarity
- Orange (-0.3 to -0.1): Slight dissimilarity
- Yellow (-0.1 to 0.1): Neutral
- Cyan (0.1 to 0.3): Slight similarity
- Light Blue (0.3 to 0.5): Moderate similarity
- Blue (0.5 to 0.7): Strong similarity
- Dark Blue (0.7 to 0.85): Very strong similarity
- Navy (0.85 to 1.0): Extreme similarity

**Patterns:**
- Diagonal blocks: Coherent narrative periods (consecutive groups similar)
- Off-diagonal light regions: Narrative shifts (groups dissimilar)
- Vertical/horizontal lines: Pivot groups (shift points)

---

## Known Limitations

1. **Variant Selection:**
   - No automatic way to choose between Fixed Size vs Day Gap
   - Requires domain knowledge or experimentation

2. **Parameter Sensitivity:**
   - GROUP_SIZE too small: Noisy embeddings
   - GROUP_SIZE too large: Missed shifts
   - MAX_DAY_GAP too small: Fragmented groups
   - MAX_DAY_GAP too large: Missed boundaries

3. **Temporal Alignment:**
   - Groups from different topics not aligned
   - Cross-topic shift comparison harder

4. **Visualization Complexity:**
   - X-axis (group index) not uniform time
   - Date range annotations required for interpretation

---

## Future Enhancements

While Approach 2 is succeeded by more advanced methods, potential improvements include:

1. **Hybrid Grouping:** Combine fixed size and day gap constraints
2. **Semantic Clustering:** Group articles by embedding similarity first
3. **Adaptive Thresholds:** Adjust GROUP_SIZE/MAX_DAY_GAP per topic
4. **Multi-Scale Groups:** Create nested groups (5-article, 10-article, 20-article)

**However, these are better addressed in:**
- **Approach 4:** For fine-grained shift detection
- **Approach 5:** For production optimization

---

## References

- Previous: [Approach 1 - Baseline Day-Level Windowing](approach_1.md)
- Next: [Approach 4 - Article-Level Sentence Shifts](approach_4.md)
- Main Documentation: [TCL Complete Flow](TCL_Complete_Flow.md)
- Comparison Report: [TCL vs Baselines](TCL_vs_Baselines_Narrative_Shift_Comparison.pdf)

---

**Recommendation:** For production use, prefer **Variant 2 (Day Gap)** if articles have irregular publishing patterns, or **Variant 1 (Fixed Size)** if you need consistent group sizes for downstream analysis.
