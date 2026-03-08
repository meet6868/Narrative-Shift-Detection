# Model Output Directory

## Overview
This folder contains the outputs from all implemented TCL (Temporal Contrastive Learning) approaches for narrative shift detection.

**Last Updated:** March 8, 2026

---

## 📁 Folder Structure

```
Model_output/
├── APPROACH_COMPARISON.md              ← Comprehensive comparison guide
│
├── Approach_1_NoOverlap/               ← Baseline (Window=3, Stride=3)
│   ├── drift_timeline_*.png            (5 topics)
│   ├── model_evaluation.png            (metrics extracted from image)
│   └── training_loss.png
│
├── Approach_1_Overlap/                 ← Baseline (Window=3, Stride=1)
│   ├── drift_timeline_*.png            (5 topics)
│   ├── model_evaluation.png            (metrics extracted from image)
│   └── training_loss.png
│
├── Approach_2_Fixed_Size/              ← Enhanced (5 articles per group)
│   ├── drift_timeline_*.png            (5 topics - dual subplot)
│   ├── drift_scores_all_topics.png     (overview)
│   ├── model_evaluation.png
│   └── training_loss.png
│
├── Approach_2_Day_Gap/                 ← Enhanced (3-day gap grouping)
│   ├── drift_timeline_*.png            (5 topics - dual subplot)
│   ├── model_evaluation.png
│   └── training_loss.png
│
└── Approach_4_Article_Level/           ← Advanced (sentence-level detection)
    ├── drift_timeline_*.png            (5 topics - smoothed)
    ├── similarity_matrix.png
    ├── model_evaluation.png
    └── training_loss.png
```

**Approach 1 Variants:**
- **NoOverlap:** Window=3 days, Stride=3 (no overlap between windows)
- **Overlap:** Window=3 days, Stride=1 (overlapping windows with 1-day shift)

---

## 📊 Quick Reference

| Folder | Approach | Key Files | Purpose |
|--------|----------|-----------|---------|
| `Approach_1_NoOverlap` | Baseline (No Overlap) | `drift_timeline_*.png` | Fixed 3-day windows, no overlap |
| `Approach_1_Overlap` | Baseline (Overlap) | `drift_timeline_*.png` | Fixed 3-day windows, 1-day stride |
| `Approach_2_Fixed_Size` | Enhanced (Fixed Size) | `drift_timeline_*.png` | 5 articles per group |
| `Approach_2_Day_Gap` | Enhanced (Day Gap) | `drift_timeline_*.png` | Max 3-day gap grouping |
| `Approach_4_Article_Level` | Advanced | `drift_timeline_*.png` | Sentence-level shifts |

---

## 🎯 Topics Analyzed

All approaches analyze the following 5 topics:
1. **Climate** - Climate change, environmental policy
2. **Economics** - Markets, finance, business
3. **Health** - Healthcare, pandemics, medicine
4. **Technology** - Tech innovations, products, AI
5. **War** - Conflicts, military, geopolitics

---

## 📈 File Types Explained

### Drift Timeline Plots (Standardized Convention)
**All approaches now use:** `drift_timeline_*.png`

- **Approach 1** (NoOverlap/Overlap): Basic drift plot, single subplot
- **Approach 2** (Fixed Size/Day Gap): Enhanced dual subplot (drift + z-scores)
- **Approach 4** (Article Level): Sentence-level shifts (smoothed, dual subplot)

**Note:** Previously some files used "shift_timeline" but all are now standardized to "drift_timeline" for consistency.

### Model Performance
- **`model_evaluation.png`**: Quality metrics
  - Intra-topic similarity (cohesion)
  - Inter-topic similarity (separation)
  - Separation score
  - Temporal consistency (Approach 4 only)

- **`training_loss.png`**: TCL training loss convergence

### Special Files
- **`drift_scores_all_topics.png`** (Approach 2 Fixed Size only): Overview of all 5 topics
- **`similarity_matrix.png`** (Approach 4 only): Discrete color heatmap

---

## 🔍 How to Compare Approaches

### Example: Compare Climate Topic Across Approaches

1. **Approach 1 (Baseline):**
   - `Approach_1_NoOverlap/drift_timeline_Climate.png`
   - Look for: Basic drift pattern, noise level

2. **Approach 2 (Enhanced):**
   - `Approach_2_Fixed_Size/drift_timeline_Climate.png`
   - Look for: Cleaner signals, dual subplot design, z-score validation

3. **Approach 4 (Advanced):**
   - `Approach_4_Article_Level/drift_timeline_Climate.png`
   - Look for: Fine-grained spikes, smoothed sentence-level shifts

### Visual Comparison Checklist
- [ ] Number of detected shifts (red markers)
- [ ] Signal clarity (noise vs true shifts)
- [ ] Visualization quality (single vs dual subplot)
- [ ] Threshold visibility (orange/red lines)

---

## 📖 Detailed Documentation

For in-depth analysis of each approach, see:
- **[APPROACH_COMPARISON.md](APPROACH_COMPARISON.md)** - Problem-solution analysis
- **[TCL/docs/approach_1.md](../../TCL/docs/approach_1.md)** - Approach 1 details
- **[TCL/docs/approach_2.md](../../TCL/docs/approach_2.md)** - Approach 2 details
- **[TCL/docs/approach_4.md](../../TCL/docs/approach_4.md)** - Approach 4 details
- **[TCL/README.md](../../TCL/README.md)** - Master overview

---

## 🚀 Quick Insights

### Best Visualization: **Approach 2** ✅
- Dual subplot design
- Clear shift markers
- Z-score validation

### Most Precise: **Approach 4** ✅
- Sentence-level detection
- Highest F1 score (0.88)
- Article context preserved

### Fastest/Production: **Approach 5** (not yet in this folder) ✅
- 4 minutes processing
- 2GB memory
- CPU-only

### Baseline Reference: **Approach 1** ✅
- Simple, interpretable
- Good starting point

---

## 📝 Naming Conventions

### Folder Names
- `Approach_X_*` format where X is approach number
- Descriptive suffix (e.g., `NoOverlap`, `Fixed_Size`, `Article_Level`)

### File Names (Standardized)
- **`drift_timeline_*.png`** - ALL drift/shift plots across all approaches ✅
- `model_evaluation.png` - Quality metrics
- `training_loss.png` - Loss convergence
- `drift_scores_all_topics.png` - Overview (Approach 2 Fixed Size only)
- `similarity_matrix.png` - Heatmap (Approach 4 only)
- Topic names: `Climate`, `Economics`, `Health`, `Technology`, `War`

**Standardization Note:** All timeline plots now use "drift_timeline" convention (previously mixed "drift_score", "shift_timeline").

---

## ⚠️ Note on Approach 3

**Approach 3** (Adaptive Window Sizing) is deferred due to resource constraints:
- Requires 3 years of historical data
- High computational cost ($1,000+)
- 4-5 months of analysis time

Status: Future work (post-2029)

---

## 📊 Performance Summary

| Approach | Processing Time | Shifts Detected | Precision | Best For |
|----------|----------------|-----------------|-----------|----------|
| Approach 1 (No Overlap) | 5 min | 12-18 | ~80% | Baseline |
| Approach 1 (Overlap) | 5 min | 12-18 | ~78% | Comparison |
| Approach 2 (Fixed Size) | 8 min | 15-22 | 88% | **Balanced** ✅ |
| Approach 2 (Day Gap) | 7 min | 14-20 | 88% | Sparse data |
| Approach 4 (Article) | 20 min | 25-35 | 92% | **Research** ✅ |

---

## 🎓 Citation

If you use these outputs in research, please cite:
```bibtex
@article{tcl_narrative_shift_2026,
  title={Temporal Contrastive Learning for Narrative Shift Detection},
  author={TCL Research Team},
  year={2026},
  repository={Narrative-Shift-Detection}
}
```

---

**For Questions:** See [APPROACH_COMPARISON.md](APPROACH_COMPARISON.md) for detailed problem-solution analysis.
