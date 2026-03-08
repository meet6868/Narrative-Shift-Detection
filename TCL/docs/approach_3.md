# Approach 3: Adaptive Window Sizing Based on Semantic Drift

## Status

⚠️ **Theoretical - Not Yet Implemented** (Resource and Time Intensive)

---

## Overview

Approach 3 proposes **adaptive window sizing per topic** based on semantic drift patterns discovered through multi-year historical analysis. Unlike Approaches 1 and 2 which use fixed windows (3 days) or fixed grouping rules (5 articles), this approach dynamically determines optimal window sizes for each topic by analyzing long-term semantic drift characteristics.

**Pipeline File:** None (not yet implemented)  
**Status:** ⚠️ Theoretical / Research Phase  
**Reason:** Requires extensive computational resources and time for historical drift analysis  

---

## Methodology

### Core Concept: Topic-Specific Optimal Windows

**Key Insight:**
Different topics have different narrative rhythms. Some topics (e.g., War, Breaking News) shift rapidly, while others (e.g., Climate Policy, Economics) have slower, more gradual shifts.

**Approach 3 Hypothesis:**
> "Instead of using a one-size-fits-all 3-day window, we should discover the optimal window size for each topic by analyzing semantic drift patterns over historical data."

### Three-Phase Process

#### Phase 1: Historical Semantic Drift Analysis
**Objective:** Analyze 1-3 years of historical data per topic to find drift patterns

**Process:**
```python
def analyze_historical_drift(topic_data, years=3):
    """
    Analyze semantic drift over multiple years to find patterns.
    
    Steps:
    1. Create windows of varying sizes (1-day, 2-day, 3-day, ..., 30-day)
    2. Calculate drift scores for each window size
    3. Identify window size with most stable/meaningful shifts
    4. Analyze shift frequency patterns
    """
    
    window_sizes = range(1, 31)  # Test 1 to 30 days
    drift_analysis = {}
    
    for window_size in window_sizes:
        windows = create_windows(topic_data, size=window_size)
        drift_scores = compute_drift(windows)
        
        drift_analysis[window_size] = {
            'avg_drift': np.mean(drift_scores),
            'std_drift': np.std(drift_scores),
            'num_shifts': count_shifts(drift_scores),
            'shift_clarity': measure_shift_clarity(drift_scores),
            'temporal_coverage': len(windows)
        }
    
    # Find optimal window size
    optimal_size = select_optimal_window(drift_analysis)
    return optimal_size, drift_analysis
```

**Metrics for Optimal Window Selection:**
1. **Shift Clarity:** Clear distinction between shift and non-shift periods
2. **Shift Frequency:** Not too many (noisy) or too few (missing shifts)
3. **Temporal Coverage:** Adequate number of windows for analysis
4. **Variance Ratio:** High between-shift vs within-window variance

**Expected Results:**
```
Topic: War
  Optimal Window Size: 1-2 days (fast-moving, breaking news)
  Reasoning: Conflicts escalate/de-escalate rapidly

Topic: Climate
  Optimal Window Size: 7-14 days (slow-moving, policy-driven)
  Reasoning: Climate narratives change around conferences, reports

Topic: Economics
  Optimal Window Size: 3-5 days (moderate pace)
  Reasoning: Market cycles, quarterly reports

Topic: Health
  Optimal Window Size: 2-4 days (variable, event-driven)
  Reasoning: Pandemics (fast) vs chronic issues (slow)

Topic: Technology
  Optimal Window Size: 3-7 days (moderate-fast)
  Reasoning: Product launches, tech news cycles
```

#### Phase 2: Window Size Determination Table
**Objective:** Create a lookup table of optimal window sizes per topic

**Output:**
```json
{
  "optimal_windows": {
    "Climate": {
      "window_size": 10,
      "confidence": 0.87,
      "analysis_period": "2023-01-01 to 2025-12-31",
      "shift_frequency": "~2 shifts per month",
      "rationale": "Climate narratives align with bi-weekly news cycles and monthly policy events"
    },
    "Economics": {
      "window_size": 4,
      "confidence": 0.92,
      "analysis_period": "2023-01-01 to 2025-12-31",
      "shift_frequency": "~8 shifts per month",
      "rationale": "Economic news driven by weekly market reports and quarterly earnings"
    },
    "War": {
      "window_size": 2,
      "confidence": 0.79,
      "analysis_period": "2023-01-01 to 2025-12-31",
      "shift_frequency": "~15 shifts per month",
      "rationale": "Conflict news is highly dynamic with daily developments"
    },
    "Health": {
      "window_size": 3,
      "confidence": 0.84,
      "analysis_period": "2023-01-01 to 2025-12-31",
      "shift_frequency": "~10 shifts per month",
      "rationale": "Health news mixes breaking events (pandemics) with ongoing trends"
    },
    "Technology": {
      "window_size": 5,
      "confidence": 0.88,
      "analysis_period": "2023-01-01 to 2025-12-31",
      "shift_frequency": "~6 shifts per month",
      "rationale": "Tech news cycles around product launches and major announcements"
    }
  },
  "metadata": {
    "analysis_date": "2026-03-08",
    "total_articles_analyzed": 150000,
    "years_of_data": 3,
    "computation_time": "72 hours (GPU cluster)"
  }
}
```

#### Phase 3: Deployment with Adaptive Windows
**Objective:** Use discovered window sizes in production

**Implementation:**
```python
def create_adaptive_windows(articles, optimal_window_config):
    """
    Create windows with topic-specific sizes.
    """
    windows = {}
    
    for topic in articles['Topic'].unique():
        topic_data = articles[articles['Topic'] == topic]
        window_size = optimal_window_config[topic]['window_size']
        
        # Create windows with topic-specific size
        topic_windows = create_windows(topic_data, size=window_size)
        windows[topic] = topic_windows
    
    return windows

# Usage
optimal_config = load_json('optimal_windows.json')
windows = create_adaptive_windows(articles, optimal_config['optimal_windows'])
```

---

## Improvements Over Approach 2

### ✅ 1. Topic-Aware Windowing
**Approach 2 Problem:**
- All topics use same grouping strategy (5 articles or 3-day gap)
- War topic: Too slow (misses rapid shifts)
- Climate topic: Too fast (creates noise)

**Approach 3 Solution:**
- War: 1-2 day windows (captures rapid escalations)
- Climate: 7-14 day windows (captures policy shifts, not daily noise)
- Each topic optimized for its natural narrative rhythm

**Example:**
```
War Topic Shift:
  Approach 2: Detected between Group 45-46 (5 articles, spans 6 days)
  Approach 3: Detected between Window 45-46 (2 days, precise)
  
Climate Topic Shift:
  Approach 2: Detected between Group 12-13 (false positive, daily noise)
  Approach 3: Detected between Window 5-6 (10 days, true policy shift)
```

### ✅ 2. Data-Driven Window Selection
**Approach 2 Limitation:**
- Window size (3 days) chosen arbitrarily
- No empirical validation

**Approach 3 Advantage:**
- Window sizes discovered from data
- Backed by 3 years of historical analysis
- Confidence scores for each topic

### ✅ 3. Reduced False Positives
**Impact:**
- Slow-moving topics (Climate): Larger windows → Less noise
- Fast-moving topics (War): Smaller windows → Better precision
- Expected: 20-30% reduction in false positives

### ✅ 4. Better Semantic Alignment
**Benefit:**
- Windows align with natural narrative boundaries
- Climate: ~2 weeks (conference cycles)
- Economics: ~4 days (weekly reports)
- War: ~2 days (daily briefings)

---

## Drawbacks

### ⚠️ 1. Requires Extensive Historical Data
**Challenge:** Need 1-3 years of data per topic

**Resource Requirements:**
- **Data:** 50,000-150,000 articles per topic
- **Storage:** ~100GB for embeddings and metadata
- **Processing:** Must analyze all historical data

**Barrier:**
- New topics: Cannot apply (no historical data)
- Emerging narratives: Cannot detect optimal window in real-time

**Example Issue:**
```
New Topic: "AI Regulation" (emerged 2025)
  Historical Data Available: 6 months
  Required: 3 years
  Result: Cannot reliably determine optimal window
  Fallback: Use default 3-day window (same as Approach 1)
```

### ⚠️ 2. Computational Cost
**Phase 1 Analysis Requirements:**

| Resource | Requirement | Cost |
|----------|-------------|------|
| **GPU Cluster** | 4-8 GPUs (A100 or equivalent) | $500-1000 |
| **Processing Time** | 48-72 hours per topic | - |
| **Total Time** | 240-360 hours (5 topics) | - |
| **RAM** | 128GB+ | - |
| **Storage** | 500GB-1TB | $50-100 |

**Cost Breakdown:**
```
Cloud Computing (AWS/GCP):
  - GPU hours: 360 hours × $2.50/hour = $900
  - Storage: 1TB × $0.10/GB/month = $100
  - Data transfer: $50
  Total: ~$1,050 per analysis cycle

On-Premise:
  - Initial hardware: $15,000-30,000
  - Electricity: $200-400 per analysis
  - Maintenance: $2,000/year
```

**Comparison with Other Approaches:**
```
Approach 1: $0 (runs on laptop)
Approach 2: $0 (runs on laptop)
Approach 3: $1,000+ (requires cluster) ⚠️
Approach 4: $50 (GPU server for 20 min)
Approach 5: $0 (runs on laptop)
```

### ⚠️ 3. Time Investment
**Timeline:**
```
Month 1: Data Collection & Preprocessing
  - Gather 3 years of historical articles
  - Clean and prepare embeddings
  - Set up computing infrastructure

Month 2-3: Historical Drift Analysis
  - Run analysis for each topic
  - Test window sizes 1-30 days
  - Calculate drift metrics
  - Compute optimal windows

Month 4: Validation & Refinement
  - Cross-validate on held-out data
  - Adjust window sizes based on performance
  - Create confidence intervals

Month 5: Documentation & Deployment
  - Document findings
  - Create optimal window configuration
  - Implement adaptive windowing

Total: 4-5 months of research time
```

**Personnel Requirements:**
- 1-2 researchers (full-time)
- 1 data engineer (part-time)
- Domain experts for validation

### ⚠️ 4. Non-Stationarity Risk
**Problem:** Optimal windows may change over time

**Example:**
```
Climate Topic:
  2020-2022: Optimal window = 14 days (slow policy cycles)
  2023-2025: Optimal window = 7 days (increased urgency, more events)
  2026+: Optimal window = ? (may need re-analysis)
```

**Mitigation Needed:**
- Re-run analysis every 6-12 months
- Monitor drift pattern changes
- Adaptive re-calibration

**Additional Cost:**
- Bi-annual re-analysis: $1,000 × 2 = $2,000/year
- Ongoing maintenance

### ⚠️ 5. Complexity
**Implementation Complexity:**
- Need separate codepaths for each topic
- Window size lookup table maintenance
- Confidence score tracking
- Fallback logic for new topics

**Code Overhead:**
```python
# Approach 1-2: Simple
windows = create_windows(articles, window_size=3)

# Approach 3: Complex
optimal_config = load_optimal_windows()
windows = {}
for topic in topics:
    if topic in optimal_config:
        size = optimal_config[topic]['window_size']
        confidence = optimal_config[topic]['confidence']
        
        if confidence < 0.7:
            # Low confidence, use default
            size = 3
            log_warning(f"Low confidence for {topic}, using default")
    else:
        # New topic, use default
        size = 3
        log_warning(f"No config for {topic}, using default")
    
    windows[topic] = create_windows(
        articles[articles['Topic'] == topic], 
        size=size
    )
```

**Lines of Code:**
- Approach 1-2: ~1,200 lines
- Approach 3: ~2,500 lines (2× more)

---

## Why Not Yet Implemented

### Decision Rationale

| Factor | Status | Impact |
|--------|--------|--------|
| **Resource Availability** | ❌ Limited | Cannot allocate GPU cluster for 72 hours |
| **Time Constraints** | ❌ 4-5 months | Current project timeline: 2-3 months |
| **Historical Data** | ⚠️ Partial | Only 1 year available, need 3 years |
| **ROI Uncertainty** | ⚠️ Unknown | Unclear if 20-30% improvement justifies cost |
| **Maintenance Burden** | ❌ High | Requires bi-annual re-analysis |

### Current Decision
**Defer to Future Work:**
- Complete Approaches 1, 2, 4, 5 first
- Gather more historical data over time
- Seek funding for GPU cluster
- Publish initial results, then pursue Approach 3 as extension

---

## Implementation Roadmap (If Pursued)

### Phase 1: Data Collection (Month 1)
**Tasks:**
- [ ] Collect 3 years of articles per topic (2023-2025)
- [ ] Generate embeddings for all historical articles
- [ ] Validate data quality and coverage
- [ ] Set up data storage infrastructure

**Deliverables:**
- Historical dataset: 150K+ articles
- Embedding database: ~100GB
- Data quality report

### Phase 2: Infrastructure Setup (Month 1-2)
**Tasks:**
- [ ] Provision GPU cluster (4-8 GPUs)
- [ ] Install required libraries and frameworks
- [ ] Create distributed processing pipeline
- [ ] Set up monitoring and logging

**Deliverables:**
- GPU cluster ready
- Processing scripts tested
- Monitoring dashboard

### Phase 3: Drift Analysis (Month 2-3)
**Tasks:**
- [ ] Run window size experiments (1-30 days) per topic
- [ ] Calculate drift metrics for each configuration
- [ ] Identify optimal window sizes
- [ ] Compute confidence scores

**Deliverables:**
- Drift analysis results per topic
- Optimal window size table
- Confidence intervals

### Phase 4: Validation (Month 4)
**Tasks:**
- [ ] Cross-validate on held-out data (2026 articles)
- [ ] Compare with Approach 1, 2 baselines
- [ ] Calculate precision, recall, F1 scores
- [ ] Measure false positive reduction

**Deliverables:**
- Validation report
- Performance comparison tables
- Statistical significance tests

### Phase 5: Deployment (Month 5)
**Tasks:**
- [ ] Implement adaptive windowing in pipeline
- [ ] Create optimal window configuration file
- [ ] Set up fallback logic for new topics
- [ ] Document methodology and results

**Deliverables:**
- Production-ready Approach 3 pipeline
- Optimal window configuration JSON
- Research paper/report

---

## Expected Performance (Projected)

### Accuracy Improvements (Estimated)

| Metric | Approach 2 | Approach 3 (Projected) | Improvement |
|--------|-----------|------------------------|-------------|
| **Precision** | 88% | **92-95%** | +4-7% |
| **Recall** | 75% | **80-85%** | +5-10% |
| **F1 Score** | 0.81 | **0.86-0.90** | +5-9% |
| **False Positives** | 15% | **8-12%** | -3-7% |

### Topic-Specific Benefits

**Climate Topic:**
- Approach 2: 18 shifts detected (6 false positives)
- Approach 3: 13 shifts detected (1-2 false positives)
- Benefit: Larger windows reduce daily noise

**War Topic:**
- Approach 2: 12 shifts detected (3 missed)
- Approach 3: 16 shifts detected (1 missed)
- Benefit: Smaller windows capture rapid developments

---

## Alternative: Lightweight Adaptive Approach

**If full Approach 3 is too resource-intensive, consider:**

### Simplified Version: Rule-Based Adaptive Windows
```python
# Instead of 3-year analysis, use domain knowledge
TOPIC_WINDOWS = {
    'War': 2,           # Breaking news, fast-moving
    'Health': 3,        # Moderate pace
    'Economics': 4,     # Weekly cycles
    'Technology': 5,    # Product launch cycles
    'Climate': 10       # Slow policy cycles
}

# No historical analysis required
# Based on domain expert input
# Cost: $0, Time: 1 day
```

**Trade-offs:**
- ✅ No computational cost
- ✅ Immediate implementation
- ⚠️ Not data-driven (subjective)
- ⚠️ May not be optimal

---

## Comparison with Other Approaches

| Feature | Approach 1 | Approach 2 | Approach 3 | Approach 4 |
|---------|-----------|-----------|-----------|-----------|
| **Window Type** | Fixed 3-day | Variable groups | **Topic-adaptive** | Article-based |
| **Customization** | None | Rule-based | **Data-driven** ✅ | N/A |
| **Historical Analysis** | No | No | **Yes (3 years)** | No |
| **Computational Cost** | Low | Low | **Very High** ⚠️ | High |
| **Time Investment** | 1 week | 2 weeks | **4-5 months** ⚠️ | 3 weeks |
| **False Positive Reduction** | Baseline | 10-15% | **20-30%** ✅ | 15-20% |
| **Topic Awareness** | No | No | **Yes** ✅ | No |

---

## Conclusion

**Approach 3 Status: Deferred**

While theoretically superior, Approach 3 is **not currently feasible** due to:
1. ⏳ Time requirements (4-5 months)
2. 💰 Computational costs ($1,000+ per analysis)
3. 📊 Historical data needs (3 years per topic)
4. 🔧 Ongoing maintenance (bi-annual re-analysis)

**Recommended Path:**
1. Implement Approaches 1, 2, 4, 5 first
2. Collect more historical data over next 1-2 years
3. Pursue Approach 3 as future research extension
4. Consider simplified rule-based version as interim solution

**When to Revisit:**
- After 3 years of data collection (2026-2029)
- When GPU cluster resources become available
- If funding secured for computational infrastructure
- For publication/academic extension of this work

---

## References

- Previous: [Approach 2 - Group-Based Segmentation](approach_2.md)
- Next: [Approach 4 - Article-Level Sentence Shifts](approach_4.md)
- Main Documentation: [README](../README.md)
