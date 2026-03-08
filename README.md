# Narrative Shift Detection using Temporal Contrastive Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()

A comprehensive framework for detecting narrative shifts in news articles using **Temporal Contrastive Learning (TCL)**. This project implements multiple approaches ranging from baseline windowing to advanced dynamic segmentation with entity-aware tracking.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Approaches Overview](#approaches-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Results](#results)
- [Documentation](#documentation)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Roadmap](#roadmap)

---

## 🎯 Overview

This project detects **narrative shifts** in news coverage across five major topics:

- **War & Conflict**
- **Health & Medicine**
- **Economics & Business**
- **Technology & Innovation**
- **Climate & Environment**

### What is a Narrative Shift?

A narrative shift occurs when the framing, sentiment, or focus of news coverage changes over time. For example:

- **Climate Change:** From "climate skepticism" → "climate emergency"
- **COVID-19:** From "foreign virus" → "public health crisis"
- **AI Technology:** From "automation threat" → "productivity tool"

### Methodology

We employ **Temporal Contrastive Learning (TCL)** to:

1. Learn temporally-aware embeddings of news articles
2. Detect semantic shifts between consecutive time windows
3. Quantify narrative drift with statistical significance testing
4. Track entity-specific narrative evolution (Approach 5)

---

## 📁 Project Structure

```
Naretve_Shift/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── INLP.pdf                          # Project documentation
├── reference.md                       # Research references
│
├── TCL/                              # Main implementation folder
│   ├── README.md                     # TCL approaches overview
│   ├── TCL_Pipeline_1.ipynb          # Approach 1: Baseline windowing
│   ├── TCL_Pipeline_2.ipynb          # Approach 2: Group-based segmentation
│   ├── TCL_Pipeline_4.ipynb          # Approach 4: Dynamic Ruptures segmentation
│   ├── TCL_Pipeline_5.ipynb          # Approach 5: NER-enhanced (in development)
│   └── docs/                         # Detailed documentation
│       ├── approach_1.md             # Approach 1 methodology
│       ├── approach_2.md             # Approach 2 methodology
│       ├── approach_3.md             # Approach 3 (deferred)
│       ├── approach_4.md             # Approach 4 methodology
│       ├── approach_5.md             # Approach 5 methodology
│       ├── TCL_Framework_Complete.pdf # Complete framework PDF
│       └── TCL_Framework_Complete.tex # LaTeX source
│
├── Pre_Processing/                   # Data preprocessing scripts
│   ├── CSV_Explorer.ipynb            # Data exploration
│   ├── Data_Combine.ipynb            # Dataset combination
│   ├── Data_Preprocessing.ipynb      # Data cleaning & preparation
│   └── Narrative_Shift_Detection.ipynb # Initial detection experiments
│
├── Processed_Data/                   # Cleaned and processed datasets
│   ├── all_articles_english_with_dates.csv
│   ├── ALL_Combined_Data.csv
│   ├── topic_embeddings.json
│   ├── Topic_Wise_w3/               # Topic-wise data (window=3)
│   ├── Topic_Wise_w5/               # Topic-wise data (window=5)
│   └── Soft_Labeling_Topic_Articles/ # Soft-labeled articles
│
├── DATA/                             # Raw datasets
│   ├── CNN_Articels_clean.csv
│   ├── Covid_News.csv
│   ├── NewsData.io_Sample_data_crypto.csv
│   └── Webhose free-news-datasets master News_Datasets/
│
├── Output/                           # Results and outputs
│   ├── Model/                        # Trained model checkpoints
│   ├── Model_output/                 # Approach-wise results
│   │   ├── APPROACH_COMPARISON.md    # Comprehensive comparison
│   │   ├── Approach_1_NoOverlap/     # Baseline (W=3, S=3)
│   │   ├── Approach_1_Overlap/       # Baseline (W=3, S=1)
│   │   ├── Approach_2_Fixed_Size/    # 3-day fixed groups
│   │   ├── Approach_2_Day_Gap/       # Max 3-day gap groups
│   │   └── Approach_4_Article_Level/ # Dynamic segmentation
│   ├── Data_Cleaning_Visualization/  # Data quality visualizations
│   ├── Narrative_Shift_Window_Detection/ # Change point analysis
│   └── Topic_Labeling_Comparison/    # Topic modeling results
│
├── Research_Paper/                   # Academic references
│   ├── Bayesian Online Changepoint Detection.pdf
│   ├── Dynamic Topic Models.pdf
│   ├── Event_Segnment.pdf
│   ├── TCL_vs_Baselines_Narrative_Shift_Comparison.pdf
│   └── TCL_Complete_Flow.md
│
└── Input/                            # User input samples
    └── sample_articles/
```

---

## 🚀 Approaches Overview

We implemented **5 approaches**, each building upon the previous:

### Approach 1: Baseline Day-Level Windowing

**Status:** ✅ Complete  
**Method:** Fixed 3-day temporal windows  
**Variants:** 2 (Overlap W=3,S=1 / NoOverlap W=3,S=3)

**Metrics:**
- Intra-Topic Similarity: 0.2182 (NoOverlap)
- Temporal Consistency: 0.9155
- Separation Score: -4.78

**Strengths:**
- Simple, interpretable baseline
- Clear temporal boundaries

**Limitations:**
- Fixed window size (inflexible)
- Sparse data handling issues
- No entity awareness

[📄 Full Documentation](TCL/docs/approach_1.md)

---

### Approach 2: Group-Based Segmentation

**Status:** ✅ Complete  
**Method:** Adaptive article grouping (day-based)  
**Variants:** 2 (Fixed Day / Day Gap)

**Metrics (Fixed Day):**
- Intra-Topic Similarity: 0.3365 (+54% vs Approach 1)
- Temporal Consistency: 0.9193
- Total Windows: 669

**Metrics (Day Gap):**
- Intra-Topic Similarity: 0.3185
- Temporal Consistency: 0.8948
- Total Windows: 732
- Better Separation Score: -8.83

**Strengths:**
- Better sparse data handling
- Improved intra-topic coherence
- Enhanced visualizations (dual subplots)

**Limitations:**
- Variable window interpretation
- Still day-level aggregation
- No entity awareness

[📄 Full Documentation](TCL/docs/approach_2.md)

---

### Approach 3: Adaptive Window Sizing

**Status:** ⏸️ Deferred  
**Method:** Topic-specific window optimization  

**Reason for Deferral:**
- Requires 3+ years of historical data
- High computational cost (GPU cluster)
- Timeline constraint (2029+)

[📄 Full Documentation](TCL/docs/approach_3.md)

---

### Approach 4: Dynamic Ruptures Segmentation

**Status:** ✅ Complete  
**Method:** PELT algorithm with RBF kernel for semantic change points  
**Configuration:** penalty=0.1, min_size=2 days

**Metrics:**
- Intra-Topic Similarity: **0.9997** 🏆 (Nearly perfect!)
- Temporal Consistency: **0.9877** 🏆 (Best!)
- Separation Score: 1.0872
- Total Windows: 356 (dynamic)

**Strengths:**
- Extremely high coherence (0.9997)
- Best temporal consistency (0.9877)
- Dynamic adaptive segmentation
- Sentence-level granularity

**Limitations:**
- Weak separation score (< 2.0)
- No entity awareness
- Computationally intensive

**Innovation:** Uses Ruptures library for automatic change point detection based on semantic shifts rather than fixed time windows.

[📄 Full Documentation](TCL/docs/approach_4.md)

---

### Approach 5: NER-Enhanced Entity-Aware Tracking

**Status:** 🚧 Under Development  
**Method:** Entity-aware embeddings with spaCy NER  

**Planned Features:**
- Track entity-specific narratives
- Filter shifts by entity type (PERSON, ORG, GPE, EVENT)
- Reduce false positives from entity changes
- Query capability: "Show Biden narrative shifts"

**Expected Impact:**
- 25% reduction in false positives
- Entity-level drift tracking
- Production-ready performance

[📄 Full Documentation](TCL/docs/approach_5.md)

---

## ✨ Key Features

### 1. **Multi-Approach Framework**
- Baseline to advanced techniques
- Comprehensive performance comparison
- Real metrics from actual implementations

### 2. **Temporal Contrastive Learning**
- Learn time-aware embeddings
- Contrastive loss for topic separation
- Temporal consistency metrics

### 3. **Dynamic Segmentation (Approach 4)**
- Ruptures PELT algorithm
- RBF kernel for semantic boundaries
- Adaptive segment sizing (2-40 days)

### 4. **Comprehensive Evaluation**
- Intra-topic similarity (coherence)
- Inter-topic similarity (separation)
- Separation score (distinctness)
- Temporal consistency (continuity)

### 5. **Rich Visualizations**
- Drift timeline plots
- Z-score significance testing
- Similarity matrix heatmaps
- Model evaluation metrics

### 6. **Entity-Aware Tracking (Approach 5)**
- Named Entity Recognition (NER)
- Entity-specific narratives
- Reduced false positives

---

## �� Results Summary

### Performance Comparison (All Approaches)

| Approach | Intra-Topic | Temporal | Separation | Windows | Entity-Aware |
|----------|-------------|----------|------------|---------|--------------|
| **1 (NoOverlap)** | 0.2182 | 0.9155 | -4.78 | N/A | ❌ |
| **1 (Overlap)** | 0.1431 | 0.8978 | -5.01 | N/A | ❌ |
| **2 (Fixed Day)** | 0.3365 ⬆️ | 0.9193 | -29.56 | 669 | ❌ |
| **2 (Day Gap)** | 0.3185 ⬆️ | 0.8948 | -8.83 | 732 | ❌ |
| **4 (Ruptures)** | **0.9997** 🏆 | **0.9877** �� | 1.09 | 356 | ❌ |
| **5 (NER)** | TBD | TBD | TBD | TBD | ✅ |

### Key Findings

**🏆 Best Overall Performance:** Approach 4 (Ruptures)
- Nearly perfect intra-topic similarity (0.9997)
- Highest temporal consistency (0.9877)
- Dynamic adaptive segmentation

**🏆 Best for Sparse Data:** Approach 2 Day Gap
- Most windows generated (732)
- Adaptive to data gaps
- Better separation score

**🏆 Best for Production (Planned):** Approach 5
- Entity-aware tracking
- Reduced false positives
- Query by entity capability

[📊 Detailed Comparison](Output/Model_output/APPROACH_COMPARISON.md)

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM

### Setup

```bash
# Clone repository
git clone https://github.com/meet6868/Narrative-Shift-Detection.git
cd Narrative-Shift-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (for Approach 5)
python -m spacy download en_core_web_sm
```

### Dependencies

```
torch>=1.9.0
transformers>=4.18.0
sentence-transformers>=2.2.0
ruptures>=1.1.7
spacy>=3.2.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## 🚀 Quick Start

### 1. Data Preprocessing

```bash
# Navigate to preprocessing folder
cd Pre_Processing

# Run preprocessing pipeline
jupyter notebook Data_Preprocessing.ipynb
```

### 2. Run TCL Pipeline (Approach 4 - Best Performance)

```bash
cd ../TCL

# Open and run the pipeline
jupyter notebook TCL_Pipeline_4.ipynb
```

### 3. View Results

```bash
# Check output folder
cd ../Output/Model_output/Approach_4_Article_Level/

# View drift timelines
ls drift_timeline_*.png

# See model evaluation
cat evaluation_metrics.txt
```

---

## 📖 Documentation

- **[TCL Framework Complete](TCL/docs/TCL_Framework_Complete.pdf)** - Comprehensive technical documentation
- **[Approach Comparison](Output/Model_output/APPROACH_COMPARISON.md)** - Detailed performance comparison
- **[Research References](reference.md)** - Academic papers and citations

---

## 📊 Dataset

### Data Sources

- **US Politics News Sentiment Dataset** (primary)
- **CNN Articles** (supplementary)
- **COVID-19 News** (supplementary)
- **Webhose News Datasets** (topic-specific)

### Topics Covered

1. War & Conflict
2. Health & Medicine
3. Economics & Business
4. Technology & Innovation
5. Climate & Environment

### Statistics

- **Total Articles:** 10,000+
- **Time Range:** 2020-2024
- **Languages:** English (filtered)
- **Labeled Examples:** 500+ manual annotations

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Citation

```bibtex
@article{ghelani2026narrative,
  title={Temporal Contrastive Learning for Generalized Narrative Shift Detection in News Articles},
  author={Ghelani, Meet},
  year={2026}
}
```

---

## 📄 License

MIT License

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the author.

---

## 🗺️ Roadmap

- [x] Approach 1: Baseline windowing
- [x] Approach 2: Group-based segmentation
- [x] Approach 4: Dynamic Ruptures segmentation
- [ ] Approach 5: NER-enhanced tracking (in progress)
- [ ] Approach 3: Adaptive window sizing (2029+)
- [ ] Web interface for narrative exploration
- [ ] Real-time news monitoring system
- [ ] Multi-language support

---

**Repository:** [github.com/meet6868/Narrative-Shift-Detection](https://github.com/meet6868/Narrative-Shift-Detection)
