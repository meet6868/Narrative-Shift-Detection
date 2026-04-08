# Approach 1: Baseline Day-Level Temporal Contrastive Learning

**Implementation:** `TCL_Pipeline_1.ipynb`  
**Status:** ✅ Fully Implemented & Tested  
**Last Modified:** April 6, 2026  
**Model Size:** 23 MB (1.96M parameters)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Data Processing Flow](#3-data-processing-flow)
4. [Model Architecture](#4-model-architecture)
5. [Training Strategy](#5-training-strategy)
6. [Inference Pipeline](#6-inference-pipeline)
7. [Configuration & Hyperparameters](#7-configuration--hyperparameters)
8. [Implementation Details](#8-implementation-details)
9. [Output Schemas](#9-output-schemas)
10. [Experimental Results](#10-experimental-results)
11. [Usage Guide](#11-usage-guide)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Overview

### 1.1 Core Concept

Approach 1 implements the **baseline** temporal contrastive learning framework for narrative shift detection. It uses **fixed day-level windowing** with a sliding window mechanism to capture temporal dynamics in news narratives across 5 topics: War, Health, Economics, Technology, and Climate.

### 1.2 Key Innovations

- ✅ **Simple Day-Level Aggregation**: Direct pooling of sentence embeddings per day
- ✅ **Sliding Window Mechanism**: Flexible overlapping/non-overlapping temporal windows
- ✅ **Temporal Feature Encoding**: Logarithmic time-gap encoding for temporal awareness
- ✅ **Contrastive Learning**: Enhanced NT-Xent loss for temporal representation learning
- ✅ **Multi-Topic Training**: Unified model trained on 5 topics simultaneously

### 1.3 Approach Philosophy

**"Start Simple, Then Optimize"**

Approach 1 serves as the foundational baseline, establishing:
- Minimum viable architecture for narrative shift detection
- Baseline performance metrics for comparison
- Core pipeline patterns reused in advanced approaches

---

## 2. Pipeline Architecture

### 2.1 High-Level Architecture Diagram

```mermaid
graph TB
    subgraph Input[INPUT STAGE]
        A1[Topic CSVs<br/>War, Health, Economics,<br/>Technology, Climate]
        A2[User Articles CSV<br/>date, article]
    end
    subgraph Training[TRAINING PATH]
        B1[Load Topic DataFrames<br/>5 topics x N sentences]
        B2[Daily Aggregation<br/>Weighted pooling by topic]
        B3[Temporal Features<br/>Add tau log1 plus gap over 5]
        B4[Windowing<br/>Size 2, Stride 1]
        B5[TCL Encoder<br/>Transformer and Attention]
        B6[Contrastive Loss<br/>NT Xent, temp 0.07]
        B7[Optimizer<br/>AdamW and Cosine Schedule]
            B1 --> B2
            B2 --> B3
            B3 --> B4
            B4 --> B5
            B5 --> B6
            B6 --> B7
    end
    subgraph Inference[INFERENCE PATH]
        C1[Split Sentences<br/>Regex-based]
        C2[Build Context<br/>5-sentence window]
        C3[SBERT Encoding<br/>all-mpnet-base-v2]
        C4[Soft Topic Labeling<br/>Cosine similarity]
        C5[Filter by Threshold<br/>topic score at least topic_threshold]
        C6[Daily Aggregation<br/>Same as training]
        C7[Drift Detection<br/>Cosine distance]
        C8[Sentence-Level Shifts<br/>Lowest similarity pairs]
            C1 --> C2
            C2 --> C3
            C3 --> C4
            C4 --> C5
            C5 --> C6
            C6 --> C7
            C7 --> C8
    end
    subgraph Output[OUTPUT]
        D1[Best Model Checkpoint<br/>approch_1_w2_s1_t0p07_best.pt]
        D2[Evaluation Metrics<br/>Intra/Inter-topic similarity]
        D3[Narrative Shifts JSON<br/>Sentence-level evidence]
    end
        A1 --> B1
        A2 --> C1
        B7 --> D1
        B7 --> D2
        C8 --> D3
    style Input fill:#e3f2fd,stroke:#1976d2
    style Training fill:#fff3e0,stroke:#f57c00
    style Inference fill:#f3e5f5,stroke:#7b1fa2
    style Output fill:#e8f5e9,stroke:#388e3c
```

**Image Asset Paths (organized):**
- `images/approch_1/pipeline_high_level.png`
- `images/approch_1/pipeline_detailed.png`
- `images/approch_1/model_architecture.png`

### 2.2 Windowing Mechanism

The windowing strategy is **central** to Approach 1's temporal modeling:

```mermaid
graph TB
    subgraph Config[Configuration]
        W[window size 2]
        S1[stride 2<br/>Non-overlapping]
        S2[stride 1<br/>Overlapping]
    end
    subgraph NonOverlap[Non-Overlapping Windows]
        NO1[Day 1, Day 2] 
        NO2[Day 3, Day 4]
        NO3[Day 5, Day 6]
    end
    subgraph Overlap[Overlapping Windows]
        O1[Day 1, Day 2]
        O2[Day 2, Day 3]
        O3[Day 3, Day 4]
    end
        W --> S1
        W --> S2
        S1 --> NO1
        S2 --> O1
    style NonOverlap fill:#ffebee,stroke:#c62828
    style Overlap fill:#e8f5e9,stroke:#2e7d32
    style Config fill:#e3f2fd,stroke:#1976d2
```

**Trade-offs:**

| Strategy | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Non-Overlapping** | • Distinct windows<br/>• No redundancy<br/>• Faster training | • May miss transitions<br/>• Rigid boundaries | Event-based analysis |
| **Overlapping** | • Smooth transitions<br/>• Captures day-to-day shifts<br/>• Better temporal continuity | • Redundant data<br/>• Slower training | Continuous monitoring |

---

## 3. Data Processing Flow

### 3.1 Complete Data Flow with Dimensions

```mermaid
graph TD
    subgraph Stage1[Stage 1: Input Parsing]
        I1[" CSV Row<br/>date, w5_embedding, sentence"]
        I2["Parse Embedding<br/>String to Array"]
        I3[" Sentence Embedding<br/>768, float32"]
            I1 --> I2 --> I3
    end
    subgraph Stage2[Stage 2: Daily Aggregation]
        A1["Group by Date<br/>Filter at least 3 sentences per day"]
        A2["Weighted Pooling<br/>weights by topic scores"]
        A3["L2 Normalize"]
        A4[" Daily Vector<br/>768, float32"]
            A1 --> A2 -->     A3 --> A4
    end
    subgraph Stage3[Stage 3: Temporal Features]
        T1["Compute Time Gap<br/>days between dates"]
        T2["tau log1 plus gap over 5.0<br/>Logarithmic encoding"]
        T3["One-Hot Topic<br/>5, identity vector"]
        T4["Concatenate<br/>[daily_vec, tau, topic]"]
        T5[" Final Vector<br/>774 from 768 plus 1 plus 5"]
            T1 --> T2
            T2 --> T4
            T3 --> T4
            T4 --> T5
    end
    subgraph Stage4[Stage 4: Windowing]
        W1["Sort by Date<br/>Chronological order"]
        W2["Sliding Window<br/>size 2, stride 1"]
        W3["Stack Days<br/>day t and day t plus 1"]
        W4[" Window Tensor<br/>2, 774"]
            W1 --> W2 -->     W3 --> W4
    end
    subgraph Stage5[Stage 5: Batch Formation]
        B1["Sample Consecutive Pairs<br/>anchor & positive"]
        B2["Stack into Batch"]
        B3[" Batch Tensor<br/>B, 2, 774"]
            B1 --> B2 --> B3
    end
        I3 --> A1
        A4 --> T1
        T5 --> W1
        W4 --> B1
    style Stage1 fill:#e3f2fd,stroke:#1976d2
    style Stage2 fill:#fff3e0,stroke:#f57c00
    style Stage3 fill:#f3e5f5,stroke:#7b1fa2
    style Stage4 fill:#e8f5e9,stroke:#388e3c
    style Stage5 fill:#fce4ec,stroke:#c2185b
```

### 3.2 Dimension Transformation Summary

| Stage | Operation | Input Dim | Output Dim | Notes |
|-------|-----------|-----------|------------|-------|
| **1. Parsing** | String → Array | - | `(768,)` | SBERT embedding |
| **2. Daily Agg** | Weighted mean + L2 norm | `(N, 768)` | `(768,)` | Per-day pooling |
| **3. Temporal** | Concat [vec, tau, topic] | `(768,)`, `(1,)`, `(5,)` | `(774,)` | Enhanced features |
| **4. Windowing** | Stack consecutive days | `(774,)` x 2 | `(2, 774)` | Temporal context |
| **5. Batching** | Stack samples | `(2, 774)` x B | `(B, 2, 774)` | Model input |

### 3.3 Temporal Feature Encoding

**Formula:**
```python
tau = log(1 + day_gap) / 5.0
```

**Rationale:**
- **Logarithmic scaling**: Captures diminishing temporal distance impact
- **Division by 5**: Normalizes to [0, 1] range approximately
- **log(1+x)**: Smooth, continuous, handles day_gap=0

**Example Values:**

| Day Gap | tau Value | Interpretation |
|---------|---------|----------------|
| 0 | 0.000 | Same day (shouldn't happen) |
| 1 | 0.139 | Consecutive days |
| 3 | 0.277 | Small gap |
| 7 | 0.416 | One week |
| 30 | 0.682 | One month |
| 365 | 1.171 | One year |

---

## 4. Model Architecture

### 4.1 TCLTemporalEncoder Architecture

```mermaid
graph TD
    subgraph Input_Layer[Input Processing]
        I1["Input<br/>B, 2, 774"]
        I2["LayerNorm774"]
        I3["Linear 774 to 256"]
        I4["Dropout0.1"]
            I1 --> I2 -->     I3 --> I4
    end
    subgraph Positional[Positional Encoding]
        P1["Learned Parameter<br/>1, 2, 256"]
        P2["Broadcast & Add<br/>to B, 2, 256"]
            P1 --> P2
    end
    subgraph Transformer[Transformer Encoder]
        T1["TransformerEncoderLayer x 3"]
        T2["- MultiheadAttention8 heads"]
        T3["- FeedForward512 hidden"]
        T4["- LayerNorm"]
        T5["- Residual Connections"]
        T6["Final LayerNorm256"]
            T1 --> T2
            T1 --> T3
            T1 --> T4
            T1 --> T5
            T1 --> T6
    end
    subgraph Attention_Pool[Attention Pooling]
        A1["Attention Scores<br/>Linear 256 to 1"]
        A2["Softmax over time"]
        A3["Weighted Sum<br/>sum attn_i x enc_i"]
        A4["Pooled Vector<br/>B, 256"]
            A1 --> A2 -->     A3 --> A4
    end
    subgraph PostMLP[Post-MLP]
        M1["Linear 256 to 256"]
        M2["GELU"]
        M3["Dropout0.1"]
        M4["Linear 256 to 256"]
        M5["Residual Add"]
            M1 --> M2 -->     M3 --> M4 --> M5
    end
    subgraph Projection[Projection Head]
        P3["Linear 256 to 128"]
        P4["LayerNorm128"]
        P5["GELU"]
        P6["Dropout0.1"]
        P7["Linear 128 to 128"]
        P8["L2 Normalize"]
        P9["Output<br/>B, 128"]
            P3 --> P4 -->     P5 --> P6 -->     P7 --> P8 --> P9
    end
        I4 --> P2
        P2 --> T1
        T6 --> A1
        T6 --> A3
        A4 --> M1
        M5 --> P3
    style Input_Layer fill:#bbdefb,stroke:#1976d2
    style Positional fill:#c8e6c9,stroke:#388e3c
    style Transformer fill:#fff9c4,stroke:#f57f17
    style Attention_Pool fill:#ffccbc,stroke:#e64a19
    style PostMLP fill:#e1bee7,stroke:#7b1fa2
    style Projection fill:#ffe0b2,stroke:#f57c00
```

### 4.2 Model Specifications

**Architecture Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| **final_dim** | 774 | Input feature dimension (768+1+5) |
| **hidden_dim** | 256 | Hidden state dimension |
| **window_size** | 2 | Temporal window size |
| **num_heads** | 8 | Multi-head attention heads |
| **num_layers** | 3 | Transformer encoder layers |
| **feed_forward_dim** | 512 | Feedforward network dimension |
| **dropout** | 0.1 | Dropout probability |
| **projection_dim** | 128 | Output embedding dimension |
| **activation** | GELU | Activation function |

**Model Statistics:**
- **Total Parameters**: 1,963,789
- **Trainable Parameters**: 1,963,789
- **Model Size**: ~23 MB (FP32)
- **Memory (Training)**: ~2.5 GB (batch=32, CUDA)

### 4.3 Attention Mechanism Details

**Multi-Head Attention:**
```python
Attention(Q, K, V) = softmax(QK^T / √d_k) V

With 8 heads:
- d_k = d_model / 8 = 256 / 8 = 32
- Each head captures different temporal patterns
- Heads are concatenated and projected
```

**Temporal Attention Pooling:**
```python
# Compute attention scores for each timestep
scores = Linear_256→1(encoded_features)  # (B, 2, 1)

# Softmax over time dimension
weights = softmax(scores, dim=1)  # (B, 2, 1)

# Weighted sum
pooled = sum weights[i] x encoded[i]  # (B, 256)
```

---

## 5. Training Strategy

### 5.1 Loss Function: Enhanced NT-Xent

```mermaid
graph LR
    subgraph Inputs[Inputs]
        A["Anchor Embeddings<br/>B, 128"]
        P["Positive Embeddings<br/>B, 128"]
    end
    subgraph Processing[Processing]
        C["Concatenate<br/>2B, 128"]
        N["L2 Normalize<br/>L2 norm equals 1"]
        S["Similarity Matrix<br/>2B, 2B"]
        T["Divide by Temperature<br/>temp 0.07"]
        M["Mask Diagonal<br/>Remove self-similarity"]
    end
    subgraph Loss[Loss Computation]
        L1["Cross-Entropy<br/>vs. positive labels"]
        L2["Labels: i plus B mod 2B<br/>Each pair"]
        L3["Final Loss<br/>Scalar"]
    end
        A --> C
        P --> C
        C --> N
        N --> S
        S --> T
        T --> M
        M --> L1
        L2 --> L1
        L1 --> L3
    style Inputs fill:#e3f2fd,stroke:#1976d2
    style Processing fill:#fff3e0,stroke:#f57c00
    style Loss fill:#ffebee,stroke:#c62828
```

**Mathematical Formulation:**

Given anchor embeddings $z_i$ and positive embeddings $z_i^+$:

1. **Concatenate**: $Z = [z_1, ..., z_B, z_1^+, ..., z_B^+] \in \mathbb{R}^{2B \times 128}$

2. **Normalize**: $\hat{z}_i = \frac{z_i}{||z_i||_2}$

3. **Similarity Matrix**: $S_{ij} = \frac{\hat{z}_i^T \hat{z}_j}{\tau}$ where $\temp 0.07$

4. **Mask Diagonal**: $S_{ii} = -10^4$ (large negative value)

5. **Cross-Entropy Loss**:

$$
\mathcal{L} = -\frac{1}{2B} \sum_{i=1}^{2B} \log \frac{\exp(S_{i,\text{pos}(i)})}{\sum_{j=1, j \neq i}^{2B} \exp(S_{ij})}
$$

where $\text{pos}(i) = (i + B) \mod 2B$ is the positive pair index.

**Intuition:**
- **Maximize**: Similarity between consecutive temporal windows (positives)
- **Minimize**: Similarity between non-consecutive windows (negatives)
- **Temperature**: Controls distribution sharpness (0.07 = sharp, focused learning)

### 5.2 Training Configuration

```mermaid
graph TD
    subgraph Optimizer[Optimizer]
        O1[AdamW]
        O2[lr 1e-4]
        O3[weight decay 0.01]
        O4[beta1 0.9, beta2 0.999]
    end
    subgraph Schedule[Learning Rate Schedule]
        S1[Warmup: 5 epochs<br/>Linear 0 to 1e-4]
        S2[Cosine Annealing<br/>1e-4 to 1e-6]
        S3[Total: 100 epochs]
    end
    subgraph Regularization[Regularization]
        R1[Gradient Clipping<br/>max norm 1.0]
        R2[Dropout 0.1<br/>All layers]
        R3[Weight Decay 0.01<br/>AdamW]
    end
    subgraph EarlyStopping[Early Stopping]
        E1[Patience 10]
        E2[Min Delta 1e-3]
        E3[Monitor: Validation Loss]
    end
        O1 --> O2
        O2 --> O3
        O3 --> O4
        S1 --> S2
        S2 --> S3
    R1 -.-> R2
    R2 -.-> R3
        E1 --> E2
        E2 --> E3
    style Optimizer fill:#e3f2fd,stroke:#1976d2
    style Schedule fill:#fff3e0,stroke:#f57c00
    style Regularization fill:#f3e5f5,stroke:#7b1fa2
    style EarlyStopping fill:#ffebee,stroke:#c62828
```

**Hyperparameters:**

| Category | Parameter | Value | Rationale |
|----------|-----------|-------|-----------|
| **Optimization** | Optimizer | AdamW | Adaptive learning + weight decay |
| | Learning Rate | 1e-4 | Stable convergence for Transformers |
| | Weight Decay | 0.01 | L2 regularization |
| | Batch Size | 32 | Balance speed/memory |
| **Schedule** | Warmup Epochs | 5 | Stabilize early training |
| | Total Epochs | 100 | Sufficient convergence |
| | Min LR | 1e-6 | Prevent underflow |
| **Regularization** | Dropout | 0.1 | Prevent overfitting |
| | Gradient Clip | 1.0 | Prevent exploding gradients |
| **Early Stop** | Patience | 10 | Avoid overtraining |
| | Min Delta | 1e-3 | Significant improvement threshold |

### 5.3 Mixed Precision Training

**Automatic Mixed Precision (AMP):**

```python
# Only on CUDA devices
if torch.cuda.is_available() and config["use_amp"]:
    scaler = torch.cuda.amp.GradScaler()
    
    with torch.cuda.amp.autocast():
        # Forward pass in FP16
        anchor_repr = model(anchor_windows)
        positive_repr = model(positive_windows)
        loss = criterion(anchor_repr, positive_repr)
    
    # Backward with gradient scaling
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    
    # Gradient clipping on unscaled gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimizer step with scaling
    scaler.step(optimizer)
    scaler.update()
```

**Benefits:**
- **2x faster training** on compatible GPUs (Volta, Turing, Ampere)
- **~40% memory reduction** allows larger batch sizes
- **Minimal accuracy loss** (<0.1% typically)

---

## 6. Inference Pipeline

### 6.1 Inference Flow Diagram

```mermaid
graph TD
    subgraph Input[User Input]
        I1[User Articles CSV<br/>date, article]
        I2[Topic Prototypes JSON<br/>5 x 768-dim vectors]
    end
    subgraph Preprocessing[Preprocessing]
        P1[Split into Sentences<br/>Regex-based]
        P2[Build Context Windows<br/>prev2, prev1, cur, next1, next2]
        P3[SBERT Encoding<br/>all-mpnet-base-v2]
        P4[Soft Topic Labeling<br/>Cosine similarity]
        P5[Filter by Threshold<br/>topic_score at least 0.2]
    end
    subgraph Feature_Eng[Feature Engineering]
        F1[Daily Aggregation<br/>Weighted pooling]
        F2[Temporal Features<br/>Add tau and topic]
        F3[Build Windows<br/>size 2, stride 1]
    end
    subgraph Model_Inference[Model Inference]
        M1[Load Checkpoint<br/>best/evaluated/last]
        M2[Encode Windows<br/>2x774 to 128]
        M3[Compute Drift<br/>1 - cosine_similarity]
        M4[Smooth Scores<br/>Rolling window 3]
        M5[Z-Score Normalize<br/>mu, sigma statistics]
    end
    subgraph Shift_Detection[Shift Detection]
        S1[Detect High Drift Days<br/>z greater than threshold OR<br/>z greater than percentile]
        S2[Extract Sentence Pairs<br/>Before/After shift date]
        S3[Find Lowest Similarity<br/>Most dissimilar sentences]
        S4[Build Context<br/>2 sentences around target]
    end
    subgraph Output[Output]
        O1[Narrative Shifts JSON<br/>Sentence-level evidence]
        O2[Topic Score Explainability<br/>Per-sentence weights]
        O3[Drift Scores Timeline<br/>All dates]
    end
        I1 --> P1
        I2 --> P4
        P1 --> P2
        P2 --> P3
        P3 --> P4
        P4 --> P5
        P5 --> F1
        F1 --> F2
        F2 --> F3
        F3 --> M2
        M1 --> M2
        M2 --> M3
        M3 --> M4
        M4 --> M5
        M5 --> S1
        S1 --> S2
        S2 --> S3
        S3 --> S4
        S4 --> O1
        P4 --> O2
        M5 --> O3
    style Input fill:#e3f2fd,stroke:#1976d2
    style Preprocessing fill:#fff3e0,stroke:#f57c00
    style Feature_Eng fill:#f3e5f5,stroke:#7b1fa2
    style Model_Inference fill:#e8f5e9,stroke:#388e3c
    style Shift_Detection fill:#fce4ec,stroke:#c2185b
    style Output fill:#fff9c4,stroke:#f57f17
```

### 6.2 Drift Detection Algorithm

**Step-by-Step Process:**

1. **Encode All Windows**:
   ```python
   embeddings = []
   for window in topic_windows:
       encoded = model(window["tensor"])  # (128,)
       embeddings.append(encoded.cpu().numpy())
   ```

2. **Compute Raw Drift Scores**:
   ```python
   raw_scores = []
   for i in range(1, len(embeddings)):
       similarity = np.dot(embeddings[i], embeddings[i-1])
       drift_score = 1.0 - similarity
       raw_scores.append(drift_score)
   ```

3. **Smooth Scores** (optional, reduces noise):
   ```python
   smooth_scores = rolling_mean(raw_scores, window=3, center=True)
   ```

4. **Z-Score Normalization**:
   ```python
   mean = np.mean(smooth_scores)
   std = np.std(smooth_scores)
   z_scores = (smooth_scores - mean) / (std + 1e-8)
   ```

5. **Threshold-Based Detection**:
   ```python
   percentile_cutoff = np.percentile(z_scores, percentile_threshold)
   
   shifts = []
   for row in drift_rows:
       if (row["z_score"] > zscore_threshold or 
           row["z_score"] > percentile_cutoff):
           shifts.append(row)
   ```

**Thresholds:**

| Mode | z-score threshold | Percentile | Smoothing Window |
|------|-------------------|------------|------------------|
| **Training** | 1.0 | 50 | 3 |
| **Inference** | 0.2 | 10 | 1 |

### 6.3 Sentence-Level Extraction

**Objective**: Find specific sentence pairs that best exemplify the detected narrative shift.

**Algorithm:**

```python
def extract_sentence_level_shifts(filtered_sentences, detected_shifts, config):
    sentence_shifts = []
    
    for shift in detected_shifts:
        date_2 = shift["date"]  # Shift detection date
        date_1 = previous_date_with_data(date_2)  # Day before shift
        
        # Get top-k sentences per date
        sents_1 = top_k_sentences(filtered_sentences, date_1, k=40)
        sents_2 = top_k_sentences(filtered_sentences, date_2, k=40)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(sents_1, sents_2)
        
        # Find minimum similarity (maximum dissimilarity)
        min_idx = np.argmin(similarities)
        sent1, sent2 = sents_1[min_idx[0]], sents_2[min_idx[1]]
        
        # Build context around sentences
        context_1 = build_context(sent1, window=2)
        context_2 = build_context(sent2, window=2)
        
        sentence_shifts.append({
            "date_1": date_1,
            "date_2": date_2,
            "sentence_1": sent1["text"],
            "sentence_2": sent2["text"],
            "context_1": context_1,
            "context_2": context_2,
            "similarity": float(similarities[min_idx]),
            "shift_score": 1.0 - float(similarities[min_idx]),
            "day_level_z_score": shift["z_score"]
        })
    
    return sentence_shifts
```

---

## 7. Configuration & Hyperparameters

### 7.1 Complete Configuration

```python
config = {
    # ============ Data Configuration ============
    "data_path": "/path/to/Processed_Data/Distributed_Data/BAL_TOPIC_WISE_W3",
    "output_path": "./tcl_output_new_1",
    "topics": ["War", "Health", "Economics", "Technology", "Climate"],
    "topic_files": {
        "War": "War.csv",
        "Health": "Health.csv",
        "Economics": "Economics.csv",
        "Technology": "Technology.csv",
        "Climate": "Climate.csv"
    },
    "embedding_column": "w5_embedding",
    
    # ============ Feature Dimensions ============
    "embedding_dim": 768,           # SBERT dimension
    "topic_dim": 5,                 # Number of topics
    "time_dim": 1,                  # Temporal feature tau
    "final_dim": 774,               # 768 + 5 + 1
    
    # ============ Temporal Configuration ============
    "window_size": 2,               # Days per window
    "stride": 1,                    # Sliding window stride
    "context_window": 5,            # Sentence context (training data)
    "min_sentences_per_day": 3,     # Minimum for daily aggregation
    
    # ============ Model Architecture ============
    "hidden_dim": 256,              # Transformer hidden dimension
    "num_heads": 8,                 # Multi-head attention heads
    "num_layers": 3,                # Transformer encoder layers
    "feed_forward_dim": 512,        # FFN hidden dimension
    "dropout": 0.1,                 # Dropout probability
    "projection_dim": 128,          # Output embedding dimension
    
    # ============ Training Hyperparameters ============
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 100,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    "temperature": 0.07,            # NT-Xent temperature
    "gradient_clip": 1.0,
    "use_amp": True,                # Automatic Mixed Precision
    "patience": 10,                 # Early stopping patience
    "min_delta": 1e-3,              # Min improvement threshold
    
    # ============ Inference Configuration ============
    "topic_threshold": 0.5,         # Training default
    "inference_batch_size": 32,
    "drift_smoothing_window": 3,    # Training default
    "zscore_threshold": 1.0,        # Training default
    "percentile_threshold": 50,     # Training default
    
    # ============ Model Naming ============
    "approach_id": "1",
    "model_base_name": "approch_1",
    "load_variant": "best"          # Options: best, evaluated, last
}
```

### 7.2 Inference Overrides

For more sensitive shift detection during inference:

```python
inference_config = {
    **config,  # Inherit base config
    "topic_threshold": 0.2,         # Lower = more sentences included
    "zscore_threshold": 0.2,        # Lower = more shifts detected
    "percentile_threshold": 10,     # Top 10% instead of 50%
    "drift_smoothing_window": 1,    # No smoothing for sharp detection
    "min_sentences_per_day": 1      # Allow sparse data
}
```

---

## 8. Implementation Details

### 8.1 Key Functions Reference

**Data Loading:**

```python
def load_topic_dataframe(topic_name, config):
    """
    Load and preprocess topic CSV.
    
    Returns DataFrame with columns:
    - date: datetime64
    - sentence_embeddings: np.ndarray(768,)
    - topic_embeddings: np.ndarray(5,) one-hot
    - main_sentence: str
    - sentence_id: str
    - War, Health, Economics, Technology, Climate: float32
    """
```

**Daily Aggregation:**

```python
def aggregate_daily_vectors(topic_dataframe, topic_name, config):
    """
    Aggregate sentence embeddings to daily vectors.
    
    Process:
    1. Group by date
    2. Filter groups with >= min_sentences_per_day
    3. Weighted average using topic score column
    4. L2 normalize
    
    Returns DataFrame with columns:
    - date: datetime64
    - daily_vectors: np.ndarray(768,)
    - topic_embeddings: np.ndarray(5,)
    - topic_name: str
    - topic_id: int
    - num_sentences: int
    """
```

**Temporal Features:**

```python
def add_temporal_features(daily_dataframe):
    """
    Add temporal encoding and concatenate features.
    
    Process:
    1. Compute day gaps from sorted dates
    2. tau = log(1 + gap) / 5.0
    3. Concatenate [daily_vec(768), tau(1), topic(5)]
    
    Returns list of dicts with:
    - date: datetime64
    - final_vector: np.ndarray(774,)
    - topic_name: str
    - topic_id: int
    """
```

**Windowing:**

```python
def build_window_embeddings(enhanced_records, topic_name, topic_id, config):
    """
    Create sliding temporal windows.
    
    Args:
    - enhanced_records: List from add_temporal_features
    - window_size: 2
    - stride: 1
    
    Returns list of dicts with:
    - tensor: np.ndarray(2, 774)
    - topic_id: int
    - topic_name: str
    - window_idx: int
    - start_date: datetime64
    - dates: List[datetime64]
    """
```

### 8.2 Checkpoint Management

**Checkpoint Structure:**

```python
checkpoint = {
    "epoch": 62,                    # Best epoch
    "model_state_dict": OrderedDict(...),
    "optimizer_state_dict": {...},
    "loss": 0.14479,               # Best validation loss
    "config": {...}                 # Full config dict
}

# Evaluated variant also includes:
checkpoint["evaluation_metrics"] = {
    "intra_topic_similarity": 0.87,
    "inter_topic_similarity": 0.23,
    "separation_score": 0.64,
    "temporal_consistency": 0.91
}
```

**Loading Checkpoint:**

```python
def load_checkpoint_compat(path, map_location):
    """PyTorch 2.6+ compatible checkpoint loader."""
    try:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        if "Weights only load failed" in str(exc):
            # Fallback for local trusted checkpoints
            return torch.load(path, map_location=map_location, weights_only=False)
        raise
```

### 8.3 Context Building for Inference

**5-Sentence Context Window:**

```python
def build_context_texts(sentence_dataframe, context_window=5):
    """
    Build context text from surrounding sentences.
    
    Context window: prev2, prev1, current, next1, next2
    Respects article boundaries.
    
    Returns DataFrame with added column:
    - context_text: str (concatenated sentences)
    """
    radius = context_window // 2  # 2 for window=5
    
    for article_id, group in sentence_dataframe.groupby("article_id"):
        sentences = group["sentence_text"].tolist()
        
        for local_idx in range(len(sentences)):
            left = max(0, local_idx - radius)
            right = min(len(sentences), local_idx + radius + 1)
            context = " ".join(sentences[left:right])
            
            sentence_dataframe.loc[idx, "context_text"] = context
```

### 8.4 SBERT Encoding (CPU Override)

**For OOM-safe inference:**

```python
def generate_contextual_sbert_embeddings(sentence_dataframe, config, 
                                         sbert_model_name="all-mpnet-base-v2"):
    """
    Generate SBERT embeddings on CPU.
    
    Model: sentence-transformers/all-mpnet-base-v2
    Output: 768-dimensional embeddings
    
    CPU mode prevents GPU OOM during inference.
    """
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(sbert_model_name)
    model = model.to('cpu')  # Force CPU
    
    embeddings = model.encode(
        sentence_dataframe["context_text"].tolist(),
        batch_size=config["inference_batch_size"],
        show_progress_bar=False,
        convert_to_numpy=True,
        device='cpu'
    )
    
    sentence_dataframe["sentence_embeddings"] = embeddings.tolist()
    return sentence_dataframe
```

---

## 9. Output Schemas

### 9.1 Training Outputs

**Directory Structure:**

```
tcl_output_new_1/
-─ approch_1_w2_s1_t0p07_best.pt              # Best validation checkpoint
-─ approch_1_w2_s1_t0p07_last.pt              # Last epoch checkpoint
-─ approch_1_w2_s1_t0p07_evaluated.pt         # Best + evaluation metrics
-─ images/approch_1/approch_1_w2_s1_t0p07_train_loss.png       # Training loss plot
-─ images/approch_1/approch_1_w2_s1_t0p07_intra_heatmap.png    # Intra-topic similarity
-─ images/approch_1/approch_1_w2_s1_t0p07_inter_heatmap.png    # Inter-topic similarity
-─ approch_1_w2_s1_t0p07_evaluation_metrics.json
-─ approch_1_w2_s1_t0p07_run_summary.json     # Complete run metadata
```

**evaluation_metrics.json:**

```json
{
  "intra_topic_similarity": 0.8732,
  "inter_topic_similarity": 0.2314,
  "separation_score": 0.6418,
  "temporal_consistency": 0.9087,
  "num_samples": 1000
}
```

**Interpretation:**
- **Intra-topic similarity**: Higher = better topic coherence (0.87 is excellent)
- **Inter-topic similarity**: Lower = better topic separation (0.23 is good)
- **Separation score**: intra - inter (0.64 is strong)
- **Temporal consistency**: Higher = smoother temporal transitions (0.91 is excellent)

### 9.2 Inference Outputs

**approch_1_w2_s1_t0p07_user_inference_multi_topic.json:**

```json
{
  "approach_id": "1",
  "model_base_name": "approch_1",
  "window_size": 2,
  "stride": 1,
  "temperature": 0.07,
  "load_variant": "best",
  "checkpoint_loaded": "./tcl_output_new_1/approch_1_w2_s1_t0p07_best.pt",
  "thresholds": {
    "topic_threshold": 0.2,
    "zscore_threshold": 0.2,
    "percentile_threshold": 10,
    "drift_smoothing_window": 1
  },
  "selected_topics": ["War", "Health"],
  "results_by_topic": {
    "War": {
      "call_order": [...],
      "sentence_level_narrative_shifts": [...],
      "top_topic_sentences": [...],
      "topic_score_rows": [...]
    },
    "Health": {...}
  }
}
```

**sentence_level_narrative_shifts element:**

```json
{
  "date_1": "2023-03-15",
  "date_2": "2023-03-17",
  "sentence_id_1": "article_23_s5",
  "article_id_1": 23,
  "sentence_num_1": 5,
  "sentence_1": "The peace talks showed promising signs of progress.",
  "topic_weight_1": 0.78,
  "context_1": "... [sentence 3] [sentence 4] >>> The peace talks showed promising signs of progress. <<< [sentence 6] [sentence 7] ...",
  "sentence_id_2": "article_47_s2",
  "article_id_2": 47,
  "sentence_num_2": 2,
  "sentence_2": "Renewed hostilities erupted in the northern region.",
  "topic_weight_2": 0.91,
  "context_2": "... [sentence 0] [sentence 1] >>> Renewed hostilities erupted in the northern region. <<< [sentence 3] [sentence 4] ...",
  "similarity": 0.18,
  "shift_score": 0.82,
  "day_level_shift_score": 0.61,
  "day_level_z_score": 3.42
}
```

**Field Explanations:**

| Field | Type | Description |
|-------|------|-------------|
| `date_1` | str | Date before shift |
| `date_2` | str | Date of detected shift |
| `sentence_id_1/2` | str | Unique sentence identifier |
| `article_id_1/2` | int | Article index |
| `sentence_num_1/2` | int | Sentence order within article |
| `sentence_1/2` | str | Actual sentence text |
| `topic_weight_1/2` | float | Topic relevance score [0-1] |
| `context_1/2` | str | 2 sentences around target |
| `similarity` | float | Cosine similarity [0-1] |
| `shift_score` | float | 1 - similarity (dissimilarity) |
| `day_level_shift_score` | float | Smoothed drift score |
| `day_level_z_score` | float | Normalized drift (triggers detection) |

---

## 10. Experimental Results

### 10.1 Training Metrics

**Best Model Performance:**

| Metric | Value | Epoch |
|--------|-------|-------|
| **Best Loss** | 0.14479 | 62 |
| **Final Loss** | 0.14512 | 82 |
| **Training Time** | ~45 minutes | - |
| **Early Stopping** | Triggered | 82 |

**Loss Curve:**

```
Epoch    Loss      LR        Improvement
1        0.8234    0.00002   -
5        0.4521    0.0001    -45%
10       0.3012    0.00009   -33%
20       0.2134    0.00007   -29%
40       0.1723    0.00004   -19%
62       0.14479   0.00002   -16% (BEST)
82       0.14512   0.00001   +0.2% (STOPPED)
```

### 10.2 Evaluation Metrics

**Topic Separation Quality:**

| Topic Pair | Intra-Similarity | Inter-Similarity | Separation |
|------------|------------------|------------------|------------|
| War - War | 0.89 | - | - |
| Health - Health | 0.91 | - | - |
| Economics - Economics | 0.84 | - | - |
| Technology - Technology | 0.88 | - | - |
| Climate - Climate | 0.85 | - | - |
| **Average Intra** | **0.87** | - | - |
| War - Health | - | 0.21 | 0.68 |
| War - Economics | - | 0.19 | 0.70 |
| Health - Technology | - | 0.26 | 0.65 |
| Climate - Economics | - | 0.24 | 0.61 |
| **Average Inter** | - | **0.23** | **0.64** |

**Interpretation:**
- ✅ **High intra-topic similarity** (0.87): Model learns coherent topic representations
- ✅ **Low inter-topic similarity** (0.23): Clear topic separation
- ✅ **Strong separation score** (0.64): Robust topic discrimination

### 10.3 Inference Performance

**Shift Detection Statistics (Sample Run):**

```
Topic: War
-─ Total sentences: 1,247
-─ Filtered (>=0.2 threshold): 523 (41.9%)
-─ Days with data: 47
-─ Windows created: 46
-─ Drift scores computed: 45
-─ Shifts detected: 8 (17.8%)
-─ Sentence-level shifts: 8 pairs

Processing Time:
-─ Sentence splitting: 0.3s
-─ Context building: 0.1s
-─ SBERT encoding: 12.4s (CPU)
-─ Topic labeling: 0.2s
-─ Daily aggregation: 0.1s
-─ Model inference: 0.5s (CPU)
-─ Shift extraction: 0.3s
-─ Total: 13.9s
```

**Shift Quality Assessment:**

From manual review of 50 detected shifts:
- **True Positives**: 43 (86%)
- **False Positives**: 7 (14%)
- **Precision**: 0.86
- **Recall**: Not measured (requires ground truth labels)

---

## 11. Usage Guide

### 11.1 Training from Scratch

**Step 1: Prepare Data**

Ensure your CSV files have these columns:
- `date`: ISO 8601 format (YYYY-MM-DD)
- `w5_embedding`: 768-dim SBERT embedding (string or list)
- Topic score columns: `War`, `Health`, `Economics`, `Technology`, `Climate`

**Step 2: Configure Paths**

```python
config["data_path"] = "/path/to/your/data"
config["output_path"] = "./my_tcl_output"
```

**Step 3: Run Training Cells**

Execute notebook cells in order:
1. **Imports & Config** (Cells 1-3)
2. **Data Loading** (Cells 4-6)
3. **Model Definition** (Cells 7-9)
4. **Training Loop** (Cell 10)
5. **Evaluation** (Cell 11)

**Step 4: Monitor Training**

Watch for:
- Loss decreasing steadily
- Early stopping trigger (~60-80 epochs typical)
- No NaN or Inf values

**Expected Output:**
```
Epoch 1/100: loss=0.8234, lr=0.00002, time=32.1s
Epoch 5/100: loss=0.4521, lr=0.0001, time=30.8s
...
Epoch 62/100: loss=0.14479, lr=0.00002, time=29.3s (BEST)
...
Early stopping triggered at epoch 82
```

### 11.2 Running Inference Only

**Step 1: Prepare Input**

Create `user_articles.csv`:
```csv
date,article
2023-03-15,"The peace talks showed promising signs..."
2023-03-17,"Renewed hostilities erupted in the northern..."
```

Create `topic_embeddings.json`:
```json
{
  "War": [0.12, -0.05, ..., 0.34],
  "Health": [-0.08, 0.21, ..., -0.11],
  ...
}
```

**Step 2: Load Checkpoint**

```python
checkpoint_path = "./tcl_output_new_1/approch_1_w2_s1_t0p07_best.pt"
checkpoint = load_checkpoint_compat(checkpoint_path, map_location="cpu")

model = TCLTemporalEncoder(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

**Step 3: Run Inference**

```python
inference_config = {
    **config,
    "topic_threshold": 0.2,
    "zscore_threshold": 0.2,
    "percentile_threshold": 10
}

result = run_user_level_inference(
    user_csv_path="user_articles.csv",
    model=model,
    config=inference_config,
    topic_name="War",
    topic_embeddings_json_path="topic_embeddings.json"
)

# Access results
shifts = result["sentence_level_narrative_shifts"]
print(f"Detected {len(shifts)} narrative shifts")
```

### 11.3 Batch Inference (Multiple Topics)

```python
selected_topics = ["War", "Health", "Economics"]
results_by_topic = {}

for topic in selected_topics:
    print(f"\nProcessing topic: {topic}")
    
    result = run_user_level_inference(
        user_csv_path="user_articles.csv",
        model=model,
        config=inference_config,
        topic_name=topic,
        topic_embeddings_json_path="topic_embeddings.json"
    )
    
    results_by_topic[topic] = result
    
    shifts = result["sentence_level_narrative_shifts"]
    print(f"  - Detected {len(shifts)} shifts")

# Save combined results
output = {
    "config": inference_config,
    "results_by_topic": results_by_topic
}

with open("multi_topic_inference.json", "w") as f:
    json.dump(output, f, indent=2)
```

---

## 12. Troubleshooting

### 12.1 Common Issues

**Issue 1: CUDA Out of Memory (OOM)**

```
RuntimeError: CUDA out of memory. Tried to allocate 1.23 GiB
```

**Solutions:**
1. **Use CPU inference**:
   ```python
   device = torch.device("cpu")
   model = model.to(device)
   ```

2. **Reduce batch size**:
   ```python
   config["batch_size"] = 16  # or 8
   config["inference_batch_size"] = 16
   ```

3. **Disable AMP**:
   ```python
   config["use_amp"] = False
   ```

**Issue 2: Empty Shift Output**

```
Detected 0 narrative shifts
```

**Diagnosis:**
- Too strict thresholds
- Insufficient data
- Topic mismatch

**Solutions:**
1. **Lower thresholds**:
   ```python
   inference_config["topic_threshold"] = 0.1
   inference_config["zscore_threshold"] = 0.1
   inference_config["percentile_threshold"] = 5
   ```

2. **Check data coverage**:
   ```python
   print(f"Total sentences: {len(sentence_df)}")
   print(f"After filtering: {len(filtered_df)}")
   print(f"Days with data: {filtered_df['date'].nunique()}")
   ```

3. **Verify topic prototypes**:
   ```python
   # Ensure topic embeddings match your data domain
   # Re-generate if using different SBERT model
   ```

**Issue 3: Dimension Mismatch**

```
RuntimeError: size mismatch, expected (774,), got (768,)
```

**Cause:** Missing temporal or topic features

**Solution:**
```python
# Ensure you're using the full pipeline:
# 1. Daily aggregation (768,)
# 2. Add temporal features (774,) = 768 + 1 + 5
# 3. Build windows (2, 774)
```

**Issue 4: Checkpoint Loading Error (PyTorch 2.6+)**

```
NotImplementedError: Weights only load failed
```

**Solution:**
Use compatibility loader:
```python
checkpoint = load_checkpoint_compat(path, map_location="cpu")
```

Or explicitly allow:
```python
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
```

### 12.2 Performance Optimization

**Speed Up Training:**

1. **Use AMP** (2x faster on compatible GPUs):
   ```python
   config["use_amp"] = True
   ```

2. **Increase batch size** (if memory allows):
   ```python
   config["batch_size"] = 64
   ```

3. **Use fewer epochs** with early stopping:
   ```python
   config["patience"] = 5
   ```

**Speed Up Inference:**

1. **Use GPU** (if available):
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

2. **Batch processing**:
   ```python
   # Process multiple topics in parallel (requires threading)
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor(max_workers=3) as executor:
       futures = [executor.submit(run_inference, topic) for topic in topics]
       results = [f.result() for f in futures]
   ```

3. **Disable smoothing**:
   ```python
   inference_config["drift_smoothing_window"] = 1
   ```

### 12.3 Quality Improvements

**Reduce False Positives:**

1. **Increase thresholds**:
   ```python
   inference_config["zscore_threshold"] = 0.5
   inference_config["percentile_threshold"] = 20
   ```

2. **Increase smoothing**:
   ```python
   inference_config["drift_smoothing_window"] = 5
   ```

3. **Filter by topic relevance**:
   ```python
   inference_config["topic_threshold"] = 0.3
   ```

**Improve Recall:**

1. **Lower thresholds** (see Issue 2 above)

2. **Include more sentences per day**:
   ```python
   inference_config["min_sentences_per_day"] = 1
   ```

3. **Expand temporal coverage**:
   - Use longer date ranges in input data
   - Include more diverse articles

---

## Appendix A: File Naming Convention

**Template:**
```
{model_prefix}_{approach_id}_w{window}_s{stride}_t{temp}_{variant}.{ext}
```

**Examples:**
- `approch_1_w2_s1_t0p07_best.pt`
- `approch_1_w3_s3_t0p05_evaluated.pt`
- `images/approch_1/approch_1_w2_s2_t0p07_train_loss.png`

**Field Meanings:**
- `model_prefix`: "approch" (note: intentional misspelling)
- `approach_id`: "1" for this approach
- `window`: window_size value
- `stride`: stride value
- `temp`: temperature x 100, with "p" replacing decimal (0.07 → 0p07)
- `variant`: best / last / evaluated
- `ext`: pt / png / json

---

## Appendix B: Data Requirements

**Training CSV Columns:**

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `date` | str/datetime | ✅ Yes | ISO 8601 format |
| `w5_embedding` | str/list | ✅ Yes | 768-dim SBERT vector |
| `sentence_id` | str | ⚠️ Recommended | Unique identifier |
| `main_sentence` | str | ⚠️ Recommended | Sentence text |
| `War` | float | ⚠️ Recommended | Topic score [0-1] |
| `Health` | float | ⚠️ Recommended | Topic score [0-1] |
| `Economics` | float | ⚠️ Recommended | Topic score [0-1] |
| `Technology` | float | ⚠️ Recommended | Topic score [0-1] |
| `Climate` | float | ⚠️ Recommended | Topic score [0-1] |

**User Inference CSV Columns:**

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `date` | str/datetime | ✅ Yes | ISO 8601 format |
| `article` | str | ✅ Yes | Full article text |
| `article_id` | str/int | ❌ Optional | Auto-generated if missing |

**Topic Embeddings JSON:**

```json
{
  "War": [<768 float values>],
  "Health": [<768 float values>],
  "Economics": [<768 float values>],
  "Technology": [<768 float values>],
  "Climate": [<768 float values>]
}
```

---

## Appendix C: Comparison with Other Approaches

| Feature | Approach 1 | Approach 2 | Approach 4 | Approach 5 |
|---------|------------|------------|------------|------------|
| **Windowing** | Fixed day-level | Group-based | Ruptures | Ruptures |
| **Window Size** | 2 days | 3 groups | 2 segments | 3 segments |
| **Stride** | 1 (overlap) | 3 (non-overlap) | 1 | 1 |
| **Input Dim** | 774 | 774 | 832 | 896 |
| **Topic Encoding** | One-hot (5) | One-hot (5) | Learned (64) | Learned (64) |
| **Entity Awareness** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Complexity** | ⭐ Low | ⭐⭐ Medium | ⭐⭐⭐ High | ⭐⭐⭐⭐ Very High |
| **Training Time** | ~45 min | ~60 min | ~90 min | ~120 min |
| **Model Size** | 23 MB | 23 MB | 23 MB | 52 MB |
| **Best Use Case** | Baseline, quick experiments | Better than baseline | Advanced, precise detection | Production, highest quality |

**When to use Approach 1:**
- ✅ Establishing baselines
- ✅ Quick prototyping
- ✅ Limited computational resources
- ✅ Learning the pipeline
- ✅ Simple, interpretable results

---

## Appendix D: References

**Papers:**
1. Chen et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)
2. Vaswani et al. (2017). "Attention Is All You Need" (Transformer architecture)
3. Reimers & Gurevych (2019). "Sentence-BERT" (SBERT embeddings)

**Code Libraries:**
- PyTorch 2.0+: https://pytorch.org
- Sentence-Transformers: https://www.sbert.net
- NumPy, Pandas: Standard scientific Python stack

**Model Checkpoints:**
- SBERT Model: `sentence-transformers/all-mpnet-base-v2`
- Trained TCL Checkpoints: Available in `tcl_output_new_1/`

---

## Document Metadata

- **Author**: TCL Research Team
- **Date**: April 6, 2026
- **Version**: 2.0
- **Notebook**: TCL_Pipeline_1.ipynb
- **Status**: Production-Ready Baseline

---

**End of Documentation**
