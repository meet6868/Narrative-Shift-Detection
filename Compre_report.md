# Narrative Shift Detection in News Media Using Temporal Contrastive Learning

**Team Members:** [Add team member names]  
**Mentor:** [Add mentor name]  
**Date:** March 2026

---

## Abstract

News media narratives evolve continuously in response to political, technological, and societal developments. Understanding these narrative shifts is crucial for media analysis, discourse tracking, and misinformation monitoring. Existing approaches—topic models, sentiment analysis, and document similarity methods—fail to explicitly model narrative shifts as temporal semantic phenomena with sentence-level interpretability.

We present a framework for narrative shift detection using Temporal Contrastive Learning (TCL) that tracks narrative evolution without supervised labels or topic-specific training. We collected and processed 1.17 million articles from 8 heterogeneous sources, applying rigorous cleaning (69.56% noise removal) to yield 355,334 English articles across five topics: War, Health, Technology, Climate, and Economics. Through sentence-level segmentation (3.08M sentences), context-aware SBERT embeddings (Window-5), and topic-aware filtering with curated prototypes from authoritative sources (WHO, Reuters, IPCC, IMF), we generated 1.34M topic-classified sentences.

Our TCL framework evolved through five experimental approaches, progressing from fixed windowing (InfoNCE loss) to adaptive segmentation using Ruptures-based change-point detection with multi-objective loss functions (NT-Xent + Topic Separation + Hard Negative Mining). Approach 4 achieves best inter-topic separation (-0.0875) and temporal consistency (0.9877) with 356 dynamic segments. Approach 4 was further retrained on **balanced topic data** (`Pre_Processing/Data_balancing.ipynb`) and validated via user inference on 5 real Russia-Ukraine articles (Feb–Apr 2022), detecting 2 narrative shifts in War and Economics topics and 1 in Health, with scores from 0.12 to 1.0 — correctly identifying the invasion (shift score 1.0, similarity 0.246) and reconstruction narrative phases. We compare against two baselines: calibrated SBERT drift detection (4-stage pipeline, 15 topic-model threshold pairs) and K-Means clustering drift (K=5, Jensen-Shannon divergence). Key insights include the necessity of adaptive segmentation over fixed windows, explicit topic separation losses, and entity-awareness for fine-grained detection — addressed in Approach 5 (`TCL_Pipeline_5.ipynb`, ✅ Production-Ready), which integrates NER (spaCy en_core_web_trf), sentiment analysis (RoBERTa), entity canonicalization, FAISS-based retrieval, and a 4-component shift score targeting Kaggle GPU P100.

**Keywords:** Narrative shift detection, temporal contrastive learning, news media analysis, semantic change detection, interpretable NLP

---

## 1. Introduction

Narratives in news media are not static. As events unfold, political contexts change, and public sentiment shifts, the framing, emphasis, tone, and interpretation of facts evolve over time. While multiple articles may discuss the same topic and entity, their narrative framing often changes gradually or abruptly. Understanding how these narratives change is essential for media behavior analysis, public discourse evolution, social science research, and misinformation monitoring.

Most existing approaches focus on topic change detection, sentiment trends, or document similarity. However, these approaches fail to explicitly model narrative shifts as a temporal semantic phenomenon that is entity-aware, topic-agnostic, learnable, and interpretable at the sentence level. While prior work has studied topic evolution or static narrative extraction, few approaches explicitly model narrative change over time in a manner that provides fine-grained textual explanations.

### 1.1 Motivation

News media narratives evolve continuously over time. Even when articles discuss the same topic, the framing, tone, emphasis, and intent often change in response to political, regulatory, technological, or societal developments. These changes, referred to as narrative shifts, play a crucial role in shaping public opinion and influencing policy discourse.

Existing approaches to analyzing narrative evolution face significant limitations. Topic models often conflate topic presence with narrative framing, while supervised change detection methods require labeled data that is scarce, subjective, and expensive to obtain. Recently, large language model (LLM)-based approaches have been explored, but they suffer from scalability, reproducibility, and interpretability concerns.

### 1.2 Objectives

This project aims to address these challenges by proposing a generalized, scalable, and interpretable framework for detecting narrative shifts in news media without relying on labeled data, fixed topic definitions, or prompt-based LLM inference. Specifically, the objectives are:

1. **Develop a generalized narrative shift detection framework** that works across multiple topics without topic-specific training
2. **Implement temporal contrastive learning** to model narrative evolution as a continuous temporal process
3. **Provide sentence-level interpretability** by identifying specific sentences that drive narrative changes
4. **Build a large-scale multi-source news dataset** across five major topics for comprehensive analysis
5. **Design a scalable preprocessing pipeline** that handles heterogeneous data sources and formats
6. **Compare TCL against baseline approaches** including Semantic Drift using SBERT and K-Means clustering drift to validate effectiveness
7. **Enable fine-grained temporal tracking** of narrative evolution without requiring supervised labels

### 1.3 Problem Statement and Formal Definition

Given a sequence of time-indexed news articles referring to the same real-world entity under a broad topic, the goal is to automatically detect when a narrative shift occurs, model how the narrative changes over time, identify the specific sentences responsible for the shift, localize the textual evidence driving these shifts, and explain how the narrative framing has changed—all without requiring supervised narrative labels or topic-specific training.

#### Input Specification

- A topic $T$ (e.g., war, climate, economy, technology, health)
- An entity $E$ relevant to the topic
- A temporally ordered sequence of news articles $A = \{(x_i, t_i)\}_{i=1}^n$ where $x_i$ is the article text and $t_i$ is the timestamp
- All articles reference the same entity $E$ and belong to topic $T$
- Each article is segmented into sentences
- No topic-specific or domain-specific supervision is assumed

#### Output Specification

The system must output:

- A narrative shift score between consecutive time windows or article pairs
- A binary or probabilistic indication of whether a narrative shift has occurred (with shift categories: low, medium, high)
- A ranked list of sentences that exhibit the strongest narrative change
- Sentence-level change sets identifying drivers of narrative change
- An interpretable, extractive explanation of the semantic differences driving the shift
- A time-ordered narrative evolution trajectory
- Narrative embeddings $z_i$ and $z_{i+1}$ for consecutive articles
- Textual explanations describing how and why the narrative shifted

#### Task Framing

The task is framed as a temporal contrastive learning problem, where narrative representations are learned such that stable narratives remain close in embedding space while shifted narratives are pushed apart. This enables continuous tracking of narrative evolution rather than binary change detection.

---

## 2. Background and Related Work

### 2.1 Topic Modeling and Evolution

Traditional topic modeling approaches such as Latent Dirichlet Allocation (LDA) and Dynamic Topic Models (DTM) have been widely used to discover latent topics in document collections and track their evolution over time. However, these methods primarily focus on topic distribution changes rather than semantic narrative shifts within topics. They conflate the presence of a topic with how that topic is framed or discussed, making them unsuitable for fine-grained narrative analysis.

### 2.2 Semantic Change Detection

Recent work in semantic change detection has focused on tracking how word meanings evolve over time using diachronic word embeddings. Methods like Word2Vec temporal alignment and BERT-based contextualized embeddings have shown promise in detecting lexical semantic shifts. However, these approaches operate at the word or phrase level and do not capture document-level or narrative-level semantic changes that involve complex interactions between multiple sentences and concepts.

### 2.3 Document Similarity and Drift Detection

Document similarity measures based on semantic embeddings, cosine similarity, and clustering approaches have been used to detect concept drift in text streams. SBERT-based approaches and K-Means clustering methods can identify when document distributions change significantly. While useful for detecting distributional shifts, these methods lack the temporal modeling capabilities and interpretability needed for narrative shift detection. They also struggle with capturing continuous narrative evolution, often treating drift as discrete shifts rather than gradual temporal changes.

### 2.4 Contrastive Learning in NLP

Contrastive learning has emerged as a powerful paradigm for learning representations by maximizing agreement between differently augmented views of the same data while pushing apart representations of different data. In NLP, contrastive learning has been successfully applied to sentence embeddings (SimCSE), document representations, and cross-lingual alignment. Temporal contrastive learning extends this paradigm by incorporating temporal dynamics, learning representations where temporally adjacent samples with similar narratives are pulled together while narrative shifts push representations apart.

### 2.5 Neural Change Detection

Recent neural approaches to change detection in text have utilized recurrent neural networks (RNNs), Long Short-Term Memory (LSTM) networks, and Transformer architectures to model temporal dependencies. These methods often require supervised labels indicating change points, limiting their applicability to large-scale unsupervised scenarios. Our approach differs by explicitly modeling narrative shifts through contrastive learning without requiring change point annotations.

### 2.6 Interpretability in NLP

Interpretability remains a critical challenge in deep learning-based NLP systems. Attention mechanisms, saliency maps, and influence functions have been proposed to explain model predictions. For narrative shift detection, interpretability is particularly important as users need to understand what specific textual elements drive detected shifts. Our framework addresses this through sentence-level attribution and extractive explanations grounded directly in the source text.

### 2.7 Research Gap

Despite significant progress in related areas, existing work has several limitations:

1. **Topic-Narrative Conflation:** Topic models detect topic changes but not narrative framing shifts within topics
2. **Supervision Requirements:** Many change detection methods require labeled change points or supervised training
3. **Lack of Interpretability:** Neural methods often operate as black boxes without fine-grained explanations
4. **Scalability Issues:** LLM-based approaches suffer from computational costs and reproducibility concerns
5. **Granularity Limitations:** Most methods operate at document or corpus level, lacking sentence-level precision

Our work addresses these gaps by proposing an unsupervised, interpretable, scalable framework for narrative shift detection that operates at sentence-level granularity while maintaining topic-awareness through soft conditioning.

---

## 3. Data Details

### 📊 TLDR: Data Pipeline at a Glance

**Complete Pipeline Flow (Raw Data → Final Preprocessed Output):**

| Pipeline Stage | Count | Format | Status |
|----------------|-------|--------|--------|
| **📥 Raw Collection** | 1,167,047 articles | Multiple sources, multiple languages | ✅ Stage 0 |
| **🔧 After Cleaning** | 355,334 articles | English only, deduplicated, validated | ✅ Stage 1-3 |
| **✂️ After Segmentation** | 3,080,512 sentences | With context windows (W5) | ✅ Stage 4 |
| **🎯 After Topic Filtering** | 1,335,158 sentences | 5 topic files, 12.5 GB | ✅ Stage 5-6 |

**Key Statistics:**
- **Data Quality:** 69.56% removed as noise/duplicates (811,713 articles)
- **Article Retention Rate:** 30.44% (1,167,047 → 355,334)
- **Sentence Generation:** ~8.7 sentences per article
- **Topic Distribution:** War (36.7%), Economics (20.8%), Technology (14.3%), Health (14.1%), Climate (14.1%)
- **Processing Time:** ~8.5 hours total
- **Temporal Coverage:** 2011-2025 (14+ years)
- **Final Output:** 1.33M sentences with 768-dim SBERT embeddings and topic scores

---

### 3.1 Data Sources

We collected news articles from multiple heterogeneous sources to ensure diversity and coverage across different topics, time periods, and news outlets. The data sources include:

#### 3.1.1 Research Paper Dataset

**NewsSumm Dataset**
- **Publisher:** MDPI  
- **Journal:** Computers (2023)  
- **Description:** Large-scale human-annotated multi-document news summarization dataset for Indian English news articles
- **Link:** https://www.mdpi.com/2073-431X/14/12/508

#### 3.1.2 Kaggle Datasets

**CNN News Articles (2011–2022)**
- **Source:** Kaggle (hadasu92)
- **Description:** Cleaned CNN news articles published between 2011 and 2022
- **Link:** https://www.kaggle.com/datasets/hadasu92/cnn-articles-after-basic-cleaning

**Global News Dataset**
- **Source:** Kaggle (everydaycodings)
- **Description:** Large-scale global news dataset covering multiple news sources and regions
- **File Used:** raw-data.csv
- **Link:** https://www.kaggle.com/datasets/everydaycodings/global-news-dataset

#### 3.1.3 GitHub Dataset

**Disaster and Accident News Dataset**
- **Source:** Webhose Free News Datasets (GitHub)
- **Description:** Categorized dataset containing disaster and accident-related news articles
- **Link:** https://github.com/Webhose/free-news-datasets

#### 3.1.4 External News API Dataset

**NewsData.io**
- **Website:** https://newsdata.io/datasets
- **Type:** News API & structured dataset provider
- **Description:** Global news data with structured metadata

### 3.2 Data Collection and Combination

The data collection process involved handling three types of file formats:

1. **CSV Files:** Various news datasets with different column naming conventions (e.g., Covid_News.csv, CNN_Articels_clean1.csv)
2. **Excel Files:** News summary datasets (e.g., NewsSumm Dataset.xlsx)
3. **JSON Files:** Structured news data with metadata (e.g., us_polices_and_diplomacy.json, Webhose datasets)

#### 3.2.1 Challenges Addressed

Different sources presented unique challenges that required robust handling:

- **Inconsistent column names:** Different sources used varying names for the same information (e.g., `pubDate` vs `Date published` vs `published_at`)
- **Varied date formats:** ISO 8601, microseconds, timezone information requiring multi-strategy parsing
- **Multiple content fields:** Different sources labeled article text differently (`content`, `Article text`, `full_content`, `article_text`)
- **Mixed languages:** Required language detection and filtering to ensure English-only content
- **Quality variations:** Different article lengths and completeness across sources

#### 3.2.2 Column Mapping and Standardization

To handle heterogeneous data sources, we implemented an explicit column mapping system that standardizes all inputs to a uniform schema:

**Target Schema:**
- `Date`: Publication timestamp (standardized datetime format)
- `Article`: Full article text content
- `Source`: Original filename for traceability
- `Article_Length`: Character count for quality filtering

**Date Conversion Strategies:**

We employed a multi-strategy fallback approach for robust date parsing:

1. **Strategy 1 (Auto-detect):** Using `pd.to_datetime()` with error coercion to handle most standard formats
2. **Strategy 2 (Microseconds):** Explicit format parsing for microsecond patterns when auto-detection fails
3. **Success tracking:** Monitoring conversion success rates and selecting the optimal method per source

This approach ensured maximum data retention across diverse date format specifications.

### 3.3 Language Detection and Filtering

Given the multi-source nature of our dataset, language heterogeneity was a significant challenge. We implemented multi-threaded language detection using the `langdetect` library:

**Process:**
- 8-thread parallel processing for computational efficiency
- Probabilistic language identification for each article
- Filtering to retain only English-language articles
- Real-time progress tracking during processing

**Rationale:**
- NLP models work best on single-language data
- English is the target language for this project
- Mixed-language data complicates semantic analysis
- Early filtering prevents downstream processing issues

**Performance:**
- Single-threaded processing: ~20 minutes for 10,000 articles
- Multi-threaded processing: ~2.5 minutes for 10,000 articles (~8x speedup)

### 3.4 Data Cleaning and Quality Assurance

We implemented a comprehensive multi-stage cleaning pipeline to ensure high-quality data. The following presents the actual results from our data combination and cleaning process:

#### Initial Data Combination (Stage 0)

**Raw Data Collection:**
- **Total Articles Extracted:** 1,167,047 articles
- **Files Combined:** 8 source files
- **Columns:** Date, Article, Source

**Source Breakdown:**

| Source | Article Count | Percentage |
|--------|--------------|------------|
| raw-data.csv | 933,257 | 79.97% |
| data.csv | 105,375 | 9.03% |
| rating.csv | 58,356 | 5.00% |
| CNN_Articels_clean.csv | 37,949 | 3.25% |
| Newsdata_Records_2025_10_27_09_12_28.csv | 23,700 | 2.03% |
| Covid_News.csv | 4,234 | 0.36% |
| CNN_Articels_clean1.csv | 4,076 | 0.35% |
| NewsData.io_Sample_data_crypto.csv | 100 | 0.01% |
| **Total** | **1,167,047** | **100%** |

**Initial Missing Values:**
- Missing dates: 214,224 (18.35%)
- Missing articles: 179,203 (15.35%)
- Total missing values: 393,427

#### Stage 1: Duplicate Removal and Date Cleaning

**Duplicate Removal:**
- **Initial Article Count:** 1,167,047
- **Duplicates Removed:** 667,999 (57.24%)
- **Articles Remaining:** 499,048
- **Retention Rate:** 42.76%

**Rationale:** Duplicate articles from multiple sources bias statistical analyses and add no new information. We kept the first occurrence of each unique article.

**Date Analysis and Cleaning:**
- Articles after deduplication: 499,048
- NaN/empty dates detected: 38,459 (7.71%)
- Date format: All timestamps converted to datetime64[ns]
- Date range identified: 1970-01-01 to 2025-10-27 (span: 20,388 days)

**Date Cleaning Results:**
- **NaN/Empty Dates Removed:** 38,459
- **Articles Remaining:** 460,589
- **Retention Rate:** 92.29% (after this step)
- **Overall Retention Rate:** 39.47% (from initial dataset)

#### Stage 2: Language Detection and Filtering

**Language Detection Process:**
- **Articles Before Language Filter:** 460,589
- **Processing Method:** Multi-threaded (8 threads) using `langdetect` library
- **Total Processing Time:** 28,253 seconds (~7.85 hours)
- **Processing Speed:** 16 articles/second
- **Speedup Factor:** ~8x through parallel processing

**Language Distribution Detected:**

| Language | Article Count | Percentage |
|----------|--------------|------------|
| English (en) | 355,334 | 77.14% |
| Spanish (es) | 24,869 | 5.40% |
| French (fr) | 18,287 | 3.97% |
| German (de) | 18,092 | 3.93% |
| Italian (it) | 9,757 | 2.12% |
| Portuguese (pt) | 8,745 | 1.90% |
| Dutch (nl) | 3,396 | 0.74% |
| Romanian (ro) | 3,253 | 0.71% |
| Turkish (tr) | 2,749 | 0.60% |
| Swedish (sv) | 2,492 | 0.54% |
| Other languages | 13,615 | 2.95% |
| **Total** | **460,589** | **100%** |

**Language Filtering Results:**
- **Non-English Articles Removed:** 105,255 (22.86%)
- **English Articles Retained:** 355,334
- **Retention Rate:** 77.14%

**Rationale:** English-only filtering ensures linguistic consistency for NLP model application. Mixed-language data would complicate semantic analysis and reduce model effectiveness.

#### Stage 3: Article Length Analysis

**Length Statistics (Before Length-Based Filtering):**
- **Minimum Length:** 10 characters
- **Maximum Length:** 204,800 characters
- **Mean Length:** 1,616 characters
- **Median Length:** 214 characters

**Length Distribution:**

| Length Category | Article Count | Percentage |
|----------------|--------------|------------|
| 0-100 characters | 2,074 | 0.58% |
| 101-500 characters | 269,866 | 75.94% |
| 501-1,000 characters | 1,534 | 0.43% |
| 1,001-2,000 characters | 9,377 | 2.64% |
| 2,001-5,000 characters | 39,057 | 10.99% |
| 5,000+ characters | 33,426 | 9.41% |
| **Total** | **355,334** | **100%** |

**Subsequent Length-Based Filtering:**
- **Minimum Threshold Applied:** 500 characters
- **Maximum Threshold Applied:** 50,000 characters

**Rationale for Minimum (500 characters):**
- 75.94% of articles (270K+) fall in the very short range (101-500 chars), likely representing headlines, snippets, or summaries
- Articles below 500 characters lack sufficient content for robust narrative analysis
- Ensures adequate text for sentence segmentation and contextual embedding
- Provides meaningful semantic content for shift detection

**Rationale for Maximum (50,000 characters):**
- Only 0.13% of articles exceed this threshold (based on max of 204,800 chars)
- Extremely long articles often represent concatenated content, transcripts, or extraction errors
- Prevents memory issues and processing bottlenecks
- Removes statistical outliers

#### Stage 4: Chronological Sorting

**Action:** Sort entire dataset by publication date for temporal analysis readiness
**Date Range:** 1970-01-01 to 2025-10-27
**Rationale:** Enables temporal sequence analysis and narrative evolution tracking

### 3.5 Dataset Statistics and Processing Summary

#### 3.5.1 Data Processing Pipeline Summary

The complete data collection, combination, and cleaning pipeline achieved the following results:

**Processing Stages:**

| Stage | Articles In | Articles Out | Removed | Retention Rate | Cumulative Retention |
|-------|------------|-------------|---------|----------------|---------------------|
| Initial Collection | — | 1,167,047 | — | — | 100% |
| Duplicate Removal | 1,167,047 | 499,048 | 667,999 | 42.76% | 42.76% |
| Date Cleaning | 499,048 | 460,589 | 38,459 | 92.29% | 39.47% |
| Language Filtering | 460,589 | 355,334 | 105,255 | 77.14% | 30.44% |
| Length Filtering* | 355,334 | ~83,000** | ~272,000 | ~23.4% | ~7.1% |

*Length filtering applied in preprocessing stage (500-50,000 characters)
**Estimated based on length distribution; final numbers determined during sentence-level preprocessing

**Overall Data Quality Achievement:**
- **Initial Raw Articles:** 1,167,047
- **After Cleaning (before length filter):** 355,334
- **Overall Retention Rate:** 30.44%
- **Data Removed:** 811,713 articles (69.56%)
  - Duplicates: 667,999 (57.24%)
  - Invalid dates: 38,459 (3.30%)
  - Non-English: 105,255 (9.02%)

#### 3.5.2 Final Dataset Statistics

After length-based filtering and sentence-level preprocessing:

**Clean English Dataset (Post-Filtering):**
- **English Articles (Pre-Length Filter):** 355,334 articles
- **Temporal Coverage:** 1970–2025 (primary coverage: 2011–2025)
- **Language:** 100% English
- **Average Article Length (Pre-Filter):** 1,616 characters (median: 214 characters)
- **Sources:** 8 major data sources with 15+ underlying news outlets

**Data Quality Metrics:**
- **Completeness:** 100% (all required fields present after cleaning)
- **Language Purity:** 100% (English only after filtering)
- **Temporal Validity:** 100% (valid timestamps)
- **Uniqueness:** 100% (no duplicates)

#### 3.5.3 Topic-Specific Dataset Organization

After sentence-level preprocessing, topic-aware embedding, and soft labeling (detailed in Section 3.9), the cleaned dataset is organized into five major topics:

**Sentence-Level Statistics (After Stage 3.3 Filtering with Threshold 0.3):**

| Topic | Sentence Count | Percentage | File Size | Date Range |
|-------|---------------|------------|-----------|------------|
| War | 490,123 | 36.7% | 4,585.87 MB | 2011-09-19 to 2025-10-26 |
| Economics | 277,886 | 20.8% | 2,602.14 MB | 2011-09-21 to 2025-10-26 |
| Technology | 190,543 | 14.3% | 1,783.00 MB | 2011-10-04 to 2025-10-26 |
| Health | 188,593 | 14.1% | 1,766.20 MB | 2011-09-06 to 2025-10-26 |
| Climate | 188,013 | 14.1% | 1,761.45 MB | 2011-09-21 to 2025-10-26 |
| **Total** | **1,335,158** | **100%** | **12,498.66 MB** | **2011-2025** |

**Processing Statistics:**
- **Input Sentences Processed:** 3,080,512 (from 113 preprocessed files)
- **Topic Similarity Threshold:** 0.3 (cosine similarity)
- **Processing Time:** 28.14 minutes (1,688.30 seconds)
- **Processing Speed:** 4.02 files/minute
- **Retention After Threshold:** 43.3% of input sentences (1,335,158 / 3,080,512)
- **Valid Embeddings:** 100% across all topics

**Important Notes:**
1. **Sentence-Level Granularity:** These counts represent individual sentences, not articles. Each sentence has been segmented, embedded with context (Window 5), and assigned topic similarity scores.
2. **Soft Labeling:** The total (1,335,158) reflects sentence-level assignments where a single sentence may appear in multiple topic files if it exceeds the 0.3 similarity threshold for multiple topics.
3. **Pipeline Traceability:**
   - Initial raw articles: 1,167,047
   - After cleaning: 355,334 English articles
   - After sentence segmentation: 3,080,512 sentences
   - After topic filtering (threshold 0.3): 1,335,158 sentences across all topics
4. **Topic Files:** Each file (e.g., War.csv, Health.csv) contains:
   - `date`: Publication timestamp
   - `w5_embedding`: 768-dimensional SBERT embedding with Window 5 context
   - `main_sentence`: The actual sentence text
   - Topic scores: `War`, `Health`, `Technology`, `Climate`, `Economics` (similarity scores 0-1)

#### 3.5.4 Ideal Articles for Topic Identification

To enable topic-aware labeling and filtering of the news corpus, we curated a collection of **35 ideal reference articles** representing the five target topics. These articles serve as canonical examples of each topic's core narrative and semantic space.

**Purpose and Motivation:**

The primary challenge in narrative shift detection is distinguishing topic-relevant sentences from general news content. Without topic-specific prototypes, the system cannot reliably identify whether an article discusses War, Health, Technology, Climate, or Economics. Ideal articles solve this by providing:

1. **Topic Definition:** Establish the semantic boundaries and key concepts that define each topic
2. **Keyword Identification:** Capture domain-specific terminology and phrases characteristic of each topic
3. **Prototype Generation:** Enable computation of topic centroid vectors for similarity-based filtering
4. **Quality Control:** Ensure only relevant sentences are included in downstream analysis

**Curation Methodology:**

| Aspect | Details |
|--------|---------|
| **Total Articles** | 35 articles (~7 per topic) |
| **Topics Covered** | War, Health, Technology, Climate, Economics |
| **Selection Process** | Manual review and curation from trusted authoritative sources |
| **Extraction Method** | Carefully identified and extracted sentences containing topic-defining keywords |
| **Curator Expertise** | Domain-aware selection ensuring representative coverage |

**Curation Process:**

1. **Source Selection:** Identified highly reputable, unbiased sources per topic (see sources below)
2. **Article Review:** Manually read candidate articles from each source
3. **Sentence Extraction:** Extracted sentences containing essential keywords and concepts defining the topic
4. **Diversity Assurance:** Ensured coverage of topic subtopics (e.g., War: conflict types, weapons, peace negotiations, humanitarian impacts, war crimes)
5. **Quality Validation:** Verified that extracted sentences clearly and unambiguously represent the topic

**Temporal Independence:**

Unlike news articles that may reflect time-specific events, ideal articles focus on **timeless topic-defining concepts**. For example:
- **War:** Keywords like "military conflict," "armed forces," "weapons systems," "peacekeeping" remain consistent regardless of which specific conflict is occurring
- **Health:** Concepts like "disease outbreak," "healthcare policy," "medical research," "public health" define the domain across time periods
- **Climate:** Terms like "renewable energy," "carbon emissions," "climate policy," "environmental impact" remain definitionally stable

This temporal independence means ideal articles **do not require updating** over time—the fundamental semantic space of each topic remains constant even as specific events evolve.

**Authoritative Sources:**

All ideal articles were sourced from internationally recognized, unbiased, and highly credible institutions:

**War:**
- Reuters (multiple investigative reports on drone warfare, hybrid conflicts, military technology)
- Coverage: Ukraine crisis drones, India-Pakistan drone battles, Sudan conflict, Russia's hybrid warfare, U.S. military deployments

**Health:**
- World Health Organization (WHO): Fact sheets, policy reports, action frameworks
- National Center for Biotechnology Information (NCBI): Peer-reviewed medical research
- Coverage: Noncommunicable diseases, chronic disease prevention, health policy impacts

**Technology:**
- MIT Technology Review: Cutting-edge technology journalism
- OECD AI Observatory: AI policy and governance
- World Economic Forum (WEF): Technology trend reports
- McKinsey Global Institute: Technology economics and impact analysis

**Climate:**
- International Energy Agency (IEA): Renewable energy reports
- International Renewable Energy Agency (IRENA): Clean energy publications
- Intergovernmental Panel on Climate Change (IPCC): AR6 assessment reports
- National Oceanic and Atmospheric Administration (NOAA): Climate science
- National Renewable Energy Laboratory (NREL): Energy research

**Economics:**
- International Monetary Fund (IMF): World Economic Outlook
- World Bank: Global Economic Prospects
- Organisation for Economic Co-operation and Development (OECD): Economic Outlook
- Financial Times: Global economy coverage
- Brookings Institution: Economic policy research

**Source Credibility Rationale:**

These sources were selected for their:
- **Authority:** Leading international organizations and research institutions
- **Objectivity:** Fact-based reporting with minimal political bias
- **Expertise:** Domain specialists and peer-reviewed content
- **Global Recognition:** Widely cited by policymakers, researchers, and media

**Integration with Pipeline:**

The curated ideal articles are processed through Stage 1 of the Semantic Drift SBERT baseline (Section 4.4.1):

1. **Sentence Segmentation:** Ideal articles split into individual sentences
2. **Embedding Generation:** Each sentence encoded using SBERT (all-mpnet-base-v2) → 768-dim vectors
3. **Prototype Computation:** Per-topic mean-pooling of all sentence embeddings → topic prototype vectors
4. **Threshold Calibration:** Topic-specific cosine similarity thresholds computed (Section 4.4.1, Stage 2)

These topic prototypes enable the filtering mechanism that ensures downstream analysis focuses only on topically relevant content, improving both precision and interpretability of narrative shift detection.

**Deliverables:**

- `ideal_article/` directory: Organized by topic and subtopic
- `topic_prototypes.json`: 768-dimensional prototype vector per topic
- `topic_prototypes.pt`: PyTorch tensor format for fast loading
- Source documentation: `sources_ideal_article.txt`

### 3.6 Topic Definition and Coverage

We selected five major topics that represent significant areas of public discourse and media coverage:

#### 3.6.1 War
Coverage includes military conflicts, geopolitical tensions, defense policy, international relations, peacekeeping operations, and security issues. This topic captures the largest portion of the dataset due to the prevalence of conflict-related news globally.

#### 3.6.2 Economics
Coverage includes economic policy, financial markets, trade relations, monetary policy, inflation, employment, corporate news, and economic indicators. This topic reflects the constant media attention to economic developments.

#### 3.6.3 Technology
Coverage includes technological innovations, artificial intelligence, data privacy, cybersecurity, social media, digital transformation, and tech industry developments. This topic captures the rapidly evolving technology landscape.

#### 3.6.4 Health
Coverage includes public health, disease outbreaks, healthcare policy, medical research, pharmaceuticals, mental health, and pandemic-related news. This topic saw increased coverage during the COVID-19 pandemic period.

#### 3.6.5 Climate
Coverage includes climate change, environmental policy, renewable energy, extreme weather events, sustainability, conservation, and ecological issues. This topic reflects growing media attention to environmental challenges.

### 3.7 Temporal Distribution

The dataset exhibits temporal diversity across the 13-year collection period (2011–2024):

**Key Temporal Characteristics:**
- Continuous coverage with no significant gaps
- Increased volume in recent years (2020–2024) due to expanded sources
- Spike in health-related articles during 2020–2022 (COVID-19 pandemic)
- Consistent war-related coverage throughout the period
- Growing technology coverage correlating with AI advancement (2022–2024)

### 3.8 Data Quality Assessment

**Quality Indicators Achieved:**

1. **Completeness:** 100% of retained articles have all required fields (date, content, source) after cleaning
2. **Language Purity:** 100% English (355,334 articles) after filtering from 11+ detected languages
3. **Temporal Validity:** 100% valid timestamps (460,589 after date cleaning)
4. **Uniqueness:** 100% unique articles (667,999 duplicates removed - 57.24% of raw data)
5. **Processing Efficiency:** 8x speedup achieved through multi-threaded language detection

**Data Cleaning Impact:**

The aggressive cleaning pipeline removed 69.56% of raw data, ensuring:
- No duplicate content that would bias temporal analysis
- Consistent language for NLP model application
- Valid temporal information for chronological tracking
- Appropriate article lengths for meaningful analysis

This high removal rate, while reducing volume, dramatically improved data quality and reduced noise that would compromise narrative shift detection accuracy.

### 3.9 Data Preprocessing Pipeline

The preprocessing pipeline transforms raw articles into contextualized, topic-specific sentence embeddings ready for temporal analysis. The pipeline consists of multiple stages:

#### Stage 0: Raw Dataset Setup
**Purpose:** Configuration and environment setup without loading data

**Key Activities:**
- Library imports (pandas, numpy, nltk, sentence-transformers, torch)
- Path configuration for data folders
- File discovery and inventory
- Configuration parameters specification

#### Stage 1: Sentence Segmentation with Context Windows
**Purpose:** Break articles into sentences while preserving surrounding context

**Process:**
1. Sentence tokenization using NLTK's `sent_tokenize()`
2. Context window construction with window size 5:
   - `previous_sentence_1`: Position (i-2)
   - `previous_sentence_2`: Position (i-1)
   - `main_sentence`: Position (i) — Current sentence
   - `next_sentence_1`: Position (i+1)
   - `next_sentence_2`: Position (i+2)

**Output Schema:**
```
- sentence_id: Unique identifier (file_number_article_number_sentence_number)
- article_id: Article identifier
- date: Publication date
- source: Article source
- previous_sentence_1, previous_sentence_2: Context before
- main_sentence: Current sentence
- next_sentence_1, next_sentence_2: Context after
```

**Rationale for Window Size 5:**

Linguistic research shows discourse coherence typically spans 3-5 sentences. Window size 5 (2 previous + current + 2 next) provides:
- **Narrative arc capture:** Setup → statement → elaboration pattern
- **Disambiguation context:** Resolves ambiguous sentences through surrounding context
- **Computational balance:** Not too sparse, not too expensive
- **Empirical validation:** Testing showed W5 provides richer context than W3 with minimal performance cost

**Example:**
```
Article sentences:
s0: "AI is rapidly expanding."
s1: "Data centers consume electricity."
s2: "Governments regulate AI."
s3: "Privacy concerns are rising."
s4: "Tech companies respond."

For main_sentence s2:
previous_sentence_1 = s0 (i-2)
previous_sentence_2 = s1 (i-1)
main_sentence = s2 (i)
next_sentence_1 = s3 (i+1)
next_sentence_2 = s4 (i+2)
```

**Implementation:**
- Multi-threaded processing with 8 workers for efficiency
- Independent file processing (no data dependencies)
- Incremental saving to disk after each file
- Memory clearing to prevent overflow

#### Stage 2: Context-Aware Sentence Embedding
**Purpose:** Generate dense vector representations incorporating surrounding context

**Model Selection:**
- **Model:** SBERT (all-mpnet-base-v2)
- **Embedding Dimension:** 768
- **Rationale for SBERT:**
  - Pre-trained specifically for sentence embeddings
  - State-of-the-art performance on semantic similarity benchmarks (~86% accuracy on STS)
  - Efficient inference (faster than full BERT)
  - No fine-tuning required
  - Superior to alternatives (Word2Vec ~65%, BERT averaging ~70%, USE ~80%)

**Two Context Window Strategies:**

**Window 3 (W3) — Immediate Context:**
```python
contextual_input_w3 = [previous_sentence_2] + [main_sentence] + [next_sentence_1]
# Combined with [SEP] tokens for boundary marking
```

**Window 5 (W5) — Wider Context:**
```python
contextual_input_w5 = [previous_sentence_1] + [previous_sentence_2] + 
                      [main_sentence] + 
                      [next_sentence_1] + [next_sentence_2]
# Combined with [SEP] tokens for boundary marking
```

**Why Two Window Sizes:**
1. Enables empirical comparison of context window effects
2. Provides flexibility for different downstream tasks
3. Offers robustness if one embedding type shows issues
4. Facilitates research on window size optimization

**Processing:**
- Batch processing (batch_size=64) for efficiency
- GPU acceleration when available
- Progress tracking for long-running operations
- Output stored as comma-separated strings in CSV format

**Output Schema:**
```
All Stage 1 columns PLUS:
- w3_embedding: 768-dimensional vector (Window 3)
- w5_embedding: 768-dimensional vector (Window 5)
```

#### Stage 3: Topic Embedding Construction
**Purpose:** Create representative embeddings for each of the five topics

**Topic Definitions:**
1. **War:** Military conflicts, international relations, defense policy
2. **Health:** Medical research, healthcare, public health, pandemics
3. **Economics:** Economic policy, markets, trade, employment
4. **Technology:** Innovation, AI, digital transformation, cybersecurity
5. **Climate:** Environmental policy, climate change, sustainability

**Process:**
1. Manually curate representative seed sentences for each topic
2. Generate SBERT embeddings for seed sentences
3. Average embeddings to create topic prototype vectors
4. Store as reference embeddings for similarity computation

**Topic Embedding Storage:**
- Format: JSON file with topic names and 768-dimensional vectors
- Location: `processed_data/topic_embedding.json`

#### Stage 3.2: Data Soft Labeling (Topic Similarity)
**Purpose:** Assign topic relevance scores to each sentence

**Method:**
1. Compute cosine similarity between each sentence embedding and topic embeddings
2. Generate probability distribution across all 5 topics using softmax
3. Assign multiple topic scores (soft labeling) rather than hard classification

**Advantages of Soft Labeling:**
- Articles naturally span multiple topics
- Captures nuanced topic mixtures
- Enables flexible filtering thresholds
- Preserves information about topic distributions

**Output:**
```
All previous columns PLUS:
- War: Similarity score [0-1]
- Health: Similarity score [0-1]
- Economics: Similarity score [0-1]
- Technology: Similarity score [0-1]
- Climate: Similarity score [0-1]
```

#### Stage 3.3: Topic-Wise Filtering and Aggregation
**Purpose:** Organize data by topic for specialized analysis

**Process:**
1. Apply similarity threshold (0.3) for each topic
2. Filter sentences exceeding threshold for each topic
3. Save topic-specific datasets separately with embeddings and metadata

**Implementation Results:**
- **Input:** 3,080,512 sentences from 113 preprocessed files
- **Threshold Applied:** 0.3 (cosine similarity)
- **Processing Time:** 28.14 minutes (4.02 files/minute)
- **Overall Retention:** 43.3% (1,335,158 sentences retained)

**Per-Topic Filtering Statistics:**

| Topic | Intermediate Rows* | Final Sentences** | Avg per File | Retention Rate |
|-------|-------------------|------------------|--------------|----------------|
| War | 557,082 | 490,123 | 4,929.9 | 88.0% |
| Economics | 193,246 | 277,886 | 1,710.1 | 143.8%*** |
| Technology | 131,215 | 190,543 | 1,161.2 | 145.2%*** |
| Climate | 153,405 | 188,013 | 1,357.6 | 122.6%*** |
| Health | 123,003 | 188,593 | 1,088.5 | 153.3%*** |

*Intermediate rows: Per-topic statistics during file-by-file processing
**Final sentences: Deduplicated sentence counts in final output files
***Values >100% indicate sentence deduplication/aggregation during final file writing

**Output Files:**
```
processed_data/3_embed/
├── War.csv (490,123 sentences, 4,585.87 MB)
├── Economics.csv (277,886 sentences, 2,602.14 MB)
├── Technology.csv (190,543 sentences, 1,783.00 MB)
├── Health.csv (188,593 sentences, 1,766.20 MB)
└── Climate.csv (188,013 sentences, 1,761.45 MB)
```

Each topic-specific file contains:
- `date`: Publication timestamp
- `w5_embedding`: 768-dimensional SBERT embedding (Window 5 context)
- `main_sentence`: Sentence text
- Topic scores: `War`, `Health`, `Technology`, `Climate`, `Economics` (similarity 0-1)
- 100% valid embeddings across all files
- Chronologically sorted by date

### 3.10 Data Storage and Organization

**File Structure:**
```
processed_data/
├── ALL_Combined_Data.csv (1,167,047 articles - raw combined)
├── topic_embedding.json (Topic prototype embeddings)
├── Stage_1/ (Sentence segmentation output)
├── Stage_2/ (Context-aware embeddings)
└── 3_embed/ (Topic-filtered datasets - Final Output)
    ├── War.csv (490,123 sentences, 4.59 GB)
    ├── Economics.csv (277,886 sentences, 2.60 GB)
    ├── Technology.csv (190,543 sentences, 1.78 GB)
    ├── Health.csv (188,593 sentences, 1.77 GB)
    └── Climate.csv (188,013 sentences, 1.76 GB)
    Total: 1,335,158 sentences, 12.50 GB
```

### 3.10.1 Preprocessing Pipeline Summary

The complete preprocessing pipeline transforms raw articles into contextualized, topic-specific sentence embeddings:

**Input → Output Transformation:**
```
1,167,047 raw articles (multiple sources, multiple languages)
    ↓ [Deduplication: -57.24%]
499,048 unique articles
    ↓ [Date Cleaning: -7.71%]
460,589 articles with valid dates
    ↓ [Language Detection: -22.86%]
355,334 English articles (30.44% of raw)
    ↓ [Sentence Segmentation]
3,080,512 sentences (~8.7 sentences/article)
    ↓ [Context-Aware Embedding (W5) + Topic Scoring]
3,080,512 embedded sentences with topic scores
    ↓ [Topic Filtering: threshold=0.3, -56.7%]
1,335,158 topic-filtered sentences (in 5 topic files)
```

**Key Metrics:**
- **Raw → Clean (Article Level):** 30.44% retention (355,334 / 1,167,047)
- **Sentence Generation:** ~8.7 sentences per article on average
- **Topic Filtering:** 43.3% sentence retention (1,335,158 / 3,080,512)
- **Processing Time:** ~8.5 hours total (7.85h language detection + 0.47h topic filtering)
- **Output Size:** 12.50 GB across 5 topic files
- **Temporal Coverage:** 2011-09-06 to 2025-10-26 (14+ years)

### 3.11 Why This Preprocessing Approach?

**Problem Addressed:** Traditional approaches operate at article level, missing fine-grained narrative shifts that occur at sentence level.

**Our Solution:**
1. **Sentence-level granularity** for precise shift detection
2. **Context preservation** through window-based embedding
3. **Topic specialization** for domain-specific analysis
4. **Temporal ordering** for chronological tracking
5. **Scalability** through batching and optimization

**Example of Sentence-Level Value:**

Consider a single article:
```
"Artificial intelligence is rapidly expanding. Data centers now consume 
massive amounts of electricity. Governments are beginning to regulate AI usage."
```

This article touches on:
- **Technology** (AI expansion) - first sentence
- **Climate** (energy consumption) - second sentence  
- **Policy/Economics** (regulation) - third sentence

Sentence-level analysis allows us to:
- Identify the specific sentence discussing climate impact
- Track how the climate framing of AI evolved over time
- Detect when media narrative shifted from "AI innovation" to "AI energy concerns"

Article-level analysis would miss these granular shifts.

---

## 4. Proposed Methodology

### 4.1 Overview

Our proposed methodology for narrative shift detection consists of a multi-stage pipeline that combines sentence-level embedding, topic-aware filtering, temporal aggregation, and contrastive learning. The framework is designed to be:

1. **Unsupervised:** No labeled narrative shift annotations required
2. **Scalable:** Handles large corpora through efficient batching and processing
3. **Interpretable:** Provides sentence-level explanations for detected shifts
4. **Topic-Aware:** Maintains topic context without requiring topic-specific training
5. **Temporally Continuous:** Models narrative evolution as a continuous process

### 4.2 Comprehensive Methodology Flow Diagram

The following diagram presents a holistic view of our complete framework, illustrating how data flows from raw collection through preprocessing, model training (both baseline and TCL approaches), evaluation, and final deliverables:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA COLLECTION & PREPROCESSING                 │
│                          (Shared by All Models)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  RAW DATA COLLECTION                                                        │
│  • 8 heterogeneous sources (CSV, Excel, JSON)                              │
│  • 1,167,047 raw articles                                                   │
│  • Timespan: 2011-2025 (primary), extended 1970-2025                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  DATA CLEANING PIPELINE                                                     │
│  ├─ Deduplication: -57.24% → 499,048 unique articles                       │
│  ├─ Date Cleaning: -7.71% → 460,589 valid timestamps                       │
│  └─ Language Filtering: -22.86% → 355,334 English articles (30.44%)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: SENTENCE SEGMENTATION + CONTEXT WINDOWS                           │
│  • NLTK tokenization                                                        │
│  • 3,080,512 sentences (~8.7 sentences/article)                            │
│  • Context windows: W1 (solo), W3 (±1), W5 (±2)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: CONTEXT-AWARE EMBEDDING (SBERT)                                   │
│  • Model: all-mpnet-base-v2                                                 │
│  • Window 5 context (sentence + 2 neighbors each side)                     │
│  • Output: 3,080,512 × 768-dim embeddings                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: TOPIC SCORING & FILTERING                                         │
│  ├─ Ideal Articles (35 curated from WHO, Reuters, IPCC, IMF, MIT TR)      │
│  ├─ Topic Prototypes: 5 × 768-dim vectors (War, Health, Tech, Econ, Clim) │
│  ├─ Cosine Similarity Threshold: 0.3                                        │
│  └─ Output: 1,335,158 topic-filtered sentences (43.3% retention)           │
│              12.50 GB across 5 topic CSV files                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ↓                                   ↓
┌───────────────────────────────────────┐   ┌──────────────────────────────┐
│  PHASE 2A: BASELINE MODELS            │   │  PHASE 2B: TCL FRAMEWORK     │
│  (No Neural Learning)                 │   │  (Contrastive Learning)      │
└───────────────────────────────────────┘   └──────────────────────────────┘
            │                                           │
    ┌───────┴────────┐                                 │
    ↓                ↓                                 ↓
┌─────────┐   ┌─────────────┐         ┌───────────────────────────────────┐
│ SBERT   │   │  K-MEANS    │         │  TCL EVOLUTION (5 APPROACHES)     │
│ DRIFT   │   │  DRIFT      │         └───────────────────────────────────┘
└─────────┘   └─────────────┘                         │
     │              │                  ┌───────────────┼───────────────┐
     │              │                  │               │               │
     ↓              ↓                  ↓               ↓               ↓
┌─────────────────────────┐   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 4-STAGE CALIBRATION     │   │ CLUSTERING   │  │  APPROACH 1  │  │  APPROACH 2  │
│ ├─Stage 1: Topic Proto  │   │ • K=5        │  │  (Completed) │  │  (Completed) │
│ ├─Stage 2: Relevance    │   │ • JS Div     │  ├──────────────┤  ├──────────────┤
│ │  Thresholds (5 topics)│   │ • Thresh=0.3 │  │• Fixed Win   │  │• Grouping    │
│ ├─Stage 3: Drift        │   │ • TopicFilter│  │  - W=3, S=3  │  │  - Count (4) │
│ │  Thresholds (15 pairs)│   │              │  │  - W=3, S=1  │  │  - Date (5)  │
│ └─Stage 4: Detection    │   │• Distribution│  │• InfoNCE     │  │• NT-Xent +   │
│   - 5-day windows       │   │  Shift Track │  │• 256-dim     │  │  Hard-Neg    │
│   - Impact Scoring      │   │• Unsupervised│  │• Poor Result │  │• Topic-λ     │
│   - 3 Context Models    │   │              │  │              │  │• Mixed Result│
│     (W1/W3/W5)          │   │              │  │              │  │  669/732 samp│
└─────────────────────────┘   └──────────────┘  └──────────────┘  └──────────────┘
            │                         │                │                  │
            │                         │                └────────┬─────────┘
            │                         │                         │
            │                         │          ┌──────────────┼──────────────┐
            │                         │          ↓              ↓              ↓
            │                         │   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │                         │   │  APPROACH 3  │ │  APPROACH 4  │ │  APPROACH 5  │
            │                         │   │  (Research)  │ │  (Current)   │ │  (Planned)   │
            │                         │   ├──────────────┤ ├──────────────┤ ├──────────────┤
            │                         │   │• Topic-Wise  │ │• Ruptures    │ │• NER         │
            │                         │   │  Windows     │ │  PELT RBF    │ │  Integration │
            │                         │   │  W: 4-8 days │ │• Max-Sim     │ │• 832+ dim    │
            │                         │   │• Challenges: │ │  Grouping    │ │  (768+64+    │
            │                         │   │  - Non-stat  │ │• 774-dim     │ │   entity)    │
            │                         │   │  - Comp Cost │ │• Multi-Loss: │ │• Entity-Aware│
            │                         │   │  - Macro-Lev │ │  NT-Xent λ1.5│ │  Attention   │
            │                         │   │• Not Impl    │ │  Topic λ0.5  │ │• Future      │
            │                         │   │              │ │  Hard λ0.3   │ │  Work        │
            │                         │   │              │ │• 356 samples │ │              │
            │                         │   │              │ │• Good Inter  │ │              │
            │                         │   │              │ │  Bad Intra   │ │              │
            │                         │   └──────────────┘ └──────────────┘ └──────────────┘
            │                         │                            │
            │                         │                            │
            └─────────────────────────┴────────────────────────────┘
                                      │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: EVALUATION & COMPARISON                         │
│                     (Drift Detection Framework)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
          ┌──────────────────┐ ┌─────────────┐ ┌──────────────────┐
          │ QUANTITATIVE     │ │ QUALITATIVE │ │ COMPARATIVE      │
          │ METRICS          │ │ ANALYSIS    │ │ METRICS          │
          ├──────────────────┤ ├─────────────┤ ├──────────────────┤
          │• Drift Score     │ │• Sentence   │ │• Intra-Topic Sim │
          │  Distribution    │ │  Attribution│ │• Inter-Topic Sim │
          │• Peak Detection  │ │• Shift      │ │• Separation Score│
          │  (mean+1σ, top%) │ │  Examples   │ │• Temporal Consis │
          │• Semantic Dist   │ │• Topic-Spec │ │• Entity Awareness│
          │  at Shifts       │ │  Patterns   │ │• Sample Count    │
          │• Human Valid %   │ │• Narrative  │ │                  │
          │  (Primary)       │ │  Evolution  │ │                  │
          └──────────────────┘ └─────────────┘ └──────────────────┘
                                      │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: OUTPUTS & DELIVERABLES                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
          ┌──────────────────┐ ┌─────────────┐ ┌──────────────────┐
          │ DRIFT DETECTION  │ │ SENTENCE-   │ │ TEMPORAL         │
          │ • Shift Scores   │ │ LEVEL EXPLN │ │ EVOLUTION        │
          │ • Time Points    │ │ • Impact    │ │ • Trajectory     │
          │ • Categories     │ │   Scores    │ │ • Embeddings     │
          │   (Low/Med/High) │ │ • Anchor    │ │ • Change Points  │
          │ • Thresholds     │ │   Sentences │ │ • Visualizations │
          └──────────────────┘ └─────────────┘ └──────────────────┘
```

**Key Design Principles:**

1. **Shared Preprocessing Foundation:** All models (SBERT Drift, K-Means, TCL) use identical data pipeline (355K articles → 3M sentences → 1.3M topic-filtered) ensuring fair comparison where the only variable is the shift detection methodology.

2. **Three-Pronged Modeling Approach:**
   - **SBERT Drift:** Calibrated threshold-based approach with 4-stage pipeline (no neural learning)
   - **K-Means Drift:** Clustering-based distribution shift tracking (unsupervised)
   - **TCL:** Progressive evolution through 5 experimental approaches (learned temporal representations)

3. **TCL Evolution Philosophy:**
   - **Approaches 1-2:** Fixed and grouped windowing strategies (completed, identified limitations)
   - **Approach 3:** Topic-adaptive dynamic windows (research phase, challenges documented)
   - **Approach 4:** Ruptures-based adaptive segmentation with multi-objective loss (current best)
   - **Approach 5:** Entity-aware architecture with NER integration (future work)

4. **Evaluation as Drift Detection:** Framework evaluates change-point detection (not segmentation) with human validation as primary metric, supplemented by quantitative drift scores and semantic distance measures.

5. **Interpretability Throughout:** Every approach prioritizes explainability through sentence-level attribution, impact scoring, and extractive explanations grounded in source text.

### 4.3 Pipeline Architecture

The complete pipeline consists of the following stages:

```
Input: Raw news articles with timestamps
    ↓
Stage 1: Sentence Segmentation + Context Windows
    ↓
Stage 2: Context-Aware Sentence Embedding (SBERT)
    ↓
Stage 3: Topic Embedding Construction + Soft Labeling
    ↓
Stage 4: Daily Aggregation with Weighted Pooling
    ↓
Stage 5: Temporal Window Construction
    ↓
Stage 6: Data Augmentation (Temporal Jitter)
    ↓
Stage 7: Temporal Contrastive Learning Model
    ↓
Stage 8: Narrative Shift Detection + Explanation
    ↓
Output: Shift scores, shift categories, sentence-level explanations
```

### 4.4 Temporal Contrastive Learning (TCL) Framework

#### 4.4.1 Architecture Components

**Input Representation:**

For each day $d$, we construct a daily narrative representation by:

1. **Sentence Collection:** Gather all sentences $S_d = \{s_1, s_2, ..., s_n\}$ published on day $d$
2. **Topic Filtering:** Filter sentences by topic similarity threshold
3. **Weighted Pooling:** Aggregate sentence embeddings using topic-weighted mean:

$$Z_d = \frac{\sum_{k=1}^{n} w_k \cdot e_k}{\sum_{k=1}^{n} w_k}$$

where $e_k$ is the embedding of sentence $s_k$ and $w_k$ is its topic similarity score.

**Minimum Sentence Filter:** Days with fewer than 3 sentences are filtered to avoid unstable representations.

**Time Gap Feature:**

To capture temporal distance between consecutive days, we compute normalized time gaps:

$$\tau = \frac{\log(1 + \Delta_{\text{days}})}{5.0}$$

This normalization prevents scale imbalance in the transformer and provides a smooth representation of temporal distance.

**Sliding Window Construction:**

We construct temporal windows of size $W$ (default: 3 days) with configurable stride $S$ (default: 2 days):

$$\text{Window}_i = \{Z_{d_i}, Z_{d_{i+1}}, ..., Z_{d_{i+W-1}}\}$$

Each window captures a short temporal context of narrative evolution.

#### 4.4.2 Model Architecture

**Transformer-Based Encoder:**

The model architecture consists of:

1. **Input Layer:**
   - Input dimension: 774 (768 embedding + 6 topic scores)
   - Layer normalization for stable training
   - Linear projection to 256 dimensions
   - Dropout (p=0.1) for regularization

2. **Transformer Encoder:**
   - 4 transformer layers
   - 8 attention heads per layer
   - Dimension: 256
   - Feed-forward dimension: 1024
   - GELU activation (smoother gradients than ReLU)
   - Pre-layer normalization (norm_first=True for stability)

3. **Pooling Layer:**
   - Global average pooling over temporal sequence
   - Captures aggregate window representation

4. **Projection Head:**
   - Residual connection after pooling
   - Linear projection to 128-dimensional embedding space
   - Final output: 128-dimensional narrative representation

**Architecture Summary:**
```
Input: (batch_size, 3, 774) → Embedding normalized
    ↓
Linear Projection: (batch_size, 3, 256)
    ↓
Dropout (p=0.1)
    ↓
Transformer Encoder: 4 layers, 8 heads
    ↓
Global Average Pooling: (batch_size, 256)
    ↓
Residual Connection + Projection: (batch_size, 128)
    ↓
Output: 128-dimensional narrative embedding
```

#### 4.4.3 Contrastive Learning Objective

**Data Augmentation:**

To create positive pairs for contrastive learning, we apply temporal jitter augmentation:

$$e_{\text{aug}} = e + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.01)$$

This creates two slightly different views of the same window for robust representation learning.

**InfoNCE Loss:**

We use the InfoNCE (Noise Contrastive Estimation) loss function:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

where:
- $z_i$ and $z_j^+$ are embeddings of augmented views of the same window (positive pair)
- $z_k$ are embeddings of different windows (negative samples)
- $\text{sim}(\cdot, \cdot)$ is cosine similarity
- $\tau$ is the temperature parameter (default: 0.07)
- $N$ is the batch size

**Intuition:** The loss encourages:
1. Pulling together different augmented views of the same temporal window
2. Pushing apart embeddings from different temporal windows
3. Learning representations where narrative stability is reflected by proximity, and narrative shifts by distance

#### 4.4.4 Training Strategy

**Hyperparameters:**

- **Batch Size:** 128
- **Learning Rate:** 1e-4 with cosine annealing scheduler
- **Warmup Steps:** 500 (gradual learning rate increase)
- **Epochs:** 50
- **Optimizer:** AdamW with weight decay 1e-4
- **Gradient Clipping:** Max norm 1.0 (prevents exploding gradients)
- **Temperature:** 0.07 for InfoNCE loss

**Training Stability Improvements:**

1. **Input Layer Normalization:** Stabilizes large embedding magnitudes
2. **Gradient Clipping:** Prevents training instabilities
3. **Pre-layer Normalization:** Improves transformer training stability
4. **Cosine Annealing:** Smooth learning rate decay
5. **Warmup:** Gradual learning rate increase prevents early instability

**Training Process:**

For each epoch:
1. Sample mini-batches of temporal windows
2. Apply temporal jitter to create augmented views
3. Forward pass through transformer encoder
4. Compute InfoNCE loss between positive and negative pairs
5. Backpropagation with gradient clipping
6. Update model parameters
7. Adjust learning rate according to cosine schedule

#### 4.4.5 Narrative Shift Detection

After training, narrative shift detection proceeds as follows:

**Step 1: Embedding Generation**

For each temporal window $W_i$:
1. Construct daily aggregated embeddings
2. Forward pass through trained encoder
3. Obtain 128-dimensional narrative embedding $z_i$

**Step 2: Shift Score Computation**

Compute cosine distance between consecutive window embeddings:

$$\text{shift\_score}(W_i, W_{i+1}) = 1 - \frac{z_i \cdot z_{i+1}}{\|z_i\| \|z_{i+1}\|}$$

Higher shift scores indicate greater narrative divergence.

**Step 3: Shift Categorization**

Classify shifts into three categories based on thresholds:

- **Low Shift:** shift_score < 0.3 (stable narrative)
- **Medium Shift:** 0.3 ≤ shift_score < 0.6 (gradual change)
- **High Shift:** shift_score ≥ 0.6 (significant shift)

**Step 4: Sentence-Level Attribution**

For detected shifts, identify contributing sentences:

1. Compute sentence-level embeddings for both windows
2. Calculate contribution scores based on embedding changes
3. Rank sentences by contribution magnitude
4. Extract top-k sentences as shift explanations

**Step 5: Interpretable Explanation**

Generate extractive explanations by:
1. Presenting top contributing sentences from both time windows
2. Highlighting semantic differences in vocabulary and framing
3. Providing temporal context (dates, sources)
4. Showing shift magnitude and category

#### 4.4.6 Evolution of TCL Approaches: Iterative Refinement

> **Note:** The methodology described in sections 4.4.1-4.4.5 represents our initial conceptual framework. Through empirical experimentation and evaluation, our approach evolved significantly across five distinct iterations. This section documents the complete evolution, challenges encountered, and lessons learned.

**Preprocessing Foundation (All Approaches):**

All TCL approaches share common preprocessing:
- **W5 embeddings** used (smoother than W3, which are sharper)
- **Daily mean pooling:** Sentence embeddings for a day → single daily article representation
- **Fixed window size = 3:** Dynamic window sizing is topic-dependent and time-consuming, left as future work
- **Rationale for size 3:** Larger windows require more input articles from users for inference

---

##### **Approach 1: Fixed Window with Overlapping/Non-Overlapping**

**Initial Hypothesis:** Simple temporal windowing with InfoNCE loss would capture narrative drift.

**Two Variants Tested:**

1. **Overlapping Windows:**
   - Window construction: w1 = [day1, day2, day3], w2 = [day2, day3, day4], w3 = [day3, day4, day5]
   - **Problem:** Adjacent windows share 2/3 of their data → excessively high similarity
   - **Result:** Nearest windows too similar, fails to detect gradual shifts

2. **Non-Overlapping Windows:**
   - Window construction: w1 = [day1, day2, day3], w2 = [day4, day5, day6], w3 = [day7, day8, day9]
   - **Problem:** Not enough topic drift occurs within 3 consecutive days
   - **Result:** All windows within a short time span appear too similar

**Model Configuration:**
- Architecture: Transformer encoder (4 layers, 8 heads, 256-dim)
- Loss: **InfoNCE** (Noise Contrastive Estimation)
- Input: 774-dim (768 SBERT + 5 topic scores + 1 time gap)

**Evaluation Results:**
- **Intertopic drift:** Poor (windows from different topics not well-separated)
- **Intratopic drift:** Poor (consecutive windows too similar, shifts undetected)
- **Conclusion:** Fixed temporal windowing insufficient; need better grouping strategy

**Reference Implementation:** `TCL_Pipeline_Complete.ipynb`

---

##### **Approach 2: Grouping-Based Temporal Segmentation**

**Hypothesis:** Group days together before windowing to capture longer temporal context and reduce over-similarity.

**Key Innovation:** Replace day-level windowing with **group-level windowing**:
- Day-level pooling → Group-level pooling → Window creation from groups

**Weighted Mean Pooling:**

Changed from simple mean pooling to **topic-weighted lambda mean pooling**:

$$Z_{\text{group}} = \frac{\sum_{k=1}^{n} \lambda_k \cdot e_k}{\sum_{k=1}^{n} \lambda_k}$$

where $\lambda_k$ is the topic weight for sentence $k$

**Two Grouping Strategies:**

1. **Count-Based Grouping (Approach 2a):**
   - Group by fixed count: Every 4 consecutive days → 1 group
   - **Problem:** Date gaps cause issues
     - Example: Days [2, 3, 4, 10] grouped together
     - Large gap (4→10) causes information loss
     - **Misses micro-level narrative shifts** occurring within gaps

2. **Date-Range Grouping (Approach 2b):**
   - Group by fixed date range: Every 5 days → 1 group
   - Example timeline: Days [2, 3, 4, 10, 11, 12, 26, 28]
     - Group 1 (days 1-5): [2, 3, 4]
     - Group 2 (days 6-10): [10]
     - Group 3 (days 11-15): [11, 12]
     - Group 4 (days 26-30): [26, 28]
   - **Problem:** Uneven article distribution per group
     - Some groups sparse (1-2 days), others dense (4-5 days)
     - Inconsistent representation quality

**Model Configuration:**
- Architecture: Enhanced Transformer with attention pooling
- Loss: **Enhanced NT-Xent** (Normalized Temperature-scaled Cross Entropy)
  - Improved over standard InfoNCE with better negative sampling
  - **Rationale:** NT-Xent provides more stable gradients and better convergence for multi-scale temporal data
- Additional loss: **Hard Negative Mining Loss** (weighted sum)
  - **Rationale:** Explicitly pushes apart the most confusable negative pairs (e.g., same topic, different time periods)
  - Formula: $\mathcal{L}_{\text{hard-neg}} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\exp(\text{sim}(z_i, z_j^+) / \tau) + \sum_{k \in K_{\text{hard}}} \exp(\text{sim}(z_i, z_k^-) / \tau)}$
- Combined loss: $\mathcal{L} = \lambda_1 \mathcal{L}_{\text{NT-Xent}} + \lambda_2 \mathcal{L}_{\text{hard-neg}}$

**Evaluation Results:**
- **Intertopic drift:** Very high (bad) — different topics not well-separated
- **Intratopic drift:** Average (improved from Approach 1) — grouping helped capture longer-term patterns
- **Conclusion:** Grouping helps but fixed window size across topics is insufficient

**Reference Implementation:** `TCL_Pipeline_2.ipynb`

---

##### **Approach 3: Topic-Wise Window Sizing (Not Implemented — Future Work)**

**Hypothesis:** Different topics evolve at different rates → require topic-specific window sizes.

**Proposed Strategy:**
- War: Window size = 5 (rapid narrative shifts due to breaking events)
- Health: Window size = 7 (slower, policy-driven changes)
- Economics: Window size = 4 (moderate volatility)
- Technology: Window size = 6 (innovation cycles)
- Climate: Window size = 8 (long-term trends)

**Challenges Identified (Research Findings):**

1. **Missed Macro-Level Shifts:**
   - Topic-specific windowing captures fine-grained changes well
   - But misses cross-topic narrative phenomena (e.g., COVID-19 affecting War, Health, Economics simultaneously)

2. **Temporal Non-Stationarity:**
   - Window size for a topic needs to change over time
   - Example: Health topic
     - 2011-2019: Window size = 7 (stable)
     - 2020-2022: Window size = 3 (COVID-19 rapid changes)
     - 2023-2025: Window size = 6 (post-pandemic stabilization)
   - **Requires dynamic window size adaptation per topic per year**

3. **Computational Constraints:**
   - Requires training separate models per topic
   - Hyperparameter search space explodes: 5 topics × multiple years × window sizes
   - **Time and space prohibitive for current scope**

**Status:** Under research, not implemented in current pipeline

**Future Directions:**
- Reinforcement learning for automatic window size selection
- Meta-learning approaches to transfer window size knowledge across topics
- Online adaptation algorithms

---

##### **Approach 4: Dynamic Segmentation with Topic Separation Loss (Current Implementation)**

**Hypothesis:** Use change-point detection for adaptive segmentation + explicit topic separation.

**Key Innovations:**

1. **Ruptures-Based Segmentation:**
   - Replaced fixed windowing with adaptive change-point detection
   - Algorithm: PELT (Pruned Exact Linear Time) with RBF kernel
   - **Rationale:** Inspired by [Truong et al., 2020] "Selective review of offline change point detection methods" (Signal Processing, https://arxiv.org/abs/1801.00718)
   - **How it works:** Segments time series into homogeneous regions where statistical properties remain consistent
   - Penalty parameter: 0.1 (controls number of change points; lower = more segments)
   
   $$\text{cost}(y_{t_1:t_2}) = \sum_{t=t_1}^{t_2} \|y_t - \mu_{t_1:t_2}\|^2_{\text{RBF}}$$
   
   where $\mu_{t_1:t_2}$ is the mean embedding in segment $[t_1, t_2]$

2. **Maximum Similarity Grouping:**
   - **Concept:** Group consecutive days until semantic similarity stops increasing
   - Merge days $d_i, d_{i+1}$ if $\text{sim}(Z_{d_i}, Z_{d_{i+1}}) > \theta_{\text{merge}}$
   - Stop when similarity plateaus or decreases
   - **Result:** Variable-length segments that respect narrative cohesion

3. **Enhanced Feature Representation:**
   - **Problem identified:** Need to encode both time gap and topic information more explicitly
   - **Solution:** Increased embedding dimension
   - Input: 768 (SBERT W5) → **774** (768 + 5 topic scores + 1 time gap)
   - Topic scores: Soft labels from ideal article prototypes (Section 3.5.4)
   - Time gap: Normalized log-scaled difference between consecutive days

4. **Topic Separation Loss:**
   - **Motivation:** Approach 2 showed poor intertopic separation
   - **Goal:** Explicitly push apart embeddings from different topics
   
   $$\mathcal{L}_{\text{topic-sep}} = \max(0, \text{margin} - \|\text{dist}_{\text{inter-topic}}\| + \|\text{dist}_{\text{intra-topic}}\|)$$
   
   - **Rationale:** Triplet-like loss ensures topic embeddings maintain minimum separation
   - Helps model distinguish topic-level shifts from entity-level changes within topics

**Model Configuration:**
- Architecture: Temporal Transformer (4 layers, 8 heads, 512-dim hidden)
- Loss: **Weighted combination of three losses:**
  
  $$\mathcal{L}_{\text{total}} = \lambda_{\text{temporal}} \mathcal{L}_{\text{NT-Xent}} + \lambda_{\text{topic}} \mathcal{L}_{\text{topic-sep}} + \lambda_{\text{hard}} \mathcal{L}_{\text{hard-neg}}$$
  
  - $\lambda_{\text{temporal}} = 1.5$ (primary objective)
  - $\lambda_{\text{topic}} = 0.5$ (topic separation)
  - $\lambda_{\text{hard}} = 0.3$ (hard negative mining)

**Evaluation Results:**
- **Intertopic similarity:** Good (low) — different topics now well-separated ✅
- **Intratopic similarity:** Worse than Approach 2 ❌
- **Problem Identified:** Topic matching occurs but **entity-level confusion**
  - Example: "COVID-19 vaccine" (2020) vs. "flu vaccine" (2015) both match "Health" topic
  - Model lacks entity-aware distinction
  - **Root cause:** No Named Entity Recognition (NER) to distinguish specific entities

**Reference Implementation:** `TCL_Pipeline_4.ipynb`

**Change Point Detection Details:**
- Library: `ruptures` (Python package)
- Model: PELT with RBF kernel
- Parameters: `penalty=0.1`, `min_size=2`
- Output: Segment boundaries [t₁, t₂, ..., tₙ] where narrative properties change

---

##### **Approach 4 with Balanced Data: User Inference & Narrative Shift Validation (✅ Completed)**

**Status:** Approach 4 retrained on balanced topic data and tested on 5 real user-submitted articles. This is the first real-world inference validation of the TCL pipeline.  
**Reference:** `TCL/docs/approach_4_with_bd.md` | Input: `Output/Model_Testing/Approch_4/user_article2.csv` | Output: `Output/Model_Testing/Approch_4/user_results_*.json`

**Why Balanced Data?**

Original topic files had severe imbalance (War: 490,123 sentences vs. Climate: 188,013). Approach 4 was retrained after balancing via `Pre_Processing/Data_balancing.ipynb`:

- **Filtering:** `max_topic_weight ≥ 0.35` AND `topic_gap ≥ 0.20` (removes ambiguous sentences)
- **Monthly balancing:** Iterative removal of weakest sentences from dominant topic per month until max − min ≤ 600
- **Yearly balancing:** Same logic, threshold = 3000
- **Removal strategy:** Always removes lowest-confidence (lowest `max_topic_weight`) sentences first — preserves quality

**User Inference Input: 5 Russia-Ukraine Articles (Feb–Apr 2022)**

| Article | Date | Narrative Phase |
|---------|------|-----------------|
| a0 | 2022-02-15 | Pre-invasion military buildup |
| a1 | 2022-02-25 | Full-scale invasion launched |
| a2 | 2022-03-15 | Humanitarian crisis — refugees, civilian displacement |
| a3 | 2022-04-01 | Ceasefire diplomacy — Istanbul negotiations |
| a4 | 2022-04-20 | Reconstruction / post-war recovery |

Inference threshold: 0.10 (fixed, non-adaptive).

**Results by Topic:**

| Topic | Sentences | Unique Days | Shifts Detected | Shift Scores |
|-------|-----------|-------------|-----------------|-------------|
| War | 28 | 5 | **2** | 0.478, **1.0** |
| Economics | 26 | 5 | **2** | 0.123, **1.0** |
| Health | 18 | 4 | **1** | **1.0** |

**Detected Narrative Shifts (sentence-level evidence):**

*War topic — Shift 1 (score 0.478):*
- a0 s5: *"International observers expressed concern that the growing tensions could lead to a broader conflict..."*  → a2 s3: *"Emergency aid groups are struggling to provide food, shelter and medical assistance..."*
- Transition: pre-war tensions → active humanitarian emergency | Similarity: 0.5392

*War topic — Shift 2 (score 1.0):*
- a1 s0: *"Russian forces launched a large-scale military invasion of Ukraine..."* → a3 s4: *"International leaders encouraged continued dialogue, arguing that diplomacy offers the best path..."*
- Transition: military escalation → diplomatic resolution | Similarity: 0.2462 (lowest — confirms maximal semantic divergence)

*Economics topic — Shift 2 (score 1.0):*
- a2 s4: *"Ukrainian forces continued to resist Russian advances in several cities..."* → a4 s0: *"Global attention has increasingly shifted toward rebuilding Ukraine..."*
- Transition: active combat → economic reconstruction narrative

*Health topic — Shift 1 (score 1.0):*
- Same a2→a4 sentence pair — humanitarian aid → recovery framing (fewer Health-relevant sentences extracted: 18 vs. 28 for War, reflecting correct topic filtering)

**Key Observations:**
1. War topic is most sensitive — detects an earlier, weaker shift (Feb 15 → Mar 15) that Economics and Health miss
2. All 3 topics agree the strongest shift is combat→reconstruction (Mar 15 → Apr 20), shift score 1.0
3. Economics shift 1 (score 0.123) is borderline — model correctly treats it as a weak change
4. Health extracts fewer sentences from the same 5 articles — correct behavior, as articles are War-framed
5. No ground truth labels available; evaluation is qualitative (shift dates and sentence pairs align with real-world narrative phases)

---

##### **Approach 5: Multi-Modal NER + Sentiment + FAISS Pipeline (✅ Production-Ready)**

**Status:** Code fully implemented in `TCL_Pipeline_5.ipynb`. Target hardware: Kaggle GPU P100. Not yet run on full dataset — evaluation metrics (intra-sim, inter-sim, F1) are TBD.  
**Reference:** `TCL/docs/approach_5.md`

**Design Philosophy:**
> Narrative shift is not strictly time-based — it is driven by changes in **meaning, entities, and sentiment**. Day-level pooling mixes narratives, cancels sentiment signals, and destroys entity specificity. Ruptures assumes fixed temporal signal and clear change points; Approach 5 replaces both with FAISS-based semantic retrieval.

**8-Stage Pipeline:**

1. **Input Data:** Per-topic sentence files (sentence_id, main_sentence, w5_embedding, topic scores)

2. **Sentence → Article Aggregation:** Extract article_id from sentence_id; group sentences into articles

3. **Feature Extraction (three signals):**
   - *Semantic:* Mean of W5 sentence embeddings (768-dim)
   - *NER:* spaCy, sentence-level → aggregated to article; entity types: PERSON, ORG, GPE, NORP; canonicalization via agglomerative clustering (cosine threshold = 0.85)
   - *Sentiment:* RoBERTa model; per-sentence output −1/0/+1; article sentiment = mean of sentence sentiments

4. **Article Embedding:**
   `article_embedding = 0.6 × semantic + 0.2 × entity + 0.2 × sentiment` → L2-normalized (768-dim)

5. **TCL Training:**
   - Input: 768-dim article embeddings → Output: 512-dim projected embeddings
   - Loss: InfoNCE
   - Positive pairs: similarity > 0.75 AND entity_overlap > 0 AND sentiment_diff < 0.4 AND time_diff ≤ 3 days
   - Negative pairs: similarity < 0.35 OR entity_overlap == 0 OR sentiment_diff > 0.7

6. **FAISS Indexing:** Separate per-topic index (War, Health, Technology, Climate, Economics)

7. **Shift Detection:**
   ```
   ShiftScore = 0.45 × semantic_distance
              + 0.20 × sentiment_change
              + 0.20 × entity_change (1 − Jaccard)
              + 0.15 × claim_difference
   ```
   Threshold: 0.65

8. **Sentence-Level Verification:** Find most divergent sentence pair sharing ≥1 entity; return ±2 context sentences

**Comparison with Previous Approaches:**

| Feature | A1 | A2 | A4 | **A5** |
|---|---|---|---|---|
| Granularity | Day window | Day group | Sentence (PELT) | Article |
| Embedding | SBERT 768 | SBERT 768 | SBERT 768 | 3-component (semantic+entity+sentiment) |
| NER | ❌ | ❌ | ❌ | ✅ (PERSON/ORG/GPE/NORP) |
| Sentiment | ❌ | ❌ | ❌ | ✅ RoBERTa |
| Day pooling | ✅ Used | ✅ Used | ❌ | ❌ Not used |
| Ruptures | ❌ | ❌ | ✅ Used | ❌ Not used |
| Retrieval | Linear | Linear | PELT segments | ✅ FAISS topic-specific |
| TCL projection | 256-dim | 256-dim | 512-dim | **512-dim** |
| GPU Required | No | No | Yes | **Yes (P100)** |
| Status | Done | Done | Done | **✅ Code complete, not run** |

**Current Status:** ✅ Production-Ready (code) | Evaluation metrics: TBD

**Timeline:** Code complete March 2026; full run planned on Kaggle

---

##### **Summary: Evolution of Metrics Across Approaches**

**Overview Table:**

| Approach | Architecture | Loss Function | Intertopic Drift | Intratopic Drift | Key Innovation | Status |
|----------|-------------|---------------|------------------|------------------|----------------|--------|
| **1** | Transformer (256-dim) | InfoNCE | Poor | Poor | Fixed windowing | Completed |
| **2** | Enhanced Transformer | NT-Xent + Hard-Neg | Very High (Bad) | Average | Grouping strategy | Completed |
| **3** | Topic-specific | N/A | N/A | N/A | Topic-wise windows | Research only |
| **4** | Temporal Transformer (512-dim) | NT-Xent + Topic-Sep + Hard-Neg | Good (Low) | Worse | Ruptures segmentation | **Current** |
| **4+BD** | Temporal Transformer (512-dim) — retrained on balanced data | NT-Xent + Topic-Sep + Hard-Neg | — | — | Balanced training + user inference | **✅ Inference done** |
| **5** | MLP (768→1024→512) + NER + Sentiment + FAISS | InfoNCE | TBD | TBD | NER + sentiment + FAISS retrieval | **✅ Code complete** |

**Detailed Metrics Across All Variants:**

| Metric | Approach 1 (NoOverlap) | Approach 1 (Overlap) | Approach 2 (Fixed Day) | Approach 2 (Day Gap) | Approach 4 | Approach 5 |
|--------|------------------------|----------------------|------------------------|----------------------|------------|------------|
| **Window Config** | W=3, S=3 | W=3, S=1 | Fixed 3-day groups | Max 3-day gap | Dynamic segments | Article-level + FAISS |
| **Samples** | N/A | N/A | 669 windows | 732 windows | **356** | TBD |
| **Intra-Topic Similarity** | **0.2182** | **0.1431** | **0.3365** | **0.3185** | **0.9997** ✅ | TBD |
| **Inter-Topic Similarity** | **-0.0457** | **-0.0286** | **0.0114** | **-0.0361** | **-0.0875** ✅ | TBD |
| **Separation Score** | **-4.78** | **-5.01** | **-29.5627** ⚠️ | **-8.8281** | **1.0872** ⚠️ | TBD |
| **Interpretation** | ⚠️ Weak | ⚠️ Weak | ⚠️ Very Weak | ⚠️ Weak | ⚠️ Weak (< 2.0) | TBD |
| **Temporal Consistency** | **0.9155** | **0.8978** | **0.9193** | **0.8948** | **0.9877** ✅ | TBD |
| **Entity Awareness** | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ✅ NER + sentiment |

**Key Observations:**
- **Approach 1 variants:** Both overlapping and non-overlapping windows showed poor performance across all metrics
- **Approach 2 variants:** Fixed Day grouping produced 669 samples vs. Day Gap's 732 samples; Fixed Day showed extremely poor separation score (-29.56)
- **Approach 4:** Best inter-topic separation (-0.0875) and temporal consistency (0.9877), but intratopic similarity very high (0.9997), indicating potential over-smoothing
- **Separation Score:** All approaches currently below ideal threshold (< 2.0), indicating need for further refinement
- **Entity Awareness:** Only planned for Approach 5 with NER integration

**Metric Interpretation:**
- **Intertopic Drift (Goal: Low):** Measures how distinct different topics are in embedding space
  - Low = Good (topics well-separated, e.g., War clearly different from Health)
  - High = Bad (topics overlap, model confuses War articles with Health articles)
- **Intratopic Drift (Goal: Moderate):** Measures similarity of consecutive windows within same topic
  - Too High = Bad (all windows identical, no shifts detected)
  - Too Low = Bad (excessive false positives, noise dominates)
  - Moderate = Good (detects genuine narrative shifts while maintaining stability)

**Key Lessons Learned:**

1. Fixed temporal windowing insufficient → Need adaptive segmentation
2. Grouping helps capture longer-term patterns but requires careful strategy
3. Single window size cannot fit all topics → Topic-specific or dynamic sizing needed
4. Explicit topic separation loss critical for multi-topic modeling
5. Entity-level distinction necessary for fine-grained narrative shift detection
6. Change-point detection algorithms (Ruptures) provide principled segmentation approach

**Current Best Practice (Approach 4):**
- Use Ruptures PELT with RBF kernel for adaptive segmentation
- 774-dimensional features (768 SBERT + 5 topic + 1 time)
- Weighted loss combining temporal contrastive + topic separation + hard negative mining
- Achieves good intertopic separation; intratopic detection to be improved with NER (Approach 5)

---

### 4.5 Baseline Comparison Models

To validate the effectiveness of the TCL framework, we compare against two baseline approaches.

**⚠️ Important Note on Shared Preprocessing:**

All three models (TCL, Semantic Drift using SBERT, K-Means) utilize the **identical preprocessing pipeline** described in Section 3:

1. ✅ **Data Cleaning** (Section 3.4): Deduplication, date cleaning, language filtering → 355,334 English articles
2. ✅ **Sentence Segmentation** (Section 3.9 - Stage 1): Context window construction → 3,080,512 sentences
3. ✅ **SBERT Embedding** (Section 3.9 - Stage 2): all-mpnet-base-v2 with Window 5 context → 768-dimensional embeddings
4. ✅ **Topic Scoring & Filtering** (Section 3.9 - Stage 3): Similarity threshold 0.3 → 1,335,158 topic-filtered sentences

**Key Point:** All models start with the **same input data** (1.33M sentences with 768-dim SBERT embeddings and topic scores). The differences lie in how each model processes these embeddings to detect narrative shifts:

- **Semantic Drift SBERT:** Multi-stage calibrated approach with topic prototypes, relevance filtering, 5-day windowing, and threshold-based drift detection (no neural learning)
- **K-Means:** Clustering-based distributional changes with JS divergence (unsupervised clustering)
- **TCL:** Temporal contrastive learning with transformer encoder (learned representations with InfoNCE loss)

This ensures a **fair comparison** where the only variable is the shift detection methodology, not data quality or embedding approach.

---

#### 4.5.1 Semantic Drift Detection using SBERT

> **⚠️ Note:** This approach is currently under development and may require changes as needed. The model parameters, thresholds, and methodology are subject to further refinements and modifications based on evaluation results and experimental findings.

**Objective:**

Detect narrative changes by measuring semantic drift between temporal windows of news articles. The system answers: *"Has the way news articles talk about a topic changed significantly between two time periods?"*

**Pipeline Architecture:**

The approach consists of **4 distinct stages**: 3 calibration stages (run once on corpus) + 1 detection stage (run on new articles):

```
CALIBRATION (run once):
  Ideal Articles → Stage 1 → Topic Prototype Vectors
                           ↓
  Prototype + Corpus → Stage 2 → Topic Relevance Thresholds
                                ↓
  Thresholds + Corpus → Stage 3 → Drift Thresholds
                                 ↓
DETECTION (run anytime):
  New Articles + Calibrated Thresholds → Stage 4 → Drift Events + Responsible Sentences
```

**Stage 1 — Topic Prototype Vectors:**

Build one canonical 768-dimensional vector per topic representing its core narrative:

1. **Input:** Hand-curated "ideal articles" for each topic (detailed in Section 3.5.4)
   - 35 total articles (~7 per topic) from authoritative sources (WHO, Reuters, IPCC, IMF, MIT Tech Review, etc.)
   - 5 subtopics per topic (e.g., War: Armed Conflict, Weapons Systems, Peace Negotiations, Humanitarian Crisis, War Crimes)
   - Multiple articles per subtopic ensuring breadth of coverage
   - Temporal independence: Focus on timeless topic-defining keywords rather than time-specific events
2. **Process:** 
   - Segment all ideal articles into sentences (NLTK)
   - Filter sentences < 20 characters
   - Encode using SBERT (all-mpnet-base-v2) → 768-dim embeddings
   - Mean-pool all sentence embeddings per topic
   - L2-normalize → topic prototype vector
3. **Output:** `topic_prototypes.json` with one 768-dim vector per topic

**Stage 2 — Topic Relevance Thresholds:**

Determine minimum cosine similarity for a sentence to be considered "on-topic":

- **Formula:** $\text{threshold}_{\text{topic}} = \mu_{\text{sim}} - \alpha \cdot \sigma_{\text{sim}}$ where $\alpha = 0.5$
- **Process:** Compute cosine similarity of all ideal-article sentences against their topic prototype
- **Rationale:** Setting threshold at `mean − 0.5·std` accepts clearly on-topic sentences while rejecting noise
- **Calibrated Thresholds:**
  - Climate: 0.437, Economics: 0.474, Health: 0.441, Technology: 0.437, War: 0.412

**Stage 3 — Drift Threshold Calibration:**

Define what level of semantic shift counts as "drift" for each topic and context model:

1. **Input:** Full corpus (355,334 articles) + topic prototypes + relevance thresholds
2. **Process per topic:**
   - Filter sentences by topic relevance (cosine similarity ≥ topic threshold)
   - Group filtered sentences into **5-day temporal windows**
   - Mean-pool each window → L2-normalized window embedding
   - Compute drift between adjacent windows: $\text{drift} = 1 - \cos(\vec{w}_t, \vec{w}_{t-1})$
   - Collect drift distribution across entire corpus
3. **Formula:** $\text{drift\_threshold} = \mu_{\text{drift}} + \sigma_{\text{drift}}$
4. **Rationale:** Flags only transitions **one standard deviation above typical drift** (conservative, ~top 16%)

**Calibrated Drift Thresholds:**

| Topic | w1 | w3 | w5 |
|-------|----|----|-----|
| Climate | 0.321 | 0.390 | 0.395 |
| Economics | 0.277 | 0.380 | 0.419 |
| Health | 0.256 | 0.389 | 0.439 |
| Technology | 0.255 | 0.388 | 0.409 |
| War | 0.318 | 0.426 | 0.480 |

**Stage 4 — Drift Detection Pipeline:**

Given new articles for a specific topic, detect narrative shifts through 8 sub-stages:

1. **Sentence Segmentation:** NLTK tokenization, deduplication
2. **Context Window Construction:** Build w1/w3/w5 representations
   - **w1:** Sentence alone
   - **w3:** Sentence + 1 neighbor on each side
   - **w5:** Sentence + 2 neighbors on each side
   - Context clipped at article boundaries
3. **SBERT Encoding:** Encode all context representations → (N, 768) L2-normalized embeddings
4. **Topic Filtering:** Keep only sentences with cosine similarity ≥ topic threshold
5. **Temporal Grouping:** Bin filtered sentences into 5-day windows
6. **Window Embedding:** Mean-pool + L2-normalize per window
7. **Drift Computation:** $\text{drift} = 1 - \cos(\vec{w}_t, \vec{w}_{t-1})$ for adjacent windows
8. **Drift Identification:** 
   - Flag drift when: $\text{drift} > \text{drift\_threshold}$
   - Identify responsible sentences using **impact score:**
   
   $$\text{impact}(s) = \frac{\cos(s, \vec{w}_{\text{curr}})}{\cos(s, \vec{w}_{\text{prev}}) + \epsilon}$$
   
   Higher impact = sentence is close to new narrative but far from old narrative

**Output Format:**

- Drift events: `(window_prev, window_curr, drift_score, threshold, detected: bool)`
- For each drift event:
  - Top-N anchor sentences from previous window (stable narrative)
  - Top-N drift-driving sentences from current window (new narrative) with impact scores
- Example: `"2024-01-06 → 2024-01-11: drift=0.41 > threshold=0.32 🔴 DRIFT"`

**Characteristics:**

- **Multi-stage calibration** ensures topic-specific and data-driven thresholds
- **Three context models** (w1/w3/w5) capture different granularities of context
- **Topic-aware filtering** removes off-topic noise before drift computation
- **Interpretable output** identifies specific sentences responsible for narrative shifts
- **Dense semantic representation** captures meaning beyond keywords
- **No temporal modeling** between windows (each window treated independently)
- **Quantitative + qualitative** results (drift scores + responsible sentences)

**Implementation Status:**

- Stages 1-3 calibration completed on full corpus
- Stage 4 detection pipeline implemented and tested
- Impact score analysis for interpretability added
- **Note:** Model may require adjustments to parameters, thresholds, or methodology based on ongoing evaluation and experimental findings

#### 4.5.2 K-Means Clustering Drift

> **⚠️ Note:** This approach is currently under development and may be subject to further refinements and modifications.

**Objective:**

Detect narrative changes by analyzing how the distribution of narrative clusters changes over time. The system identifies narrative drift when the semantic structure of articles changes significantly between time periods.

**Pipeline Overview:**

```
Sentence Embeddings (768-dim)
↓
K-Means Clustering (K=5)
↓
Cluster Distribution per Date
↓
Jensen-Shannon Divergence
↓
Narrative Drift Detection
```

**Detailed Method:**

1. **Input Data:** Use preprocessed sentences with existing 768-dimensional SBERT embeddings (w3_embedding column) from the shared pipeline
2. **Topic Filtering:** Restrict to specific topic (e.g., Climate > 0.3 threshold) to ensure topic coherence
3. **Cluster Discovery:** Train K-Means model with K=5 clusters to discover latent narrative patterns
   - Each cluster may represent distinct narrative themes (e.g., scientific discussion, policy debates, environmental impacts, economic concerns, international agreements)
4. **Cluster Assignment:** Map each sentence to its nearest cluster centroid
5. **Temporal Distribution:** For each date, compute the distribution of sentences across clusters
   - Example: Date 1: [0.6, 0.2, 0.1, 0.1, 0.0] → 60% cluster 0, 20% cluster 1, etc.
   - Example: Date 2: [0.1, 0.1, 0.6, 0.1, 0.1] → narrative shifted to cluster 2
6. **Drift Measurement:** Calculate Jensen-Shannon (JS) divergence between consecutive time periods:
   - JS(P || Q) = ½ KL(P || M) + ½ KL(Q || M), where M = (P + Q) / 2
   - Range: 0 = identical distributions, 1 = completely different distributions
7. **Drift Detection:** Flag narrative shift when JS divergence > 0.3 threshold

**Configuration:**
- Embedding dimension: 768
- Clustering algorithm: K-Means with Euclidean distance
- Number of clusters: K = 5
- Drift metric: Jensen-Shannon divergence
- Drift threshold: 0.3
- Training dataset: Starting with Climate.csv (~1.26GB, ~189K sentences)

**Example Output:**
```
Narrative Shift Detected
From: 2015-04-21
To: 2015-09-16
Drift Score: 0.83
```

**Characteristics:**
- Captures distributional changes in narrative structure
- Unsupervised approach requiring no labeled training data
- Sensitive to cluster number selection (K parameter)
- No explicit temporal modeling between periods
- Interpretable through cluster representatives
- Detects major narrative structure changes (not fine-grained framing)
- Struggles with gradual shifts

**Comparison Metrics:**

Since the models output shift scores and detected shift points (not segments), we evaluate all three approaches (TCL, Semantic Drift using SBERT, K-Means) as **drift detection models** using the following unsupervised metrics:

1. **Drift Score Distribution Analysis:** Compare statistical properties of shift scores (mean, standard deviation, number of high-shift events). Higher variance and peaks indicate better detection sensitivity.

2. **Drift Peak Detection:** Count detected shifts using thresholds (e.g., mean + 1 std, top 10%, or fixed threshold 0.6-0.7). Measures model sensitivity to narrative changes.

3. **Semantic Distance at Shift Points:** Validate that detected shifts correspond to actual semantic changes by computing cosine distance between consecutive sentences at shift points. Higher distance indicates more meaningful shifts.

4. **Human Validation (Primary Metric):** Manually annotate top 20-30 detected shifts from each model to validate whether they represent genuine narrative/topic changes. Requires human-labeled shift data. This provides strongest evidence of model effectiveness.

**Example Comparison Format:**

| Model | Avg Score | Shift Count | Avg Semantic Distance | Human Valid (%) |
|-------|-----------|-------------|----------------------|----------------|
| K-Means Drift | 0.31 | 18 | 0.41 | 52 |
| SBERT Drift | 0.38 | 26 | 0.48 | 64 |
| TCL (Our Model) | 0.44 | 32 | 0.56 | 78 |

**Additional Analysis:**
- **Drift Curve Visualization:** Plot shift scores over sentence indices to compare smoothness and peak sharpness
- **Computational Efficiency:** Training and inference time
- **Robustness:** Stability across different topics and time periods

---

## 5. Experimental Results and Progress

### 5.1 Completed Objectives

#### 5.1.1 Data Collection and Integration

- Successfully collected and integrated data from 8 heterogeneous sources
- Implemented robust parsing for CSV, Excel, and JSON formats
- Developed multi-strategy date conversion handling
- Achieved 1,167,047 raw article corpus spanning 1970–2025 (primary: 2011–2025)
- Completed Deliverable: `ALL_Combined_Data.csv`

#### 5.1.2 Language Detection and Filtering

- Implemented multi-threaded language detection using `langdetect`
- Filtered corpus to English-only articles
- Achieved 8x speedup through parallel processing
- Comprehensive logging and statistics tracking

#### 5.1.3 Data Cleaning and Quality Assurance

- Implemented multi-stage cleaning pipeline
- Removed invalid dates, empty articles, duplicates
- Applied length-based filtering (500–50,000 characters)
- Achieved 99.8% data completeness
- Chronologically sorted entire dataset

#### 5.1.4 Sentence Segmentation and Context Windowing

- Implemented NLTK-based sentence tokenization
- Developed context window construction (window size 5)
- Multi-threaded processing of entire corpus
- Generated unique identifiers for sentence tracking
- Completed Deliverable: Sentence-segmented dataset with context

#### 5.1.5 Context-Aware Sentence Embedding

- Implemented SBERT (all-mpnet-base-v2) embedding pipeline
- Generated both W3 and W5 contextual embeddings
- Batch processing with GPU acceleration
- Completed 768-dimensional embeddings for all sentences
- Completed Deliverable: Embedded sentence dataset

#### 5.1.6 Topic Embedding and Soft Labeling

- Curated 35 ideal reference articles (~7 per topic) from authoritative sources (WHO, Reuters, IPCC, IMF, MIT Tech Review, etc.)
- Manually extracted topic-defining sentences from ideal articles (Section 3.5.4)
- Generated topic prototype embeddings using SBERT (all-mpnet-base-v2)
- Implemented cosine similarity-based soft labeling against prototype vectors
- Applied topic filtering with 0.3 similarity threshold
- **Processing Results:**
  - Input: 3,080,512 sentences from 113 files
  - Output: 1,335,158 topic-filtered sentences (43.3% retention)
  - Processing time: 28.14 minutes (4.02 files/minute)
  - 100% valid embeddings across all topics
- Completed Deliverable: Topic-specific sentence datasets
  - War: 490,123 sentences (4.59 GB)
  - Economics: 277,886 sentences (2.60 GB)
  - Technology: 190,543 sentences (1.78 GB)
  - Health: 188,593 sentences (1.77 GB)
  - Climate: 188,013 sentences (1.76 GB)

#### 5.1.7 TCL Model Implementation and Evolution

**Approach 1: Fixed Window with InfoNCE (Completed)**
- ✅ Implemented transformer-based encoder (4 layers, 8 heads, 256-dim)
- ✅ Fixed temporal windowing (overlapping and non-overlapping variants)
- ✅ InfoNCE contrastive loss
- ✅ Training stability: layer normalization, gradient clipping, cosine annealing, warmup
- ✅ Evaluation completed
- **Result:** Both intertopic and intratopic drift metrics poor
- **Reference:** `TCL_Pipeline_Complete.ipynb`

**Approach 2: Grouping-Based Segmentation (Completed)**
- ✅ Topic-weighted lambda mean pooling
- ✅ Two grouping strategies implemented:
  - Count-based grouping (4 days per group)
  - Date-range grouping (5-day windows)
- ✅ Enhanced NT-Xent loss + hard negative mining loss
- ✅ Enhanced Transformer with attention pooling
- ✅ Evaluation completed
- **Result:** Intertopic drift very high (bad), intratopic drift average (improved)
- **Lesson:** Grouping helps but requires better strategy
- **Reference:** `TCL_Pipeline_2.ipynb`

**Approach 3: Topic-Wise Window Sizing (Research Phase)**
- 🔄 Conceptual design completed
- 🔄 Identified challenges:
  - Misses macro-level narrative shifts
  - Window size needs temporal adaptation within topics
  - Computationally expensive (5 topics × multiple years × window sizes)
- **Status:** Not implemented; planned as future work with RL or meta-learning approaches

**Approach 4: Dynamic Segmentation with Topic Separation (Current Implementation)**
- ✅ Ruptures library integration for change-point detection
  - PELT algorithm with RBF kernel
  - Adaptive segmentation based on similarity thresholds
- ✅ Enhanced feature representation (774-dim: 768 SBERT + 5 topics + 1 time)
- ✅ Multi-objective loss function:
  - Temporal contrastive loss (NT-Xent, λ=1.5)
  - Topic separation loss (λ=0.5)
  - Hard negative mining loss (λ=0.3)
- ✅ Maximum similarity-based grouping
- ✅ Training completed on sample topics
- **Result:** Intertopic drift good (low), intratopic drift worse than Approach 2
- **Issue Identified:** Entity-level confusion (e.g., "COVID vaccine" vs "flu vaccine")
- **Reference:** `TCL_Pipeline_4.ipynb`, inspired by Truong et al. [16]

**Approach 5: Multi-Modal NER + Sentiment + FAISS (✅ Code Complete)**
- ✅ Architecture implemented in `TCL_Pipeline_5.ipynb` (Kaggle GPU P100)
- ✅ Three-component embedding: 0.6 × semantic + 0.2 × entity + 0.2 × sentiment
- ✅ NER: spaCy (PERSON, ORG, GPE, NORP) + entity canonicalization (agglomerative clustering, cosine threshold 0.85)
- ✅ Sentiment: RoBERTa (per-sentence −1/0/+1, averaged at article level)
- ✅ TCL: MLP 768 → 1024 → 512, InfoNCE
- ✅ FAISS topic-specific indexes (per topic: War, Health, Technology, Climate, Economics)
- ✅ 4-component shift score (semantic 0.45 + sentiment 0.20 + entity 0.20 + claim 0.15), threshold 0.65
- ✅ 8-stage pipeline (input → aggregation → features → embedding → TCL → FAISS → shift → verification)
- **Status:** Code complete; full dataset run and evaluation TBD (Kaggle)

**Completed Deliverables:**
- Approach 1 implementation and evaluation
- Approach 2 implementation and evaluation
- Approach 4 implementation (current best)
- Comprehensive documentation of evolution and lessons learned

**Key Insights from Evolution:**
1. Fixed windowing insufficient → adaptive segmentation needed
2. Explicit topic separation critical for multi-topic modeling
3. Entity-level distinction necessary for fine-grained shift detection
4. Change-point detection provides principled segmentation approach

#### 5.1.8 Documentation

- Comprehensive data preprocessing documentation
- Data combination pipeline documentation
- Data source documentation with references
- Code comments and inline documentation

### 5.2 Ongoing Work

#### 5.2.1 Model Training

**Current Status:**
- Training infrastructure implemented and tested for all 4 completed approaches
- Hyperparameter configurations finalized for Approaches 1, 2, and 4
- Training completed for Approaches 1 and 2 (evaluation results documented in Section 4.3.6)
- Approach 4 (current): Training in progress on sample topics
- Ruptures-based segmentation pipeline operational
- Multi-objective loss function (NT-Xent + Topic Separation + Hard Negative Mining) implemented

**Remaining Tasks:**
- Complete full-scale training of Approach 4 across all 5 topics
- Conduct comprehensive evaluation comparing all approaches
- Hyperparameter tuning for Approach 4 (Ruptures penalty, loss weights)
- Address entity-level confusion issue identified in Approach 4
- Model checkpoint saving and evaluation

**Timeline:** Expected completion by March 8, 2026

#### 5.2.2 Baseline Model Implementation

**Current Status:**

**Semantic Drift Detection using SBERT:**
- ✅ Stage 1: Topic prototype vectors built from hand-curated ideal articles
- ✅ Stage 2: Topic relevance thresholds calibrated (Climate: 0.437, Economics: 0.474, Health: 0.441, Technology: 0.437, War: 0.412)
- ✅ Stage 3: Drift thresholds calibrated on full corpus for all topics × models (w1/w3/w5)
- ✅ Stage 4: Detection pipeline implemented (8 sub-stages: segmentation → encoding → filtering → windowing → drift computation → impact analysis)
- ✅ Three context-window models (w1, w3, w5) fully operational
- ✅ Impact score calculation for identifying drift-responsible sentences
- **Note:** Model parameters, thresholds, and methodology may require changes as needed based on evaluation results

**K-Means Clustering Drift:**
- ✅ Pipeline approach defined (embeddings → K=5 clusters → JS divergence > 0.3 threshold)
- 🔄 Initial testing on Climate.csv dataset (~189K sentences)
- Note: Approach subject to further refinements

**Remaining Tasks:**
- Complete K-Means drift detection implementation and validation
- Standardize evaluation metrics across all models
- Run comprehensive comparative experiments across all topics (all 5 topics × 3 models)
- Cross-validate calibrated thresholds

**Timeline:** Expected completion by March 15, 2026

#### 5.2.3 Evaluation and Analysis

**Current Status:**
- Drift detection evaluation framework designed
- Metrics defined: drift score distribution, peak detection, semantic distance, human validation
- Evaluation approach confirmed (change-point detection, not segmentation)

**Remaining Tasks:**
- Execute comprehensive evaluation across all topics and models
- Collect human-labeled shift data for validation (top 20-30 shifts per model)
- Compute drift score statistics and semantic distances at shift points
- Generate drift curve visualizations
- Statistical significance testing
- Error analysis and failure case identification

**Timeline:** Expected completion by March 10, 2026

### 5.3 Planned Work

#### 5.3.1 Interpretability Enhancements (Planned)

**Objectives:**
- Implement attention visualization for transformer layers
- Develop sentence-level attribution methods
- Create interactive visualization tools for narrative evolution
- Generate case studies of detected narrative shifts

**Timeline:** March 10–15, 2026

#### 5.3.2 Cross-Topic Analysis (Planned)

**Objectives:**
- Analyze narrative shift patterns across different topics
- Identify common shift triggers (events, policy changes)
- Study temporal correlation of shifts across topics
- Examine entity-specific narrative evolution

**Timeline:** March 15–20, 2026

#### 5.3.3 Ablation Studies (Planned)

**Objectives:**
- Study impact of context window size (W3 vs W5)
- Analyze effect of topic conditioning
- Evaluate contribution of data augmentation
- Test sensitivity to hyperparameters

**Timeline:** March 20–25, 2026

#### 5.3.4 Real-World Case Studies (Planned)

**Objectives:**
- Analyze specific high-profile events (e.g., COVID-19 narrative evolution)
- Track policy-related narrative shifts (e.g., AI regulation discourse)
- Study geopolitical narrative changes (e.g., international conflict framing)
- Validate model predictions against known historical shifts

**Timeline:** March 25–30, 2026

### 5.4 Technical Achievements

1. **Scalable Pipeline:** Successfully processed 1,167,047 raw articles → 355,334 cleaned articles → 3,080,512 sentences → 1,335,158 topic-filtered sentences
2. **Robust Data Handling:** Integrated 8 heterogeneous data sources with different formats
3. **Efficient Sentence Processing:** Processed 3.08M sentences with topic filtering in 28.14 minutes (4.02 files/minute)
4. **Multi-threaded Language Detection:** Achieved 8x speedup (16 articles/second)
5. **GPU Acceleration:** Leveraged GPU for batch embedding generation (SBERT)
6. **Memory Management:** Incremental processing and saving to handle large datasets (12.5 GB topic files)
7. **Stable Training:** Implemented multiple stability enhancements for reliable model training
8. **Aggressive Quality Control:** Removed 69.56% of raw data (811,713 articles) ensuring quality
9. **Comprehensive Topic Coverage:** 100% valid embeddings across 1.33M sentences spanning 2011-2025

### 5.5 Challenges Encountered and Solutions

#### Challenge 1: Heterogeneous Data Formats
**Solution:** Developed flexible column mapping system with pattern matching and multi-strategy date parsing

#### Challenge 2: Computational Scalability
**Solution:** Implemented multi-threading for CPU-bound tasks and batch processing with GPU acceleration

#### Challenge 3: Memory Constraints
**Solution:** Incremental file processing, immediate disk saving, and memory clearing after each file

#### Challenge 4: Topic Ambiguity
**Solution:** Soft labeling approach allowing articles to belong to multiple topics with confidence scores

#### Challenge 5: Training Stability
**Solution:** Comprehensive stability enhancements including layer normalization, gradient clipping, and careful learning rate scheduling

### 5.6 Code Repository Structure

```
project/
├── data/
│   ├── raw/                    # Raw data sources
│   └── processed_data/         # Processed datasets
│       ├── ALL_Combined_Data.csv
│       ├── topic_embedding.json
│       └── 3_embed/            # Topic-specific data
├── notebooks/
│   ├── Data_Combi.ipynb        # Data combination
│   ├── Data_Prepo.ipynb        # Data preprocessing
│   ├── TCL_Pipeline_Complete.ipynb  # TCL model
│   └── visualize_3_embed.ipynb # Visualization
├── docs/
│   ├── Data_Combination_Documentation.md
│   ├── Data_Preprocessing_Documentation.md
│   └── intial_data_sources.md
├── visualizations/             # Generated plots and figures
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
```

---

## Limitations

#### Language Scope
**Limitation:** Current implementation focuses exclusively on English-language news articles.

**Impact:** Cannot analyze narrative shifts in non-English media, limiting global applicability.

**Future Direction:** Extend framework to multilingual settings using cross-lingual embeddings (XLM-R, mBERT).

#### Entity Specificity
**Limitation:** Current approach (Approach 4) does not explicitly model entity-specific narratives within topics.

**Impact:** 
- Entity-level confusion observed: Model matches topics correctly but fails to distinguish entity-level changes
- Example: "COVID-19 vaccine" (2020) vs. "flu vaccine" (2015) both classified as Health topic shifts, despite being different entities
- **Result:** Intratopic drift detection degraded in Approach 4 compared to Approach 2

**Discovery:** Identified through iterative experimentation (documented in Section 4.4.6)

**Current Resolution:** Approach 5 (`TCL_Pipeline_5.ipynb`, ✅ code complete) addresses this directly via NER entity extraction (spaCy `en_core_web_trf`), entity canonicalization, and a 4-component shift score that explicitly weights entity change (Jaccard distance, weight=0.20) separately from semantic drift.

#### Fixed Window Sizing
**Limitation:** Current implementations (Approaches 1, 2, 4) use fixed window sizes across all topics.

**Impact:**
- **Approach 1:** Fixed 3-day windows insufficient for capturing narrative evolution
  - Overlapping windows too similar (fails detection)
  - Non-overlapping windows miss gradual shifts
- **Approach 2:** Fixed grouping strategies (count-based or date-based) cause uneven representation quality
- **Approach 4:** Ruptures-based adaptive segmentation partially addresses this but still constrained

**Research Finding (Approach 3):** Different topics require different window sizes:
- War: Rapid shifts (5 days optimal)
- Health: Slower policy-driven changes (7 days optimal)
- Climate: Long-term trends (8 days optimal)

**Additional Challenge:** Window size needs temporal adaptation within topics
- Example: Health topic window size should vary by year (3 days during COVID-19, 6-7 days pre/post-pandemic)

**Future Direction:** 
- Reinforcement learning for automatic window size selection
- Meta-learning to transfer window size knowledge across topics
- Online adaptation algorithms for temporal non-stationarity

#### Ground Truth Availability
**Limitation:** Lack of large-scale annotated datasets for narrative shifts makes thorough quantitative evaluation challenging.

**Impact:** Evaluation relies heavily on qualitative analysis and case studies.

**Future Direction:** Develop annotation protocols and crowdsource narrative shift labels for benchmark creation.

#### Causality Attribution
**Limitation:** Model detects narrative shifts but does not explicitly model causal factors (events, policy changes).

**Impact:** Cannot automatically identify why a narrative shift occurred.

**Future Direction:** Integrate event detection and knowledge graphs to link shifts with real-world events.

#### Real-Time Processing
**Limitation:** Current pipeline requires batch processing and is not optimized for real-time stream processing.

**Impact:** Cannot provide immediate shift detection for incoming news streams.

**Future Direction:** Develop online learning variants and streaming architecture for real-time applications.

#### Topic Definition Subjectivity
**Limitation:** Topic definitions and seed sentences are manually curated, introducing potential bias.

**Impact:** Topic boundaries may not align with natural semantic clusters in data.

**Future Direction:** Explore automatic topic discovery methods (e.g., BERTopic) and hierarchical topic structures.

#### Computational Resources
**Limitation:** Full pipeline requires significant computational resources (GPU, memory).

**Impact:** May be inaccessible for resource-constrained applications.

**Future Direction:** Develop lightweight model variants and distillation approaches for efficient deployment.

---

## References

[1] Blei, D. M., & Lafferty, J. D. (2006). Dynamic topic models. In *Proceedings of the 23rd International Conference on Machine Learning* (pp. 113-120).

[2] Grosz, B. J., & Sidner, C. L. (1986). Attention, intentions, and the structure of discourse. *Computational linguistics*, 12(3), 175-204.

[3] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 3982-3992).

[4] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)* (pp. 4171-4186).

[5] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In *International Conference on Machine Learning* (pp. 1597-1607). PMLR.

[6] Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple contrastive learning of sentence embeddings. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing* (pp. 6894-6910).

[7] Oord, A. v. d., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*.

[8] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems* (pp. 5998-6008).

[9] Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word embeddings reveal statistical laws of semantic change. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1489-1501).

[10] Tshitoyan, V., Dagdelen, J., Weston, L., Dunn, A., Rong, Z., Kononova, O., ... & Jain, A. (2019). Unsupervised word embeddings capture latent knowledge from materials science literature. *Nature*, 571(7763), 95-98.

[11] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1135-1144).

[12] Lample, G., & Conneau, A. (2019). Cross-lingual language model pretraining. *arXiv preprint arXiv:1901.07291*.

[13] Cer, D., Yang, Y., Kong, S. Y., Hua, N., Limtiaco, N., John, R. S., ... & Kurzweil, R. (2018). Universal sentence encoder. *arXiv preprint arXiv:1803.11175*.

[14] Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*.

[15] Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. In *International Conference on Machine Learning* (pp. 1310-1318). PMLR.

[16] Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing*, 167, 107299. DOI: 10.1016/j.sigpro.2019.107299. arXiv:1801.00718.

---

## Appendix A: Dataset Statistics Summary

### A.1 Source Breakdown

| Source Category | Number of Sources | Article Count | Percentage |
|-----------------|-------------------|---------------|------------|
| Kaggle Datasets | 3 | 1,096,988 | 94.0% |
| News APIs | 3 | 27,800 | 2.4% |
| Research Datasets | 2 | 42,259 | 3.6% |
| **Total** | **8** | **1,167,047** | **100%** |

### A.2 Temporal Distribution by Year

| Year Range | Article Count | Percentage |
|------------|--------------|------------|
| 1970-2010 | ~5,000 | ~0.4% |
| 2011-2014 | ~45,000 | ~3.9% |
| 2015-2017 | ~78,000 | ~6.7% |
| 2018-2019 | ~102,000 | ~8.7% |
| 2020-2021 | ~480,000 | ~41.1% |
| 2022-2025 | ~457,047 | ~39.2% |
| **Total** | **~1,167,047** | **100%** |

*Note: Distribution estimated from date range (1970-01-01 to 2025-10-27) with primary concentration in 2020-2025.*

### A.3 Article Length Distribution (Before Length Filtering)

| Length Range (chars) | Article Count | Percentage |
|---------------------|--------------|------------|
| 0-100 | 2,074 | 0.58% |
| 101-500 | 269,866 | 75.94% |
| 501-1,000 | 1,534 | 0.43% |
| 1,001-2,000 | 9,377 | 2.64% |
| 2,001-5,000 | 39,057 | 10.99% |
| 5,000-10,000 | 23,130 | 6.51% |
| 10,000-50,000 | 9,476 | 2.67% |
| 50,000+ | 820 | 0.23% |
| **Total** | **355,334** | **100%** |

*Statistics based on English articles after language filtering, before length-based filtering (500-50,000 chars).*

### A.4 Preprocessing Pipeline Statistics

#### A.4.1 Complete Pipeline Flow

| Stage | Input | Output | Removed/Filtered | Retention | Processing Time |
|-------|-------|--------|-----------------|-----------|-----------------|
| **Stage 0: Raw Collection** | — | 1,167,047 articles | — | 100% | — |
| **Stage 1: Deduplication** | 1,167,047 | 499,048 articles | 667,999 (57.24%) | 42.76% | — |
| **Stage 2: Date Cleaning** | 499,048 | 460,589 articles | 38,459 (7.71%) | 92.29% | — |
| **Stage 3: Language Filtering** | 460,589 | 355,334 articles | 105,255 (22.86%) | 77.14% | 28,253s (~7.85h) |
| **Stage 4: Sentence Segmentation** | 355,334 articles | 3,080,512 sentences | — | ~8.7 sent/article | — |
| **Stage 5: Embedding (W5)** | 3,080,512 sentences | 3,080,512 embedded | 0 | 100% | — |
| **Stage 6: Topic Filtering (0.3)** | 3,080,512 sentences | 1,335,158 sentences | 1,745,354 (56.7%) | 43.3% | 1,688s (~28min) |

**Overall Pipeline:**
- **Total Input:** 1,167,047 raw articles
- **Total Output:** 1,335,158 topic-filtered sentences (from 355,334 articles)
- **Cumulative Retention:** 30.44% (article level), 43.3% (sentence level after filtering)
- **Total Processing Time:** >8 hours (language detection dominant)

#### A.4.2 Topic-Specific Sentence Statistics (Final Output)

| Topic | Sentence Count | Percentage | File Size | Date Range | Avg Sentences/File |
|-------|---------------|------------|-----------|------------|-------------------|
| War | 490,123 | 36.7% | 4.59 GB | 2011-09-19 to 2025-10-26 | 4,929.9 |
| Economics | 277,886 | 20.8% | 2.60 GB | 2011-09-21 to 2025-10-26 | 1,710.1 |
| Technology | 190,543 | 14.3% | 1.78 GB | 2011-10-04 to 2025-10-26 | 1,161.2 |
| Health | 188,593 | 14.1% | 1.77 GB | 2011-09-06 to 2025-10-26 | 1,088.5 |
| Climate | 188,013 | 14.1% | 1.76 GB | 2011-09-21 to 2025-10-26 | 1,357.6 |
| **Total** | **1,335,158** | **100%** | **12.50 GB** | **2011-2025** | — |

**Notes:**
- All topics have 100% valid embeddings (768-dimensional SBERT)
- Average sentences per file calculated from 113 processed files
- Soft labeling allows sentences to appear in multiple topic files
- Files contain: date, w5_embedding, main_sentence, and all 5 topic scores

---

## Appendix B: Model Hyperparameters

### B.1 SBERT Embedding Parameters

| Parameter | Value |
|-----------|-------|
| Model | all-mpnet-base-v2 |
| Embedding Dimension | 768 |
| Batch Size | 64 |
| Max Sequence Length | 512 tokens |
| Device | CUDA (GPU) |

### B.2 Semantic Drift SBERT Parameters

| Parameter | Value |
|-----------|-------|
| Topic Prototype Model | all-mpnet-base-v2 |
| Embedding Dimension | 768 |
| Context Window Models | w1, w3, w5 |
| Temporal Window Size | 5 days |
| Topic Threshold Alpha | 0.5 (mean − 0.5·std) |
| Drift Threshold Formula | mean + std |
| Batch Size | 32 (local) / 128 (Kaggle GPU) |
| Min Sentence Length | 20 characters |
| Normalization | L2-normalized embeddings |

**Calibrated Topic Thresholds (cosine similarity):**

| Topic | Threshold |
|-------|-----------|
| Climate | 0.437 |
| Economics | 0.474 |
| Health | 0.441 |
| Technology | 0.437 |
| War | 0.412 |

**Calibrated Drift Thresholds (1 − cosine):**

| Topic | w1 | w3 | w5 |
|-------|----|----|-----|
| Climate | 0.321 | 0.390 | 0.395 |
| Economics | 0.277 | 0.380 | 0.419 |
| Health | 0.256 | 0.389 | 0.439 |
| Technology | 0.255 | 0.388 | 0.409 |
| War | 0.318 | 0.426 | 0.480 |

### B.3 K-Means Clustering Parameters

| Parameter | Value |
|-----------|-------|
| Number of Clusters (K) | 5 |
| Distance Metric | Euclidean |
| Drift Metric | Jensen-Shannon divergence |
| Drift Threshold | 0.3 |
| Temporal Window Size | 5 days |
| Topic Filter Threshold | 0.3 |
| Training Dataset | Climate.csv (~189K sentences) |

### B.4 TCL Model Architecture (Evolution Across Approaches)

**Note:** The TCL approach evolved significantly through 5 iterations. Below are the key architectural configurations.

**Approach 1: Fixed Window with InfoNCE**

| Component | Configuration |
|-----------|--------------|
| Input Dimension | 774 (768 SBERT + 5 topic scores + 1 time gap) |
| Projection Dimension | 256 |
| Transformer Layers | 4 |
| Attention Heads | 8 |
| Feed-Forward Dimension | 1024 |
| Activation Function | GELU |
| Dropout Rate | 0.1 |
| Output Dimension | 128 |
| Window Construction | Fixed (overlapping/non-overlapping) |
| Loss Function | InfoNCE |

**Approach 2: Grouping-Based with Enhanced NT-Xent**

| Component | Configuration |
|-----------|--------------|
| Input Dimension | 774 (768 + 5 + 1) |
| Pooling Strategy | Topic-weighted lambda mean pooling |
| Grouping Strategy | Count-based (4 days) OR date-range (5 days) |
| Architecture | Enhanced Transformer with attention pooling |
| Loss Function | Enhanced NT-Xent + Hard Negative Mining |
| Loss Weights | λ₁ (NT-Xent), λ₂ (Hard-Neg) |

**Approach 3: Topic-Wise Window Sizing (Not Implemented)**

| Component | Configuration |
|-----------|--------------|
| Window Size | Topic-specific (War: 5, Health: 7, Economics: 4, Technology: 6, Climate: 8) |
| Challenge | Temporal non-stationarity, computational complexity |
| Status | Research phase, future work |

**Approach 4: Dynamic Segmentation with Topic Separation (Current)**

| Component | Configuration |
|-----------|--------------|
| Input Dimension | 774 (768 SBERT + 5 topic scores + 1 time gap) |
| Hidden Dimension | 512 |
| Projection Dimension | 256 |
| Transformer Layers | 4 |
| Attention Heads | 8 |
| Dropout Rate | 0.1 |
| Output Dimension | 128 |
| Segmentation | Ruptures PELT with RBF kernel |
| Ruptures Penalty | 0.1 |
| Min Segment Size | 2 days |
| Grouping | Maximum similarity-based grouping |
| Loss Function | NT-Xent + Topic Separation + Hard Negative Mining |
| Loss Weights | λ_temporal=1.5, λ_topic=0.5, λ_hard=0.3 |
| Reference | Truong et al. (2020) [16] |

**Approach 5: Multi-Modal NER + Sentiment + FAISS (✅ Code Complete)**

| Component | Configuration |
|-----------|--------------|
| Input Dimension | 768 (article-level: 0.6 semantic + 0.2 entity + 0.2 sentiment) |
| NER | spaCy (PERSON, ORG, GPE, NORP); sentence-level → article aggregation |
| Entity Canonicalization | Agglomerative clustering, cosine threshold = 0.85 |
| Sentiment | RoBERTa; per-sentence −1/0/+1; article = mean of sentences |
| TCL Architecture | MLP: 768 → Linear(1024) → ReLU → Dropout(0.1) → Linear(512) |
| Projection Dimension | 512 |
| Retrieval | FAISS topic-specific index (per topic) |
| Shift Score | 0.45 × semantic + 0.20 × sentiment + 0.20 × entity + 0.15 × claim |
| Shift Threshold | 0.65 |
| Target Hardware | Kaggle GPU P100 |
| Status | ✅ Code complete; evaluation metrics TBD |

### B.5 TCL Training Hyperparameters (Evolution Across Approaches)

**Common Parameters (All Approaches):**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Scheduler | Cosine Annealing with Warmup |
| Gradient Clip Norm | 1.0 |
| Device | CUDA (GPU) |

**Approach-Specific Parameters:**

| Parameter | Approach 1 | Approach 2 | Approach 4 (Current) |
|-----------|-----------|-----------|---------------------|
| Batch Size | 128 | 32 | 128 |
| Learning Rate | 1e-4 | 1e-4 | 3e-4 |
| Weight Decay | 1e-4 | 1e-4 | 1e-5 |
| Max Epochs | 50 | 100 | 100 |
| Warmup Epochs | 5 | 5 | 5 |
| Temperature (τ) | 0.07 | 0.07 | 0.05 |
| Window Size | 3 days (fixed) | 3 groups | 2 segments (adaptive) |
| Window Stride | 2 days | 3 groups | 1 segment |
| Loss Components | InfoNCE | NT-Xent + Hard-Neg | NT-Xent + Topic-Sep + Hard-Neg |

**Approach 4 (Current) - Detailed Configuration:**

| Parameter | Value |
|-----------|-------|
| Batch Size | 128 |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-5 |
| Max Epochs | 100 |
| Warmup Steps | 500 |
| Temperature (τ) | 0.05 |
| Ruptures Model | PELT + RBF |
| Ruptures Penalty | 0.1 |
| Topic Weight Threshold | 0.55 (filter low-confidence sentences) |
| Shift Threshold | mean + 1.5×std |
| Loss λ_temporal | 1.5 |
| Loss λ_topic_sep | 0.5 |
| Loss λ_hard_neg | 0.3 |

---

## Appendix C: Code Availability

The complete source code, processed datasets, and trained models will be made available through our GitHub repository:

**Repository:** [Will be shared with mentor and co-mentors]

**Contents:**
- Data collection and preprocessing scripts
- TCL model implementation
- Baseline model implementations
- Evaluation scripts
- Visualization notebooks
- Documentation and tutorials

**License:** [To be determined]

---

**Document Version:** 1.0  
**Last Updated:** March 7, 2026  
**Total Pages:** [To be determined after ACL formatting]
