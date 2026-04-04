# Approach 4 Technical Documentation

Detailed documentation for `TCL/TCL_Pipeline_4.ipynb`.

`TCL_Pipeline_4.ipynb` now contains the new refactored Approach 4 code (synced from `TCL_Pipeline_new_4.ipynb`) and is the canonical notebook for this pipeline.

## 1. Objective

Approach 4 detects narrative shifts with Temporal Contrastive Learning using:
- adaptive temporal segmentation with Ruptures (`PELT` + `rbf`), and
- an enhanced multi-term contrastive objective.

The pipeline targets stronger inter-topic separation while preserving temporal continuity for shift detection.

## 2. Notebook Structure

Main section flow in `TCL_Pipeline_4.ipynb`:
- `1. Imports`
- `2. Config`
- `3. Data Loading`
- `4. Preprocessing (Day -> Group via Ruptures)`
- `5. Embedding`
- `6. Model Definition`
- `7. Training`
- `8. Evaluation`
- `9. Inference (User-level)`
- `10. Utilities`
- `11. User Inference Call`
- `12. User Inference Sentence-Level Output (Final)`

## 3. End-to-End Data Flow

```text
Topic CSVs
  -> load_topic_dataframe
  -> aggregate_daily_vectors
  -> create_grouped_vectors_from_daily (ruptures)
  -> add_temporal_features (768 sentence + 64 topic = 832)
  -> build_window_embeddings (window_size=2, stride=1)
  -> TemporalWindowDataset + BalancedTopicBatchSampler
  -> TCLTemporalEncoder
  -> EnhancedNTXentLoss
  -> training checkpoints + evaluation artifacts
  -> user inference + sentence-level shift output
```

## 4. Core Configuration

Important configuration families:
- alignment profile and runtime guards
- data paths and topic schema
- feature dimensions (`embedding_dim=768`, `topic_embedding_dim=64`, `final_dim=832`)
- grouping settings (`ruptures_only=True`, `ruptures_model="rbf"`, `ruptures_penalty=0.1`, `ruptures_min_size=2`)
- inference controls (`topic_threshold`, `manual_shift_threshold`, `inference_batch_size`)
- model architecture and training controls
- artifact naming and save/load interface

Artifact base naming pattern:
- `approch_ruptures_pen0p1_4_w{window_size}_s{stride}_t{temperature_tag}`

## 5. Grouping Logic (Approach 4)

Temporal grouping is Ruptures-only:
- `detect_change_points_ruptures(...)`
- `create_groups_ruptures(...)`
- `create_grouped_vectors_from_daily(...)`

No fixed-size grouping and no day-gap grouping are used in this notebook path.

## 6. Model and Loss

Encoder:
- `TCLTemporalEncoder`
- Transformer-based temporal encoder with projection to normalized embedding space

Loss:
- `EnhancedNTXentLoss`
- composite objective combining:
  - temporal contrastive loss
  - topic separation loss
  - hard negative loss

Configured lambda weights:
- `lambda_temporal = 1.5`
- `lambda_topic_sep = 0.5`
- `lambda_hard_neg = 0.3`

## 7. Training and Evaluation

Training includes:
- balanced topic batching via `BalancedTopicBatchSampler`
- optimizer `AdamW`
- scheduler via warmup + cosine logic
- AMP support, gradient clipping, early stopping
- checkpoint save variants: `best`, `last`, `evaluated`

Evaluation includes:
- representation quality metrics
- intra-topic and inter-topic similarity heatmaps
- metrics JSON and summary artifacts saved to output path

## 8. User Inference Design

Inference reuses training-compatible feature flow:
- split user articles into sentences
- contextual SBERT embeddings
- soft topic labeling
- topic filtering
- inference alignment validation
- daily aggregation -> ruptures grouping -> window embeddings
- model encoding and drift computation
- normalized shift detection and sentence-level explanation output

Compatibility behavior:
- checkpoint candidate fallback list (`best`, `evaluated`, legacy path)
- shape-safe checkpoint loading with explicit runtime error on mismatch
- CPU-safe inference path for embedding generation and model execution

## 9. Sentence-Level Shift Output Contract

Sentence-level extraction is implemented in:
- `extract_sentence_level_narrative_shifts(...)`

Output payload fields include:
- date pair, sentence ids, article ids, sentence numbers, raw sentence text
- topic weights, similarity, shift score
- day-level drift and z-score
- context strings (`context_1`, `context_2`)

Duplicate prevention is active before appending shift pairs:
- tracks `used_sentence_ids` to prevent reusing a sentence across results
- tracks `used_sentence_pairs` to prevent repeated pair combinations
- skips candidate pairs when either sentence or pair already appeared

This ensures `sentence_1` and `sentence_2` outputs are non-duplicate in final returned shifts.

## 10. Output Artifacts

Primary outputs:
- `{model_base_name}_best.pt`
- `{model_base_name}_last.pt`
- `{model_base_name}_evaluated.pt`
- `{model_base_name}_train_loss.png`
- `{model_base_name}_evaluation_metrics.json`
- `{model_base_name}_intra_heatmap.png`
- `{model_base_name}_inter_heatmap.png`
- `{model_base_name}_run_summary.json`
- `{model_base_name}_user_inference_multi_topic.json`

Default output directory:
- `./tcl_output_new_4`
