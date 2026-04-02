# 1. Overview
This report compares four TCL notebooks used as approach variants:
- `TCL/TCL_Pipeline_1.ipynb`
- `TCL/TCL_Pipeline_2.ipynb`
- `TCL/TCL_Pipeline_4.ipynb`
- `TCL/TCL_Pipeline_5.ipynb`

Scope is limited to model architecture, contrastive objective, training loop behavior, and inference/shift detection path. The goal is to identify where the four approaches are conceptually aligned vs behaviorally inconsistent, and define one implementation-ready unification plan.

# 2. Approach-Wise Core Design
## Approach 1 (`TCL_Pipeline_1.ipynb`)
- Input contract is topic-conditioned temporal windows with `FINAL_DIM=774` and `WINDOW_SIZE=3` from `768 + 1 + 5`.
- Encoder is `ImprovedTemporalEncoder` with input LayerNorm, learned+sinusoidal position encoding, temporal decay weighting, Transformer encoder, attention pooling, residual post-MLP, and projection to 128-D normalized embedding.
- Loss is `EnhancedNTXentLoss` but operationally used as standard NT-Xent (hard-negative path is mostly disabled in training call).
- Training samples consecutive pairs from dataset API (`get_consecutive_pair_batch`) rather than iterating fixed DataLoader batches.
- Inference/evaluation includes drift score + z-score logic and a custom article branch, but custom branch exposes dimension mismatch risk (`model expects 774`, branch builds 390-D vectors).

## Approach 2 (`TCL_Pipeline_2.ipynb`)
- Keeps same base feature contract (`FINAL_DIM=774`) and 3-group windows.
- Uses a grouped temporal segmentation stage (fixed-size grouping or max-gap grouping), then creates windows from groups.
- Model class is near-equivalent improved transformer encoder to Approach 1.
- Dataset is explicit paired windows (`WindowPairDataset`) and training is a conventional DataLoader loop.
- Loss implementation is cleaner and numerically stable NT-Xent variant; no explicit multi-objective coupling.
- Inference path is mostly drift timeline and evaluation utilities over learned embeddings.

## Approach 4 (`TCL_Pipeline_4.ipynb`)
- Pipeline widens to sentence filtering by topic confidence, daily weighted aggregation, explicit change-point detection via ruptures PELT+RBF, segment construction, then windowing.
- Feature contract shifts to `832-D` input (semantic + topic embedding), then transformer projects to `256-D` output.
- Loss is multi-component (`temporal + topic separation + hard negatives`) with configurable lambdas.
- Adds balanced topic batching and richer diagnostics/plots.
- Shift detection uses model-embedding transitions and thresholded score logic.

## Approach 5 (`TCL_Pipeline_5.ipynb`)
- Most extensive pipeline: unified multi-topic training, entity extraction and entity-invariant embeddings, optional rupture grouping (training only), topic embedding concatenation to 832-D, then temporal transformer.
- Training objective explicitly modularized into NT-Xent, topic separation, hard negative mining, and weighted `MultiLoss`.
- Includes a long end-to-end inference flow for user articles with sentence windowing, topic weighting, entity cleaning, day-level pooling, topic embedding injection, window prediction, day-level shifts, and sentence-level attribution.
- Conceptually strongest end-user inference story, but highest coupling and notebook complexity.

# 3. Model Architecture Comparison
Common backbone trend:
- All approaches encode temporal windows with transformer-style encoders and produce normalized representations for contrastive objectives.

Critical architecture divergence:
- Approach 1/2 model input: `774-D` and output `128-D`.
- Approach 4/5 model input: `832-D` and output `256-D`.

Operational consequence:
- Checkpoints and inference artifacts are not directly interchangeable across approaches because feature contract and output space differ.
- Any shared evaluation script that assumes one dimensionality will silently fail or force ad-hoc padding/truncation.

Pooling/representation differences:
- Approach 1/2 use attention pooling + residual post-MLP in improved encoder.
- Approach 4/5 rely on mean temporal pooling (simpler but less selective).

# 4. Loss Function Comparison
Approach 1/2:
- Primary objective is NT-Xent style temporal contrastive loss.
- Approach 1 class name suggests enhanced hard-negative support, but training path effectively uses plain NT-Xent settings.

Approach 4/5:
- Explicit multi-objective loss:
  - Temporal contrastive term
  - Topic separation term
  - Hard negative term
- Weighted sum with lambda controls.

Key inconsistency:
- Approaches are not optimizing the same objective family, so metric and drift-score comparability is weakened. Better separation in 4/5 may come from auxiliary terms, not only better temporal modeling.

# 5. Training Pipeline Comparison
Approach 1:
- Uses dataset-driven random consecutive-pair sampling inside epoch loop (custom mini-batch generation).

Approach 2:
- Uses DataLoader-backed paired dataset loop (more conventional and reproducible).

Approach 4:
- Balanced topic batch strategy and multi-term loss training.

Approach 5:
- Similar multi-loss training but with larger upstream preprocessing surface and more saved state dependencies (topic embedding layer, mapping metadata).

Shared positives:
- Gradient clipping, scheduler use, AMP support (where configured), checkpointing are present broadly.

Training inconsistency risks:
- Different sample construction (day windows vs segment windows vs group windows) changes what "positive temporal pair" means.
- Different topic balancing policies alter gradient distribution and may dominate model behavior more than architecture changes.

# 6. Inference and Shift Detection Comparison
Approach 1/2:
- Primarily embedding-drift style detection (cosine distance + z-score/percentile thresholding).
- Practical for topic-level timeline analysis.

Approach 4:
- Stronger segmentation-aware inference framing, with richer visual diagnostics around shift scores.

Approach 5:
- Full user-facing inference chain with sentence-level attribution and context extraction.
- Most complete for explainability, but also the most fragile due to many dependency links between training-time artifacts and inference-time feature construction.

Most important inconsistency:
- Shift score definitions and thresholding conventions vary by approach (raw cosine distance, normalized score variants, z-score use, percentile cuts, or mixed). This prevents fair cross-approach shift-event comparison without calibration.

# 7. Cross-Approach Inconsistencies
1. Feature space mismatch
- `774-D` family (Approach 1/2) vs `832-D` family (Approach 4/5).

2. Embedding output mismatch
- `128-D` output (1/2) vs `256-D` output (4/5).

3. Objective mismatch
- NT-Xent-only vs multi-objective loss.

4. Temporal unit mismatch
- Day-level, group-level, rupture-segment-level windows are mixed.

5. Pairing strategy mismatch
- On-the-fly sampled pairs vs pre-built paired datasets.

6. Threshold semantics mismatch
- Different score normalizations and thresholds across notebooks.

7. Inference contract mismatch
- Some pipelines require topic embedding weights/checkpoint metadata and entity-cleaning steps; others do not.

8. Hidden behavioral mismatch in Approach 1 custom inference
- Custom branch constructs 390-D vectors while model expects 774-D and reports mismatch; this is a concrete behavior regression risk.

# 8. Unified TCL Consistency Verdict
Verdict:
- All four are TCL-inspired, but they are not one consistent TCL implementation line.
- Approach 1 and 2 are close siblings.
- Approach 4 and 5 form a second family with broader feature engineering and multi-objective optimization.
- Cross-family comparability is currently weak.

Recommendation:
- Treat current notebooks as experiments, not versions of one production pipeline.
- Define one canonical feature contract, one canonical objective, and one canonical shift-score calibration path before any final benchmark or deployment claim.

# 9. Refactoring Plan
## Target Modular Structure
- `src/config/`
  - `schema.py` (single config dataclass)
  - `profiles.py` (experiment presets)
- `src/data/`
  - `loaders.py` (topic csv loaders)
  - `parsing.py` (embedding/string parsing)
  - `aggregation.py` (day/group/segment aggregation)
  - `windowing.py` (all window builders)
- `src/features/`
  - `topic_features.py` (topic embedding injection)
  - `entity_invariant.py` (entity extraction and cleaning)
- `src/segmentation/`
  - `ruptures_segmentation.py`
- `src/model/`
  - `temporal_encoder.py` (single encoder family with pluggable dims)
  - `heads.py`
- `src/losses/`
  - `ntxent.py`
  - `topic_separation.py`
  - `hard_negative.py`
  - `composite.py`
- `src/train/`
  - `dataset.py`
  - `samplers.py`
  - `trainer.py`
  - `checkpointing.py`
- `src/infer/`
  - `topic_shift.py`
  - `sentence_shift.py`
  - `calibration.py`
- `src/eval/`
  - `metrics.py`
  - `plots.py`

## Canonical Contracts to Freeze First
1. Choose one input contract
- Prefer 832-D only if topic embedding concatenation is mandatory at inference too.
- Otherwise freeze 774-D and keep topic effects external.

2. Choose one embedding output size
- Standardize to a single dimension (128 or 256), then keep all metrics and thresholds bound to it.

3. Choose one loss family
- Either single NT-Xent baseline or composite loss baseline; keep the other as optional ablation mode.

4. Choose one temporal segmentation policy
- Keep rupture/group/day as pluggable strategy behind identical window API.

5. Choose one shift scoring and calibration scheme
- Example: cosine-distance series -> rolling smoothing -> z-score -> calibrated threshold per topic.

## Migration Sequence
1. Extract shared utilities from notebooks into `src/data` and `src/features`.
2. Implement one canonical model + loss path in `src/model` and `src/losses`.
3. Port one training notebook as reference runner script.
4. Port one inference notebook as reference runner script.
5. Add compatibility adapters for old checkpoints only if necessary.
6. Add regression tests for:
- Shape contracts
- Pair construction correctness
- Loss numeric stability
- Shift-score reproducibility for fixed seed

## End State
A single configurable TCL pipeline with experiment profiles that can reproduce Approach 1/2/4/5 behaviors as toggles, while preserving one stable production default.