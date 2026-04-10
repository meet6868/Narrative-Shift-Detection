# TCL Inference Pipelines (Approach-wise)

This file consolidates inference pipelines from all approach docs into one place.

Source docs used:
- `TCL/docs/approach_1.md`
- `TCL/docs/approach_2.md`
- `TCL/docs/approach_3.md`
- `TCL/docs/approach_4.md`
- `TCL/docs/approach_5.md`

## Common Inference Backbone

Most approaches follow this shared flow:
1. Load user CSV (`date`, `article`).
2. Split article text into sentences.
3. Build context text around each sentence.
4. Encode context with SBERT (`all-mpnet-base-v2`).
5. Compute topic relevance (soft topic labeling).
6. Filter by topic threshold.
7. Build temporal units (daily/group/windows).
8. Run TCL encoder to produce temporal embeddings.
9. Compute drift between consecutive windows.
10. Detect high-drift points as candidate shifts.
11. Extract sentence-level evidence around shift dates.
12. Save per-topic or multi-topic JSON output.

---

## Approach 1 Inference Pipeline

Reference: `TCL/docs/approach_1.md` (Section 6)

### Flow
1. Split sentences from user CSV.
2. Build 5-sentence context (`prev2, prev1, cur, next1, next2`).
3. SBERT encode context text.
4. Soft topic labeling using topic prototypes JSON.
5. Filter sentences by topic score threshold.
6. Daily weighted aggregation.
7. Add temporal features (`tau`) and build windows.
8. Encode windows with trained AP1 model.
9. Drift score: `1 - cosine_similarity` on consecutive windows.
10. Optional rolling smoothing.
11. Z-score normalization.
12. Shift trigger: `z > zscore_threshold` OR percentile cutoff.
13. Sentence-level extraction: lowest-similarity pairs + local context.

### Typical inference controls
- `topic_threshold`
- `zscore_threshold`
- `percentile_threshold`
- `drift_smoothing_window`
- `inference_batch_size`

---

## Approach 2 Inference Pipeline

Reference: `TCL/docs/approach_2.md` (Section 7)

### Flow
1. Split sentences.
2. Build context text.
3. SBERT encode.
4. Soft topic label.
5. Filter by topic threshold.
6. Daily aggregation.
7. Apply AP2 grouping strategy:
   - fixed-size grouping, or
   - max-day-gap grouping.
8. Add temporal features and build windows (`size=3`, `stride=3` in AP2 doc).
9. Encode windows with AP2 model.
10. Drift computation + z-score based shift detection.
11. Sentence-level shift evidence extraction.

### AP2-specific point
- Temporal grouping is explicit in inference (same design as training).

---

## Approach 3 Inference Pipeline

Reference: `TCL/docs/approach_3.md`

### Status
- Approach 3 is theoretical/research phase and not implemented as a runnable pipeline.
- No production inference path is defined.

### Intended idea
- Use topic-specific adaptive window sizes learned from historical drift behavior.
- Then run drift detection and shift extraction on those adaptive windows.

---

## Approach 4 Inference Pipeline

Reference: `TCL/docs/approach_4.md` (Section 7)

### Flow
1. Split sentences.
2. Build context text.
3. SBERT encode.
4. Soft topic assignment with ideal topic prototypes.
5. Filter with stricter threshold (doc highlights high-confidence filtering).
6. Daily weighted pooling.
7. Build windows with adaptive handling when user-day count is short.
8. Encode windows with AP4 model.
9. Compute drift and normalize scores.
10. Apply manual/threshold-based shift detection.
11. Extract sentence-level evidence for detected shifts.

### AP4-specific points
- Topic-aware features include learned 64-d topic embeddings in the representation path.
- Inference may use stricter topic filtering than training to reduce noise.

---

## Approach 5 Inference Pipeline

Reference: `TCL/docs/approach_5.md` (Section 6)

### Flow
1. Load user CSV.
2. Split sentences and build context.
3. Contextual SBERT embeddings.
4. Entity extraction + entity-aware embedding cleaning/projection.
5. Soft topic labeling with prototypes.
6. Filter by topic threshold.
7. Daily weighted pooling.
8. Attach learned topic vector.
9. Build windows.
10. Encode windows and compute drift.
11. Detect shifts using manual shift threshold.
12. Extract sentence-level evidence from adjacent dates.
13. Save multi-topic inference JSON.

### AP5-specific points
- Entity-aware signal is included in inference features.
- Drift score is distance-based and ranked with z-score metadata.
- Output is typically multi-topic in one JSON payload.

---

## Quick Comparison

| Approach | Inference Readiness | Main Temporal Unit | Distinctive Inference Feature |
|---|---|---|---|
| 1 | Implemented | Daily windows | Baseline drift + z-score + percentile trigger |
| 2 | Implemented | Grouped windows | Fixed/max-gap grouping in inference |
| 3 | Theoretical | Adaptive (planned) | Topic-specific learned window size (planned) |
| 4 | Implemented | Adaptive windows | Stricter topic filter + topic-aware representation |
| 5 | Implemented | Entity-aware windows | Entity-aware inference + multi-topic JSON |

---

## Practical Notes

- For all implemented approaches (1,2,4,5), keep training/inference feature schema aligned.
- If sentence-level evidence looks noisy, tune:
  - `topic_threshold`
  - shift threshold parameters (`zscore`, percentile, or manual)
  - smoothing window
- AP5 generally gives strongest interpretability when entity extraction quality is good.
