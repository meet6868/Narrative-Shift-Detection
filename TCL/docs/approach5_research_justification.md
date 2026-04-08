# Approach 5: Research Justification and Design Mapping

## Overview

This document explains the research foundations behind the design choices in Approach 5 (NER-based Temporal Contrastive Learning for Narrative Shift Detection). Each component in the code is mapped to relevant research papers and theoretical motivations.

---

# 1. Temporal Contrastive Learning (Core Idea)

## Paper
TS-TCC: Temporal and Contextual Contrasting for Time-Series Representation Learning  
Link: https://arxiv.org/abs/2106.10466

## Key Idea
- Adjacent time steps = Positive pairs  
- Distant time steps = Negative pairs  
- Learns temporal continuity without labels

## Mapping to Our Code

```
abs(window_idx_i - window_idx_j) == 1
```

## Why We Use It
Narrative shifts are inherently temporal. This method allows learning smooth narrative evolution vs sudden shifts.

---

# 2. Contrastive Loss with Temperature

## Paper
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations  
Link: https://arxiv.org/abs/2002.05709

## Key Idea
- Temperature controls similarity sharpness
- Lower values improve discrimination

## Mapping to Code

```
temperature = 0.05
```

---

# 3. Hard Negative Mining

## Paper
Supervised Contrastive Learning  
Link: https://arxiv.org/abs/2004.11362

## Key Idea
- Focus on hardest negatives for better representations

---

# 4. Entity-Invariant Representation

## Paper
Learning Not to Learn: Training Deep Neural Networks with Biased Data  
Link: https://arxiv.org/abs/1812.10352

## Key Idea
- Remove entity bias from embeddings

---

# 5. Entity-Based Contrastive Learning

## Idea
- Similar entities → similar narratives
- Different entities → different context

---

# 6. Multi-Objective Loss

```
total_loss =
    lambda_temporal * temporal_loss
  + lambda_topic_sep * topic_sep_loss
  + lambda_hard_neg * hard_neg_loss
  + lambda_entity * entity_loss
```

---

# 7. Final Justification

Our approach integrates temporal continuity, entity awareness, and topic separation into a unified contrastive learning framework for robust narrative shift detection.

---

# End
