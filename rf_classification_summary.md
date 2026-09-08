# Random Forest Classification – Summary

## Goal

Predict `meat_type` (classes 1–5) using all other features. Beat the Decision Tree baseline (53% internal, 27% stress test) and improve the weak middle classes (2, 3, 4) that overlap in price.

---

## The Journey — Step by Step

### Step 1: Random Forest Baseline (KNN Imputation)

**What we did:** Replaced the single Decision Tree with 200 trees. Used KNN imputation for stress-test missing values.

**Config:** n_estimators=200, max_depth=7, KNN(n_neighbors=5)

**Result:**

| Metric | Decision Tree | Random Forest |
|---|---|---|
| Accuracy (test) | 53% | 56% |
| Accuracy (stress) | 27% | 24% |

**Per-class F1 (internal test):**

| Class | DT F1 | RF F1 | Change |
|---|---|---|---|
| 1 (cheapest) | 0.65 | 0.74 | +0.09 |
| 2 | 0.39 | 0.48 | +0.09 |
| 3 | 0.49 | 0.39 | -0.10 |
| 4 | 0.24 | 0.42 | +0.18 |
| 5 (expensive) | 0.77 | 0.73 | -0.04 |

**Learning:** The forest spread feature importance more evenly — price dropped from 70% (single tree) to 52%, while fat, protein, and age each contributed ~8%. This diversity helped class 4 the most (+0.18 F1). But KNN imputation hurt the stress test — more features used means more chances for bad fills to mislead.

---

### Step 2: Domain-Based Imputation

**What we did:** Replaced KNN imputation with group-based fills. Instead of filling NaN from similar rows across all meat types, each NaN gets the median of its own meat_type group from the training data.

**Example of the difference:**

| Missing value | KNN fill | Domain-based fill |
|---|---|---|
| fat_content NaN, meat_type=1 | Based on 5 nearest rows (any type) | Median of meat_type 1 only |
| price NaN, meat_type=5 | Based on 5 nearest rows | Median of meat_type 5: 32.40 EUR |
| price NaN (global median) | 23.22 EUR | — |

A missing price for the most expensive meat type gets 32.40 EUR instead of 23.22 EUR — a 9 EUR difference in fill quality.

**Implementation:**
1. Build median table per meat_type from training data
2. Fill meat_type NaN with mode (needed as grouping key)
3. Fill all other NaN per group
4. Fallback: remaining NaN (unknown meat_type) → global median

**Result:**

| Imputation | Accuracy Stress |
|---|---|
| KNN | 24.18% |
| **Domain-based** | **27.47%** |

**Learning:** Fills that respect the meat category are more realistic. A beef NaN should be filled with beef values, not a global average across beef, pork, and chicken. Domain knowledge (Metzgermeister) directly improved the model.

---

### Step 3: Target Encoding

**What we did:** Created 4 new features by encoding each feature value with the mean meat_type of all training rows with that value.

**What target encoding does:**

Instead of raw numbers (cut_quality = 1, 2, 3), each value gets replaced by the average target for that group:

```
cut_quality 1 → mean meat_type where cut_quality=1 → 2.99
organic 0 → mean meat_type where organic=0 → 2.97
price bin 0 (cheapest) → mean meat_type → 1.25
price bin 4 (most expensive) → mean meat_type → 4.85
```

**Features encoded:**

| Original feature | New column | Why |
|---|---|---|
| cut_quality (1–5) | cut_quality_te | Captures quality-to-type relationship |
| organic (0/1) | organic_te | Captures organic-to-type relationship |
| marbling_score (1–10) | marbling_score_te | Captures marbling-to-type relationship |
| price_eur_per_kg (binned) | price_bin_te | Price bins mapped to typical meat type |

**Data leakage prevention:** Encoding maps computed from training data only. Same maps applied to test and stress data. The model never sees test/stress target values during encoding.

**The model gets both:** 8 original + 4 encoded = 12 features. The tree can use whichever works better per split.

**Result — best classification across all experiments:**

| Experiment | Acc Test | Acc Stress |
|---|---|---|
| DT baseline | 53% | 27% |
| RF baseline | 56% | 24% |
| RF domain imputation | 56% | 27% |
| **RF + target encoding** | **62.5%** | 25% |

**Per-class F1 — the middle classes finally improved:**

| Class | DT F1 | RF baseline F1 | RF + target encoding F1 | Total gain |
|---|---|---|---|---|
| 1 (cheapest) | 0.65 | 0.74 | 0.74 | +0.09 |
| 2 | 0.39 | 0.48 | **0.55** | **+0.16** |
| 3 | 0.49 | 0.39 | **0.51** | +0.02 |
| 4 | 0.24 | 0.42 | **0.54** | **+0.30** |
| 5 (expensive) | 0.77 | 0.73 | 0.75 | -0.02 |

**Feature importance:**

| Feature | Importance | Type |
|---|---|---|
| price_eur_per_kg | 33.7% | Original |
| price_bin_te | 21.6% | Encoded |
| fat_content_pct | 6.6% | Original |
| cut_quality | 6.7% | Original |
| animal_age_months | 6.0% | Original |
| protein_pct | 5.6% | Original |
| marbling_score | 4.6% | Original |
| storage_days | 5.4% | Original |
| marbling_score_te | 3.5% | Encoded |
| cut_quality_te | 3.5% | Encoded |
| organic_te | 1.5% | Encoded |
| organic | 1.3% | Original |

Original features total: 69.9%. Encoded features total: 30.1%. The price_bin_te encoding alone carries 21.6% — it became the second most important feature after raw price.

**Learning:** Target encoding gave the model a new way to separate overlapping classes. The raw price already had signal, but binned-and-encoded price captured group-level patterns the tree couldn't learn from raw numbers alone. This was the single most effective technique across all classification experiments.

---

## Learnings — What Worked

| Technique | Impact | Why it worked |
|---|---|---|
| Correct target selection (meat_type not cut_quality) | 22% → 53% | Correlation analysis revealed the actual signal |
| Random Forest over Decision Tree | 53% → 56% | 200 trees averaging reduces variance, spreads feature usage |
| Domain-based imputation | Stress 24% → 27% | Meat-type-specific fills are more realistic than global median |
| Target encoding | 56% → **62.5%** | New features capture group-level target patterns the raw data can't express |

---

## Dead Ends — What Didn't Work

| Technique | Result | Why it failed |
|---|---|---|
| Feature engineering (ratios) | No improvement, stress worse | Combining noise features (protein, fat) with signal (price) produced noise. Noise × signal = noise. |
| KNN imputation for stress test | Stress dropped 27% → 24% | More features used by RF means more chances for KNN fills to mislead on edge cases |
| Deeper/more complex trees | Marginal test gains only | The signal ceiling is set by the data, not model complexity |

---

## Final Results — Classification

| Experiment | Acc Test | Acc Stress | Key technique |
|---|---|---|---|
| DT baseline (cut_quality) | 22.5% | — | Wrong target |
| DT baseline (meat_type) | 53% | 27% | Correct target |
| DT + feature engineering | 53% | 26% | Dead end |
| RF baseline (KNN imputation) | 56% | 24% | Forest diversity |
| RF domain-based imputation | 56% | **27%** | Smart fills |
| **RF + target encoding** | **62.5%** | 25% | **Best overall** |

**Best internal test:** RF + target encoding — 62.5% accuracy

**Best stress test:** DT baseline and RF domain-based — 27% accuracy

---

## The Stress-Test Wall

All experiments landed between 24–27% accuracy on the stress test. The best approaches for each metric are different models:

- Best internal test: RF + target encoding (62.5%)
- Best stress test: RF domain-based imputation (27.5%)

You cannot maximize both. Higher internal accuracy comes from learning specific patterns in the clean data, which are exactly the patterns the stress test violates.

---

## Conclusion

From 22.5% (wrong target) to 62.5% (RF + target encoding) — a nearly 3x improvement through systematic experimentation. Each step added insight:

1. **Data understanding first** — correlation analysis before modeling
2. **Model upgrade** — forest beats single tree on clean data
3. **Domain knowledge** — Metzgermeister expertise improves imputation
4. **Feature engineering** — target encoding was the breakthrough technique

The stress test proved that real-world edge cases require more than model tuning. Recognizing that ceiling — and understanding why it exists — is the mark of ML maturity.

---

## Notebooks

```
random_forest_classification.ipynb        # Baseline + domain imputation
random_forest_target_encoding.ipynb       # Target encoding
```
