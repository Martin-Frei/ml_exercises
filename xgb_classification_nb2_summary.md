# XGBoost Classification – NB2: Imputation + Outlier Clipping – Summary

## Goal

Test whether better **stress-data preparation** (outlier clipping and leakage-free imputation) improves the stress-test accuracy of the XGBoost classifier for `meat_type` (classes 1–5).

**Notebook:** `xgb_classification_imputation_clip.ipynb`

---

## Setup

- Same data split as NB1 (`random_state=42`, stratified 80/20)
- Model trained **once** with the best NB1 setting: `max_depth=3`, `n_estimators=150`, `learning_rate=0.08`
- Only the preparation of the stress data changes between runs
- Stress rows without a `meat_type` label dropped: 100 → 91 rows (1 sample = 1.1 percentage points)

**Internal test (unchanged from NB1 depth 3):** 67.1% accuracy

---

## Step A – Outlier Clipping

Every stress value outside the **training** min/max gets set to the nearest training boundary (e.g. `animal_age_months` 189 → 71). The bounds come from `X_train` only, because the model has only learned that range, and future data is unknown when the preprocessing is built.

**13 values** were clipped in total.

**Result:** no effect. Native NaN, global median and KNN give identical results with and without clipping. Iterative lost one sample with clipping (21 → 20).

**Why:** a tree cannot extrapolate. Age 189 and age 71 already land in the same leaf, because both lie beyond the last split point. Clipping changes the value, not the leaf.

---

## Step B – Leakage-Free Imputation (8 runs)

| Method | How NaN are filled |
|---|---|
| Native NaN | Not filled – XGBoost uses a learned default branch |
| Global median | Training median of the feature |
| KNN (k=5) | Mean of the 5 most similar training rows (scaled features, integer features rounded) |
| Iterative | Each missing feature predicted from the other features |

All imputers fitted on `X_train` only. The target is never used.

### Results

| Variant | Correct / 91 | Accuracy | Macro F1 |
|---|---|---|---|
| **Iterative** | **21** | **23.08%** | **0.198** |
| Native NaN | 20 | 21.98% | 0.169 |
| KNN | 20 | 21.98% | 0.184 |
| Native NaN + clip | 20 | 21.98% | 0.169 |
| Iterative + clip | 20 | 21.98% | 0.180 |
| KNN + clip | 20 | 21.98% | 0.184 |
| Global median | 19 | 20.88% | 0.159 |
| Global median + clip | 19 | 20.88% | 0.159 |

**The full spread is 2 samples (19–21 of 91).** Iterative "wins" by one sample over native NaN, which is noise, not a real improvement. Global median is last, consistent with NB1, where median fill never beat native NaN.

### Recall per class across all variants

| Variant | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
|---|---|---|---|---|---|
| Native NaN | 0.11 | 0.06 | 0.08 | 0.04 | 0.79 |
| Global median | 0.06 | 0.06 | 0.08 | 0.08 | 0.74 |
| KNN | 0.11 | 0.12 | 0.08 | 0.04 | 0.74 |
| Iterative | 0.11 | 0.12 | 0.08 | 0.12 | 0.68 |
| Native NaN + clip | 0.11 | 0.06 | 0.08 | 0.04 | 0.79 |
| Global median + clip | 0.06 | 0.06 | 0.08 | 0.08 | 0.74 |
| KNN + clip | 0.11 | 0.12 | 0.08 | 0.04 | 0.74 |
| Iterative + clip | 0.06 | 0.12 | 0.08 | 0.12 | 0.68 |

**The pattern is identical in all 8 variants:**

- **Class 1 (cheapest):** at most 2 of 18 found
- **Classes 2, 3, 4:** 1–3 correct each; class 3 is exactly 1 of 12 in **every** variant
- **Class 5 (most expensive):** 13–15 of 19 found, but precision only ~0.31, because the model predicts class 5 far too often

Iterative's extra sample comes from a trade: +1 in class 2 and +2 in class 4, but −2 in class 5. The model predicts class 5 a little less often; it did not learn anything new.

---

## Step C – Leakage Demonstration (INVALID reference)

The RF classification used "domain-based imputation": each NaN filled with the median of the row's **own `meat_type`**. For classification, `meat_type` is the target, so this puts the answer into the input (**data leakage**).

| Method | Correct / 91 | Accuracy |
|---|---|---|
| Best valid (Iterative) | 21 | 23.08% |
| **Leaky group median** | **24** | **26.37%** |

**Inflation: +3 samples (+3.3 percentage points)**, more than the entire spread of all 8 valid methods.

Even with only 6–10 NaN per feature, the leakage is measurable: a missing fat value filled with "median fat of meat_type 1" hints the model toward class 1.

**Consequence:** the RF "domain imputation" stress result (27.5%) is very likely inflated by a similar amount and is **not a fair benchmark for classification**. The method is still valid for **regression**, where `meat_type` is a feature, not the target.

---

## Comparison with Previous Models

| Model | Acc Test | Acc Stress |
|---|---|---|
| DT baseline | 53.0% | 27% |
| RF baseline (KNN) | 56.0% | 24% |
| RF domain imputation (leaky) | 56.0% | 27.5% ⚠ |
| RF + target encoding | 62.5% | 25% |
| XGB NB1 (depth 3, native NaN) | 67.1% | 22.0% |
| **XGB NB2 best (Iterative)** | **67.1%** | **23.1%** |

---

## Key Findings

1. **Imputation doesn't matter.** 8 variants, spread of 2 samples out of 91.
2. **Clipping doesn't matter** for tree models, which already treat out-of-range values like the boundary value.
3. **Leakage is measurable even with few NaN.** +3 samples, more than any honest method achieved.
4. **The recall pattern never changes:** cheap meat is missed and everything is pushed toward class 5.

## Why – The Price Shift (from NB1)

| meat_type | Mean price (train) | Mean price (stress) | Shift |
|---|---|---|---|
| 1 | 14.0 € | 22.1 € | +8.1 € |
| 2 | 16.4 € | 26.6 € | +10.2 € |
| 3 | 24.2 € | 32.2 € | +8.0 € |
| 4 | 27.2 € | 33.3 € | +6.1 € |
| 5 | 32.5 € | 33.6 € | +1.1 € |

The price ↔ meat_type correlation falls from 0.82 (train) to 0.45 (stress). A class-1 cut in the stress data costs what a class-3 cut costs in training, so the model's learned price boundaries point to the wrong class. That is a **distribution shift**, and preparing the stress data cannot fix it.

---

## Conclusion

NB2 is a **valid negative result**: it systematically rules out stress-data preparation as a solution. The model has to **see** shifted data during training. That is the approach of the regression breakthrough (combined dataset training: stress R² −0.36 → +0.20).

## Next Steps

- **NB3 – Target encoding:** likely to help the internal test (best RF technique), unlikely to help the stress test, since the encodings are learned from unshifted prices
- **NB4 – Combined dataset training:** attacks the shift directly; needs a cleanly separated stress hold-out for fair evaluation
- **Add a leakage note to `rf_classification_summary.md`**
