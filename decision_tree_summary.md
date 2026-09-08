# Decision Tree – Classification & Regression Summary

## 1. The Dataset

Two CSV files with meat industry data (1200 training rows, 100 stress-test rows):

| Column | Type | Description |
|---|---|---|
| meat_type | int (1–5) | Meat category |
| fat_content_pct | float | Fat percentage |
| protein_pct | float | Protein percentage |
| marbling_score | int | Marbling grade |
| animal_age_months | int | Animal age |
| storage_days | int | Days in storage |
| organic | int (0/1) | Organic flag |
| cut_quality | int (1–5) | Cut quality grade |
| price_eur_per_kg | float | Price in EUR/kg |

The **stress-test dataset** contains intentional edge cases: missing values, outliers, and unusual feature combinations designed to challenge the model.

---

## 2. Classification – Predicting a Category

### 2.1 Attempt 1: Target = `cut_quality`

**Result: 22.5% accuracy (5 classes → 20% = random guessing)**

The model learned nothing. Correlation analysis revealed why — no feature had meaningful correlation with `cut_quality`. The highest was `price_eur_per_kg` at 0.386, all others below 0.04.

**Key lesson:** A model can only learn patterns that exist in the data. Bad accuracy is not always a model problem — it can be a data problem.

### 2.2 Attempt 2: Target = `meat_type`

**Result: 53% accuracy — a major improvement**

Switching the target worked because `meat_type` has a strong 0.82 correlation with `price_eur_per_kg`. The model learned to classify meat categories primarily through price boundaries.

**Per-class performance:**

| Class | F1-Score | Why |
|---|---|---|
| 1 (cheapest) | 0.65 | Clear price boundary at the low end |
| 2 | 0.39 | Overlapping mid-range prices |
| 3 | 0.49 | Over-predicted — acts as catch-all for middle classes |
| 4 | 0.24 | Worst class — lost in the overlap zone |
| 5 (most expensive) | 0.77 | Clear price boundary at the high end |

**Pattern:** Extreme classes (1 and 5) perform well because they sit at distinct price ranges. Middle classes (2, 3, 4) overlap in price and cannot be separated by a single feature.

### 2.3 Feature Engineering Attempt

Three new features were created:

```python
df["price_per_protein"]    = df["price_eur_per_kg"] / df["protein_pct"]
df["fat_to_protein_ratio"] = df["fat_content_pct"] / df["protein_pct"]
df["price_x_marbling"]     = df["price_eur_per_kg"] * df["marbling_score"]
```

**Result: No improvement.** Internal test stayed at 53%, stress test dropped from 27% to 26%. The engineered features combined price (signal) with noise features — noise × signal = noise. Feature engineering only works when the raw features contain hidden patterns that combinations can reveal.

### 2.4 Stress Test – Classification

**Result: 27% accuracy (down from 53%)**

The stress test broke the model because it relies on a single feature (`price_eur_per_kg`). When that feature has outliers or gets imputed from missing values, the tree's price-based splits produce wrong predictions.

### 2.5 Classification Metrics Explained

| Metric | Question it answers |
|---|---|
| **Accuracy** | How often is the model correct overall? |
| **Precision** | When it predicts class X, is it right? |
| **Recall** | Out of all real class-X samples, how many did it find? |
| **F1-Score** | Balance of precision and recall (harmonic mean) |
| **Support** | Number of test samples per class |

---

## 3. Regression – Predicting a Continuous Value

### 3.1 Setup

**Target:** `price_eur_per_kg` (continuous)  
**Features:** All other columns including `meat_type` (0.82 correlation with price)

Key differences from classification:
- Leaves predict the **mean value** of samples, not a class vote
- Split criterion is **MSE** (variance reduction), not Gini impurity
- No `stratify` in train/test split (only works for categories)

### 3.2 Regression Metrics Explained

| Metric | Perfect Score | Interpretation |
|---|---|---|
| **MAE** | 0 | Average absolute error in EUR — "predictions are X EUR off" |
| **RMSE** | 0 | Like MAE but punishes large errors harder (squares them first) |
| **R²** | 1.0 | Proportion of variance explained. 0.0 = no better than guessing the mean |

**If RMSE is much larger than MAE**, the model has some very large outlier errors.  
**If R² is negative**, the model is worse than simply predicting the mean price for every sample.

### 3.3 Grid Search over `max_depth`

A systematic search from depth 2 to 15 revealed the overfitting pattern:

| Depth | R² Test | R² Stress | Leaves |
|---|---|---|---|
| 2 | 0.7002 | -0.3607 | 4 |
| 5 | 0.8428 | -0.4699 | 32 |
| 7 | 0.8737 | -0.2544 | 127 |
| 10 | 0.8717 | -0.3167 | 559 |
| 15 | 0.8749 | -0.2935 | 949 |

**Internal test** keeps improving and plateaus around depth 7–8.  
**Stress test** is negative R² at every depth — the model is always worse than guessing the mean on out-of-distribution data.

### 3.4 Best Single-Depth Result (max_depth=7)

```
Internal Test:  MAE = 2.38 EUR | RMSE = 2.92 | R² = 0.8737
Stress Test:    MAE = 8.59 EUR | RMSE = 11.00 | R² = -0.2544
```

**Feature importance** confirmed the single-feature dependency:

| Feature | Importance |
|---|---|
| meat_type | 70.3% |
| cut_quality | 14.8% |
| marbling_score | 5.6% |
| organic | 4.3% |
| fat_content_pct | 2.0% |
| storage_days | 1.4% |
| animal_age_months | 1.1% |
| protein_pct | 0.5% |

Two features (`meat_type` + `cut_quality`) carry 85% of all decisions.

### 3.5 Hyperparameter Grid Search (GridSearchCV)

**300 combinations tested** across four parameters:

| Parameter | Values Tested | What it Controls |
|---|---|---|
| max_depth | 3, 5, 7, 9, 11 | How deep the tree grows |
| min_samples_split | 2, 5, 10, 20 | Minimum samples to split a node |
| min_samples_leaf | 1, 3, 5, 10, 15 | Minimum samples in each leaf |
| max_features | None, sqrt, log2 | Features considered per split |

**Best parameters found:**

```
max_depth=9, min_samples_leaf=5, min_samples_split=2, max_features=None
```

**Result:**

```
Internal Test:  MAE = 2.18 EUR | RMSE = 2.75 | R² = 0.8875
Stress Test:    MAE = 8.70 EUR | RMSE = 11.25 | R² = -0.3108
```

Internal test improved (R² from 0.8737 to 0.8875), but stress test got slightly worse. The grid search optimised for clean data, not for out-of-distribution robustness.

---

## 4. Key Takeaways

### What we learned about the data
- Only `meat_type` (0.82) and `cut_quality` (0.39) have meaningful correlation with price
- The other 6 features are essentially noise for price prediction
- Feature engineering with noise features does not create signal

### What we learned about Decision Trees
- **Strengths:** Fast, interpretable, automatic feature selection (ignores useless features), handles non-linear relationships
- **Weakness:** Creates rigid boundaries — a single bad split sends a sample to the wrong leaf with no recovery
- **Overfitting pattern:** Internal test improves with depth, stress test stays bad or gets worse
- **Hyperparameter tuning** can squeeze out small gains on clean data but cannot fix the fundamental fragility

### The ceiling of a single Decision Tree
No combination of `max_depth`, `min_samples_split`, `min_samples_leaf`, or `max_features` solved the stress-test problem. 300 grid-search combinations proved this is a structural limitation, not a tuning problem.

### What comes next: Random Forest
A Random Forest trains hundreds of trees on random subsets of rows and features, then averages their predictions. This directly addresses the single tree's weakness: where one tree makes a bad split, the majority of other trees still vote correctly, smoothing out sensitivity to outliers and missing values.

---

## 5. Project Structure

```
project/
├── meat_price_dataset.csv              # 1200 rows – training data
├── meat_price_100_stress_test.csv      # 100 rows – edge cases & missing values
├── decision_tree_classification.ipynb  # Classification notebook (meat_type target)
├── decision_tree_regression.ipynb      # Regression notebook (price target)
└── decision_tree_summary.md            # This file
```