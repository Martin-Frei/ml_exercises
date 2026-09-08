# Random Forest Regression – Summary

## Goal

Predict `price_eur_per_kg` (continuous) using all other features. Beat the Decision Tree baseline (R² = 0.8737 internal, R² = -0.2544 stress test) and achieve positive R² on the stress-test dataset.

---

## The Journey — Step by Step

### Step 1: Random Forest Baseline

**What we did:** Replaced the single Decision Tree with 200 trees averaging their predictions.

**Config:** n_estimators=200, max_depth=7, n_jobs=-1

**Result:**

| Metric | Decision Tree | Random Forest |
|---|---|---|
| MAE (test) | 2.38 EUR | 1.80 EUR |
| R² (test) | 0.8737 | 0.9249 |
| R² (stress) | -0.2544 | -0.3288 |

**Learning:** Averaging 200 trees smoothed out errors on clean data (R² 0.87 → 0.92). But all trees relied on the same dominant feature (meat_type at 69.5%), so the forest lacked diversity and couldn't handle the stress test.

---

### Step 2: Grid Search — Finding Optimal Parameters

**What we did:** GridSearchCV with 360 combinations, then a stress-test focused grid with 375 combinations.

**Key discovery:** `max_features` controls tree diversity. With `sqrt` or `0.3`, each tree considers only ~3 features per split, forcing some trees to learn from weaker features.

**Best stress-test result:**

```
max_features=0.3, depth=7, min_leaf=10
R² Test = 0.8422 | R² Stress = -0.2711
```

**Learning:** Feature diversity pushed stress R² from -0.33 to -0.27 — the best regression stress-test result ever. But it came at the cost of lower internal test accuracy. You can't optimize both simultaneously with this data.

---

### Step 3: Smarter Imputation — KNN vs Median

**What we did:** Replaced crude median fills with KNN imputation (fills NaN based on 5 most similar rows).

**Result:**

| Strategy | R² Stress |
|---|---|
| Median | -0.3288 |
| KNN | -0.2806 |

**Learning:** Smarter fills helped slightly (+0.048 R²). But there aren't enough missing values in the stress test to move the needle significantly. The problem isn't the fills — it's the data itself.

---

### Step 4: Combined Dataset Training

**What we did:** Merged both datasets (1200 + 100 = 1300 rows), imputed NaN with mode (categorical) and median (continuous), then trained on the combined data.

**Result:**

| Metric | Separate training | Combined training |
|---|---|---|
| R² (main rows) | 0.9249 | 0.8841 |
| R² (stress rows) | -0.3288 | -0.3082 |

**Learning:** Adding 100 noisy rows to 1200 clean rows just introduced noise. The model treated stress patterns as outliers rather than learning them. 7.7% stress data isn't enough to change learned patterns.

---

### Step 5: Log Price Transformation

**What we did:** Trained on log(price) instead of raw EUR. The model learns relative differences ("30% more expensive") instead of absolute ones ("8 EUR more").

**Result:**

| Metric | Raw price | Log(price) |
|---|---|---|
| R² (test) | 0.9249 | 0.9252 |
| R² (stress) | -0.3288 | -0.3161 |

**Learning:** Zero effect. The price distribution wasn't skewed enough for log to matter. Log transformation helps when you have many cheap items and a few very expensive ones — this dataset has a relatively even spread.

---

## Learnings — What Worked

| Technique | Impact | Why it worked |
|---|---|---|
| Random Forest over Decision Tree | R² 0.87 → 0.92 on test | 200 trees averaging reduces individual errors |
| max_features diversity | Best stress R² (-0.27) | Forces trees to use different features, creating diverse opinions |
| KNN imputation | Stress R² -0.33 → -0.28 | Context-aware fills are better than one global number |

---

## Dead Ends — What Didn't Work

| Technique | Result | Why it failed |
|---|---|---|
| Outlier clipping | Zero change | Stress test has no extreme individual values — just unusual combinations |
| Combined dataset training | Worse on both | 100 stress rows = noise in 1300 total, confused learned boundaries |
| Log(price) transformation | Zero change | Price not skewed enough to benefit from scale compression |
| Hyperparameter tuning beyond max_features | Marginal gains only | 735 grid-search combinations proved tuning alone can't fix a data problem |

---

## Final Results — Regression

| Experiment | MAE Test | R² Test | MAE Stress | R² Stress |
|---|---|---|---|---|
| DT baseline (depth 7) | 2.38 | 0.8737 | 8.59 | -0.2544 |
| DT GridSearchCV (4320 combos) | 2.18 | 0.8875 | 8.70 | -0.3108 |
| **RF baseline** | **1.80** | **0.9249** | 8.68 | -0.3288 |
| RF GridSearch (max_features=0.3) | — | 0.8422 | **8.46** | **-0.2711** |
| RF + KNN imputation | 1.80 | 0.9249 | 8.64 | -0.2806 |
| RF + KNN + outlier clip | 1.80 | 0.9249 | 8.64 | -0.2806 |
| RF combined dataset | 2.15 | 0.8841 | 8.70 | -0.3082 |
| RF log(price) | 1.80 | 0.9252 | 8.80 | -0.3161 |

**Best internal test:** RF baseline — R² = 0.9249, MAE = 1.80 EUR

**Best stress test:** RF GridSearch (max_features=0.3) — R² = -0.2711

**No configuration achieved positive R² on the stress test.**

---

## Conclusion

The Random Forest improved internal test performance from R² 0.87 (Decision Tree) to R² 0.92 — a meaningful jump. The model predicts meat price within 1.80 EUR on average on clean data.

The stress test remained negative across every experiment. We tried 5 different approaches (grid search, KNN imputation, outlier clipping, combined training, log transformation) and none broke through zero. The stress-test dataset contains price-to-feature relationships that don't exist in the training data.

The core limitation is the dataset itself: only one feature (meat_type, 0.82 correlation) drives price prediction. When the stress test presents unusual price-to-meat-type relationships, no model can predict correctly.

---

## Notebooks

```
random_forest_regression.ipynb            # Baseline
random_forest_grid_search.ipynb           # Grid search
random_forest_outlier_imputation.ipynb    # KNN + clipping
random_forest_combined_dataset.ipynb      # Combined training
random_forest_log_price.ipynb             # Log transformation
```
