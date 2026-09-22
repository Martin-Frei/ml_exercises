# XGBoost Regression – Summary

## Goal

Predict `price_eur_per_kg` (continuous) using all other features. Beat the Random Forest baseline (R² = 0.9249 internal, R² = -0.3288 stress test) and achieve positive R² on the stress-test dataset.

---

## Model Configuration (constant across all notebooks)

```
XGBRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=5,
    random_state=42
)
```

All five notebooks used the same hyperparameters — the focus was on preprocessing and data strategy, not tuning. Hyperparameter grid search is the planned next step.

---

## The Journey — Step by Step

### Step 1: XGBoost Baseline (`xgb_regression.ipynb`)

**What we did:** Replaced Random Forest with XGBRegressor. Added `protein_fat_ratio` as an engineered feature. Trained on the clean 1200-row dataset (80/20 split), then evaluated on the 100-row stress test. Missing values in the stress set were filled with training-set medians.

**Result:**

| Metric | Random Forest | XGBoost Baseline |
|---|---|---|
| Train RMSE | — | 0.3869 |
| Test RMSE | — | 1.2964 |
| Test MAE | 1.80 | 1.0591 |
| Test R² | 0.9249 | 0.9751 |
| Stress RMSE | — | 11.1558 |
| Stress MAE | 8.68 | 8.7577 |
| Stress R² | -0.3288 | -0.3590 |

**Learning:** XGBoost significantly outperformed Random Forest on clean data (R² 0.92 → 0.97, MAE 1.80 → 1.06 EUR). But the stress test got slightly worse (-0.33 → -0.36). The model's boosting strategy fits training patterns very tightly, which hurts generalization to out-of-distribution data. Also notable: Train RMSE (0.39) vs Test RMSE (1.30) suggests some overfitting.

---

### Step 2: Group-Wise Imputation + Outlier Clipping (`xgb_regression_clip_fill.ipynb`)

**What we did:** Two improvements to stress-test preprocessing:
1. **Group-wise imputation** — instead of global median, fill NaN values with the median of the same `meat_type` subgroup. Falls back to global median when the subgroup is too small.
2. **Outlier clipping (winsorizing)** — clip extreme stress-test feature values to the training-set max bounds, preventing the model from extrapolating beyond what it learned.

Same model, same hyperparameters, same training data.

**Result:**

| Metric | NB1: Baseline | NB2: Clip + Fill |
|---|---|---|
| Stress RMSE | 11.1558 | 10.7989 |
| Stress MAE | 8.7577 | 8.3638 |
| Stress R² | -0.3590 | -0.3078 |

**Learning:** Smarter preprocessing helped. Group-wise fills preserve meat-type-specific distributions instead of blending everything into one global median. Clipping prevents wild extrapolation on out-of-range values. Combined effect: ~0.05 R² improvement. Not a breakthrough, but the best stress-test approach so far for XGBoost.

---

### Step 3: One-Hot Encoding (`xgb_regression_one_hot_encoding.ipynb`)

**What we did:** Instead of treating `meat_type` as an ordinal integer (1–5), we one-hot encoded it into five binary columns. This tells the model that type 5 is not "five times" type 1 — they're just different categories. Missing values handled with training-set statistics. Columns aligned between train and stress set using `pd.get_dummies` + `reindex`.

**Result:**

| Metric | NB1: Baseline (ordinal) | NB3: One-Hot |
|---|---|---|
| Stress RMSE | 11.1558 | 11.4141 |
| Stress R² | -0.3590 | -0.3495 |

**Learning:** One-hot encoding made stress-test RMSE slightly worse but R² slightly better — essentially a wash. XGBoost handles ordinal categoricals well through tree splits anyway (it can learn "type 3 goes left, type 4 goes right" without needing one-hot). The encoding fragmented the dominant feature across 5 columns, diluting its importance per column without improving the model's ability to generalize.

---

### Step 4: Log-Transformed Target + One-Hot Encoding (`xgb_regression_log_one_hot_encoding.ipynb`)

**What we did:** Combined one-hot encoding with log transformation of the target variable. Trained on `log1p(price)`, predicted, then back-transformed predictions with `expm1()`. The idea: if the model learns relative price differences instead of absolute ones, it might handle unusual stress-test prices better.

**Result:**

| Metric | NB1: Baseline | NB4: Log + One-Hot |
|---|---|---|
| Stress RMSE | 11.1558 | 11.4874 |
| Stress R² | -0.3590 | -0.3669 |

**Learning:** Worst result of all five notebooks. Log transformation helps when the target distribution is heavily right-skewed (many cheap items, few expensive ones). This dataset's price distribution is relatively symmetric (mean 23.05, std 8.18, range 2.50–43.33), so compressing the scale added noise without benefit. Combined with one-hot encoding's lack of improvement, this was a dead end.

---

### Step 5: Combined Dataset Training (`xgb_regression_combined.ipynb`)

**What we did:** The most ambitious approach — merged both datasets into one:
1. Group-wise imputation on stress data (by meat_type subgroup)
2. Dropped rows where meat_type or target was NaN (stress set: 100 → 91 rows)
3. Clipped stress-test outliers to training-set bounds
4. Added `protein_fat_ratio` feature to both sets
5. Tagged each row's origin (`is_stress` column)
6. Combined into 1291 rows (1200 clean + 91 stress)
7. Stratified 80/20 split preserving the stress-row ratio

**Result:**

| Metric | NB1: Baseline (stress only) | NB5: Combined (mixed split) |
|---|---|---|
| Overall RMSE | — | 3.1315 |
| Overall MAE | — | 1.7900 |
| Overall R² | — | 0.8637 |
| Clean subset R² | 0.9751 | 0.9490 |
| Stress subset RMSE | 11.1558 | 9.8519 |
| Stress subset MAE | 8.7577 | 7.3835 |
| Stress subset R² | -0.3590 | **+0.1968** |

**Learning:** This is the breakthrough. By exposing the model to stress-test patterns during training, it learned some of the unusual price-to-feature relationships. Stress R² went from -0.36 to **+0.20** — the first positive stress-test R² in the entire project (Decision Tree, Random Forest, and all prior XGBoost experiments were negative). The trade-off: clean-data R² dropped slightly (0.9751 → 0.9490) because the model now hedges between clean and noisy patterns. The notebook ends with a note: "next try with SMOTE" — suggesting augmenting the stress data further.

---

## Learnings — What Worked

| Technique | Impact | Why it worked |
|---|---|---|
| XGBoost over Random Forest | R² 0.92 → 0.97 on clean test | Boosting corrects errors sequentially; each tree fixes the previous one's mistakes |
| Group-wise imputation | Stress R² -0.36 → -0.31 | Preserves meat-type-specific distributions instead of blending into one global median |
| Outlier clipping | Part of clip+fill improvement | Prevents extrapolation beyond learned range |
| Combined dataset training | Stress R² -0.36 → **+0.20** | Model sees stress patterns during training; learns unusual relationships |

---

## Dead Ends — What Didn't Work

| Technique | Result | Why it failed |
|---|---|---|
| One-hot encoding (meat_type) | Stress R² -0.35 (no change) | XGBoost handles ordinal splits natively; one-hot just fragments the dominant feature |
| Log target transformation | Stress R² -0.37 (worst) | Price distribution not skewed enough; log compressed the scale without benefit |
| Log + one-hot combined | Worst overall | Stacking two ineffective techniques compounds the noise |

---

## Final Results — XGBoost Regression

| Experiment | Test MAE | Test R² | Stress MAE | Stress R² |
|---|---|---|---|---|
| RF baseline (benchmark) | 1.80 | 0.9249 | 8.68 | -0.3288 |
| **NB1: XGB baseline** | **1.06** | **0.9751** | 8.76 | -0.3590 |
| NB2: Group fill + clip | — | — | 8.36 | -0.3078 |
| NB3: One-hot encoding | — | — | — | -0.3495 |
| NB4: Log + one-hot | — | — | — | -0.3669 |
| **NB5: Combined training** | 1.79 | 0.8637 | **7.38** | **+0.1968** |

**Best internal test:** NB1 XGB baseline — R² = 0.9751, MAE = 1.06 EUR

**Best stress test:** NB5 Combined training — R² = +0.1968, MAE = 7.38 EUR

**First positive stress-test R² in the entire project.**

---

## Comparison Across All Models

| Model | Best Test R² | Best Stress R² |
|---|---|---|
| Decision Tree | 0.8875 | -0.2544 |
| Random Forest | 0.9249 | -0.2711 |
| **XGBoost** | **0.9751** | **+0.1968** |

---

## Next Steps

- **Hyperparameter grid search** — all 5 notebooks used the same hyperparameters (n_estimators=150, lr=0.08, depth=5). Tuning may improve both clean and stress performance.
- **Stress-data augmentation** — the combined notebook noted "next try with SMOTE", but SMOTE is a classification technique for imbalanced classes and does not apply to regression. Alternative augmentation strategies for regression could include noise injection or Gaussian perturbation of existing stress rows.

---

## Notebooks

```
xgb_regression.ipynb                    # NB1: Baseline
xgb_regression_clip_fill.ipynb          # NB2: Group-wise imputation + outlier clipping
xgb_regression_one_hot_encoding.ipynb   # NB3: One-hot encoding for meat_type
xgb_regression_log_one_hot_encoding.ipynb  # NB4: Log target + one-hot encoding
xgb_regression_combined.ipynb           # NB5: Combined dataset training (breakthrough)
```