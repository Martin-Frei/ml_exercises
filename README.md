**English** | [Deutsch](README.de.md)

# ML Exercises — Meat Price Dataset

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20progress-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Decision Tree → Random Forest → XGBoost: R² 0.87 → 0.92 → 0.97 on clean data — and the first positive R² (+0.20) on out-of-distribution stress data.**

Hands-on machine learning exercises built around one synthetic meat industry dataset. The same data is used across several algorithms so that every model, technique and preprocessing idea can be compared on equal terms.

The repository documents the full learning journey: what was tried, what worked, what failed, and why. Failures are documented as carefully as successes, because they explain the limits of the data.

---

## Table of Contents

1. [The Dataset](#the-dataset)
2. [The Two Tasks](#the-two-tasks)
3. [Metrics Explained](#metrics-explained)
4. [Repository Structure](#repository-structure)
5. [Results at a Glance](#results-at-a-glance)
6. [What Worked](#what-worked)
7. [Dead Ends](#dead-ends)
8. [Why the Stress Test Is So Hard](#why-the-stress-test-is-so-hard)
9. [Key Lessons](#key-lessons)
10. [Getting Started](#getting-started)
11. [Adding a New Experiment](#adding-a-new-experiment)
12. [Conventions](#conventions)
13. [Roadmap](#roadmap)
14. [License](#license)

---

## The Dataset

Two CSV files with identical columns:

| File | Rows | Purpose |
|---|---|---|
| `meat_price_dataset.csv` | 1,200 | Clean training and test data |
| `meat_price_100_stress_test.csv` | 100 | Out-of-distribution data with missing values, outliers and unusual feature combinations |

| Column | Type | Description |
|---|---|---|
| `meat_type` | int | Meat category (1 = cheapest, 5 = most expensive) |
| `fat_content_pct` | float | Fat percentage |
| `protein_pct` | float | Protein percentage |
| `marbling_score` | int | Marbling grade |
| `animal_age_months` | int | Animal age in months |
| `storage_days` | int | Days in storage |
| `organic` | int | Organic flag (0/1) |
| `cut_quality` | int | Cut quality grade (1–5) |
| `price_eur_per_kg` | float | Price in EUR per kg |

### Clean data vs. stress data

The two files cover noticeably different value ranges:

| Column | Range (clean) | Range (stress) | Missing (stress) | Stress values outside clean range |
|---|---|---|---|---|
| `meat_type` | 1–5 | 1–5 | 9 | 0 |
| `fat_content_pct` | 2–35 | 2–**58.5** | 14 | 3 |
| `protein_pct` | 15–26 | 15.1–25 | 10 | 0 |
| `marbling_score` | 1–10 | **0**–**12** | 11 | 2 |
| `animal_age_months` | 2–71 | 2–**189** | 8 | 2 |
| `storage_days` | 0–20 | 0–**80** | 9 | 2 |
| `organic` | 0/1 | 0/1 | 9 | 0 |
| `cut_quality` | 1–5 | 1–5 | 7 | 0 |
| `price_eur_per_kg` | 2.50–43.33 | 12.14–**65.40** | 9 | 6 |

Clean prices: mean 23.05 EUR, std 8.18. Stress prices: mean 29.97 EUR, std 9.88.

The **stress test** is the hard part of this project. Every column contains missing values, some rows contain values far outside the clean range, and — most importantly — prices are systematically higher than in the clean data (see [Why the Stress Test Is So Hard](#why-the-stress-test-is-so-hard)). It measures how well a model generalizes beyond the patterns it was trained on.

**Important data fact:** only `meat_type` (correlation 0.82 with price) and `cut_quality` (0.39) carry meaningful signal. The other six features behave mostly like noise. This single-feature dependency explains most of the results below.

---

## The Two Tasks

| Task | Target | Type | Main metrics |
|---|---|---|---|
| Classification | `meat_type` (1–5) | Multi-class | Accuracy, per-class F1 |
| Regression | `price_eur_per_kg` | Continuous | MAE, RMSE, R² |

Every experiment is evaluated twice: on an internal test split (20% of the clean data) and on the stress-test dataset.

---

## Metrics Explained

### Classification

| Metric | Question it answers |
|---|---|
| **Accuracy** | How often is the model correct overall? |
| **Precision** | When the model predicts class X, how often is it right? |
| **Recall** | Of all real class-X samples, how many did the model find? |
| **F1-Score** | Balance of precision and recall (harmonic mean) |

With 5 classes, random guessing gives about 20% accuracy.

### Regression

| Metric | Perfect score | Meaning |
|---|---|---|
| **MAE** | 0 | Average absolute error in EUR — "predictions are X EUR off" |
| **RMSE** | 0 | Like MAE, but large errors are punished harder (they are squared first) |
| **R²** | 1.0 | Share of the price variance the model explains |

Two rules of thumb:

- **R² = 0** means the model is no better than always predicting the average price. **Negative R²** means it is *worse* than that.
- **RMSE much larger than MAE** means the model makes a few very large errors.

---

## Repository Structure

All files live in the repository root. Files are grouped by algorithm through their name prefix.

```
ml_exercises/
│
├── meat_price_dataset.csv                          # 1,200 rows, clean data
├── meat_price_100_stress_test.csv                  # 100 rows, edge cases + missing values
│
├── decision_tree_classification_meat_type.ipynb    # DT classification (meat_type)
├── decision_tree_regression_price_per_kilo.ipynb   # DT regression (price)
├── decision_tree_summary.md                        # DT results + explanations
├── decision_tree_tuning_summary.md                 # DT grid searches + parameter glossary
│
├── random_forest_classification.ipynb              # RF baseline + domain-based imputation
├── random_forest_target_encoding.ipynb             # RF + target encoding (best classifier)
├── random_forest_regression.ipynb                  # RF regression baseline
├── random_forest_regression_grid_search.ipynb      # RF grid search
├── random_forest_regression_outlier_imputation.ipynb  # KNN imputation + outlier clipping
├── random_forest_combined_dataset.ipynb            # Training on clean + stress rows
├── random_forest_log_price.ipynb                   # log(price) target
├── rf_classification_summary.md                    # RF classification journey
├── rf_regression_summary.md                        # RF regression journey
│
├── xgb_regression.ipynb                            # NB1: XGBoost baseline
├── xgb_regression_clip_fill.ipynb                  # NB2: group-wise imputation + clipping
├── xgb_regression_one_hot_encoding.ipynb           # NB3: one-hot encoded meat_type
├── xgb_regression_log_one_hot_encoding.ipynb       # NB4: log target + one-hot
├── xgb_regression_combined.ipynb                   # NB5: combined dataset training
├── xgb_regression_summary.md                       # XGBoost regression journey
│
├── README.md                                       # English
├── README.de.md                                    # German
├── LICENSE
└── .gitignore
```

Each algorithm has a **summary file** that explains the experiments in detail. Start there before opening the notebooks. The full glossary of tree hyperparameters (`max_depth`, `min_samples_leaf`, `ccp_alpha`, …) is in `decision_tree_tuning_summary.md`.

---

## Results at a Glance

### Classification — predicting `meat_type`

| Experiment | Accuracy (test) | Accuracy (stress) | Notebook |
|---|---|---|---|
| DT, wrong target (`cut_quality`) | 22.5% | — | `decision_tree_classification_meat_type.ipynb` |
| DT baseline (`meat_type`) | 53.0% | 27.0% | `decision_tree_classification_meat_type.ipynb` |
| DT + ratio feature engineering | 53.0% | 26.0% | `decision_tree_classification_meat_type.ipynb` |
| RF baseline (KNN imputation) | 56.0% | 24.2% | `random_forest_classification.ipynb` |
| RF + domain-based imputation | 56.0% | **27.5%** | `random_forest_classification.ipynb` |
| **RF + target encoding** | **62.5%** | 25.0% | `random_forest_target_encoding.ipynb` |

**Per-class F1 — the middle classes are the problem:**

| Class | DT | RF baseline | RF + target encoding |
|---|---|---|---|
| 1 (cheapest) | 0.65 | 0.74 | 0.74 |
| 2 | 0.39 | 0.48 | **0.55** |
| 3 | 0.49 | 0.39 | **0.51** |
| 4 | 0.24 | 0.42 | **0.54** |
| 5 (most expensive) | 0.77 | 0.73 | 0.75 |

### Regression — predicting `price_eur_per_kg`

| Model | Experiment | MAE test | R² test | MAE stress | R² stress |
|---|---|---|---|---|---|
| DT | Baseline (depth 7) | 2.38 | 0.8737 | 8.59 | -0.2544 |
| DT | GridSearchCV (best params) | 2.18 | 0.8875 | 8.70 | -0.3108 |
| RF | Baseline | 1.80 | 0.9249 | 8.68 | -0.3288 |
| RF | Grid search (`max_features=0.3`) | — | 0.8422 | 8.46 | -0.2711 |
| RF | KNN imputation | 1.80 | 0.9249 | 8.64 | -0.2806 |
| RF | KNN imputation + outlier clipping | 1.80 | 0.9249 | 8.64 | -0.2806 |
| RF | Combined dataset training | 2.15 | 0.8841 | 8.70 | -0.3082 |
| RF | log(price) target | 1.80 | 0.9252 | 8.80 | -0.3161 |
| **XGB** | **NB1: Baseline** | **1.06** | **0.9751** | 8.76 | -0.3590 |
| XGB | NB2: Group-wise fill + clipping | — | — | 8.36 | -0.3078 |
| XGB | NB3: One-hot `meat_type` | — | — | — | -0.3495 |
| XGB | NB4: log target + one-hot | — | — | — | -0.3669 |
| **XGB** | **NB5: Combined dataset training** | 1.79 | 0.8637 | **7.38** | **+0.1968** |

MAE values are in EUR per kg.

### Best result per model

| Model | Best R² test | Best R² stress |
|---|---|---|
| Decision Tree | 0.8875 | -0.2544 |
| Random Forest | 0.9249 | -0.2711 |
| **XGBoost** | **0.9751** | **+0.1968** |

**Note on the NB5 result:** it is the first positive stress-test R² in the project. It was measured on the stress rows inside the 20% test split of the combined data (roughly 18 rows), so the number is informative but noisy. It should be confirmed with cross-validation or repeated splits before it is treated as final.

---

## What Worked

| Technique | Impact | Why it worked |
|---|---|---|
| Correct target selection | Classification 22.5% → 53% | Correlation analysis showed where the signal actually is |
| Random Forest over Decision Tree | R² 0.87 → 0.92, accuracy 53% → 56% | Averaging 200 trees reduces the errors of single trees |
| `max_features` diversity (RF) | Best RF stress R² (-0.27) | Forces trees to also learn from weaker features |
| Domain-based imputation | Classification stress 24% → 27.5% | Missing values are filled with the median of the same `meat_type`, not a global value |
| Target encoding | Classification 56% → 62.5% | Encoded features capture group-level patterns the raw numbers cannot express |
| XGBoost over Random Forest | R² 0.92 → 0.97 | Boosting builds trees sequentially; each one corrects the previous errors |
| Group-wise fill + clipping (XGB) | Stress R² -0.36 → -0.31 | Realistic fills and no extrapolation beyond the training range |
| Combined dataset training (XGB) | Stress R² -0.36 → **+0.20** | The model sees stress patterns during training |

---

## Dead Ends

| Technique | Result | Why it failed |
|---|---|---|
| Ratio feature engineering | No gain, stress 27% → 26% | Price (signal) combined with fat/protein (noise) gives noise: *noise × signal = noise* |
| KNN imputation (classification) | Stress 27% → 24% | The forest uses more features, so bad fills have more chances to mislead |
| Simpler, pruned trees | R² test 0.87 → 0.74, stress unchanged | See "The pruning paradox" below |
| Outlier clipping (RF) | Zero change | Only 2–6 values per feature lie outside the clean range, and the dominant features (`meat_type`, `cut_quality`) have none. Clipping features also cannot fix the real problem: the shifted price level |
| Combined dataset training (RF) | Worse on both sets | 100 stress rows among 1,300 were treated as noise (see open question below) |
| log(price) target | Zero change or worse | Price is roughly symmetric, so log scaling adds nothing |
| One-hot encoding `meat_type` (XGB) | No change | Trees already split ordinal categories well; one-hot only fragments the strongest feature |
| Hyperparameter tuning | Marginal gains only | 300 + 4,320 DT combinations and 360 + 375 RF combinations never produced a positive stress R² |

**Open question — why did combined training fail with RF but work with XGBoost?**
One likely difference: the RF notebook filled missing prices in the stress rows with the median, while the XGBoost notebook dropped rows with a missing target. Imputed target values are fake labels, which may have taught the RF wrong patterns. This is a hypothesis and has not been tested yet.

---

## Why the Stress Test Is So Hard

**1. The prices are shifted — the main reason.** For every meat type, stress-test prices are higher on average than in the clean data:

| `meat_type` | Mean price (clean) | Mean price (stress) | Difference |
|---|---|---|---|
| 1 | 14.03 | 22.10 | +8.07 |
| 2 | 16.44 | 26.62 | +10.18 |
| 3 | 24.20 | 32.20 | +8.00 |
| 4 | 27.24 | 33.28 | +6.04 |
| 5 | 32.48 | 33.57 | +1.09 |
| **All** | **23.05** | **29.97** | **+6.92** |

A model trained only on clean data learns the clean price level and therefore predicts stress prices around 7 EUR too low on average — no matter how it is tuned. This explains why no DT or RF configuration reached a positive stress R², and why combined training (XGB NB5), where the model sees the higher prices during training, was the only approach that did.

The shift also hurts classification: in the stress data, types 3, 4 and 5 all sit around 33 EUR, so price can no longer separate them.

**2. Rigid boundaries.** A decision tree makes one hard decision at every split:

```
If meat_type <= 2.5  →  go left  →  predict 16.40 EUR
If meat_type >  2.5  →  go right →  predict 28.70 EUR
```

A stress-test row with an unusual combination crosses the wrong boundary and lands in a leaf meant for completely different samples. A single tree has no second opinion and no way to correct the mistake.

**3. The pruning paradox.** Simpler trees were expected to generalize better, but a depth-3 tree (6 leaves) reached stress R² -0.35 while a depth-7 tree (99 leaves) reached -0.31. Simplifying cost accuracy on clean data and gained nothing on the stress test. The problem is not model complexity.

**4. Extreme classes are easy, middle classes overlap.** Classes 1 and 5 sit at clearly separate price ranges and reach F1 around 0.75. Classes 2, 3 and 4 overlap in price and cannot be separated by price alone. That is why target encoding, which adds new ways to separate these groups, helped the middle classes most (class 4: F1 0.24 → 0.54).

**5. One feature dominates.** Feature importance shows how dependent the models are on a single feature:

| Model | Top feature | Importance |
|---|---|---|
| Decision Tree (regression) | `meat_type` | 70.3% |
| Random Forest (regression) | `meat_type` | 69.5% |
| Random Forest (classification) | `price_eur_per_kg` | 52% |
| RF + target encoding | `price_eur_per_kg` / `price_bin_te` | 33.7% / 21.6% |

When the dominant feature is missing, imputed or unusual in a stress-test row, the prediction breaks.

**6. XGBoost fits tightly.** The XGBoost baseline reached train RMSE 0.39 but test RMSE 1.30, which suggests some overfitting. All five XGBoost notebooks used the same untuned configuration (`n_estimators=150`, `learning_rate=0.08`, `max_depth=5`), so tuning is still open.

---

## Key Lessons

1. **Understand the data before modeling.** A correlation check would have shown that `cut_quality` was an unlearnable target before any model was trained.
2. **Better models help on clean data.** Decision Tree → Random Forest → XGBoost improved R² from 0.87 to 0.97.
3. **Better models do not fix out-of-distribution data.** Thousands of hyperparameter combinations never produced a positive stress R² with DT or RF. The stress prices are about 7 EUR higher on average — a data problem, not a tuning problem.
4. **Compare the distributions of train and test data first.** A simple per-group comparison of the two datasets reveals the price shift immediately and explains most of the stress-test results.
5. **Domain knowledge beats generic imputation.** Filling missing values per `meat_type` group gave the best stress-test results in both tasks.
6. **Target encoding was the strongest feature technique** for classification.
7. **Popular techniques are not universal.** Log transforms help skewed targets and one-hot encoding helps linear models. Neither applied here.
8. **Showing the model the hard cases worked.** Combined training with XGBoost was the only approach with a positive stress R².
9. **You usually cannot maximize both.** The best configuration on clean data is rarely the best on the stress test.

---

## Getting Started

### Requirements

- Python 3.10 or newer
- Jupyter (Notebook, JupyterLab or VS Code)

### Installation

```bash
git clone https://github.com/Martin-Frei/ml_exercises.git
cd ml_exercises

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### Suggested reading order

1. `decision_tree_summary.md` → `decision_tree_tuning_summary.md`
2. `rf_classification_summary.md` → `rf_regression_summary.md`
3. `xgb_regression_summary.md`

Then open the notebooks in the order listed in each summary. Every notebook contains step-by-step markdown explanations and can be run top to bottom on its own.

---

## Adding a New Experiment

This README is designed to grow with the project. For every new experiment:

1. Create **one new notebook** for the new process or step. Name it `<algorithm>_<task>_<technique>.ipynb`, e.g. `xgb_regression_hyperparameter_tuning.ipynb`.
2. Explain every step in markdown cells.
3. Use relative paths to the CSV files, never machine-specific absolute paths.
4. When an experiment block is finished, write `<algorithm>_<task>_summary.md`.
5. Update both READMEs (English and German): add the files to [Repository Structure](#repository-structure), add rows to [Results at a Glance](#results-at-a-glance), and add to [What Worked](#what-worked) or [Dead Ends](#dead-ends).

For a new algorithm (e.g. LightGBM, KNN, neural networks), add a new block to the structure tree and a new row to "Best result per model".

---

## Conventions

1. Code, notebooks, comments, summaries and filenames are written in **English**. A German README ([README.de.md](README.de.md)) is provided for convenience.
2. Every new process or step gets **its own notebook**. Processes are never combined in one notebook.
3. Every notebook includes step-by-step explanations in markdown cells.

---

## Roadmap

- [x] Decision Tree — classification and regression
- [x] Random Forest — classification and regression
- [x] XGBoost — regression (5 notebooks)
- [ ] XGBoost — classification
- [ ] XGBoost — hyperparameter tuning on the combined dataset (with a leakage-free stress evaluation)
- [ ] Robust evaluation of the stress-test result (cross-validation, repeated splits)
- [ ] Stress-data augmentation for regression (noise injection / Gaussian perturbation — SMOTE only applies to classification)
- [ ] Test the RF vs XGBoost combined-training hypothesis
- [ ] `requirements.txt`

---

## Acknowledgments

This project was developed as part of ML tutoring sessions with [Adeena](https://github.com/Adeenasamoo), who guided the experiments and reviewed the results.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

**Author:** Martin Freimuth — [GitHub](https://github.com/Martin-Frei)