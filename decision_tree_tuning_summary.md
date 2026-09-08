# Decision Tree Tuning & Optimization – Summary

## 1. Starting Point

After building both a Classification and Regression Decision Tree, we focused on improving the **Regression model** (predicting `price_eur_per_kg`). The baseline results at `max_depth=7` were:

```
Internal Test:  MAE = 2.38 EUR | R² = 0.8737
Stress Test:    MAE = 8.59 EUR | R² = -0.2544
```

The internal test was solid, but the stress test showed negative R² — the model was worse than simply guessing the mean price on out-of-distribution data.

---

## 2. Optimization Step 1: max_depth Grid (depth 2–15)

We trained 14 models with increasing tree depth and evaluated both datasets.

**Selected results:**

| Depth | Leaves | R² Test | R² Stress |
|---|---|---|---|
| 2 | 4 | 0.7002 | -0.3607 |
| 5 | 32 | 0.8428 | -0.4699 |
| 7 | 127 | 0.8737 | -0.2544 |
| 10 | 559 | 0.8717 | -0.3167 |
| 15 | 949 | 0.8749 | -0.2935 |

**Finding:** Internal test improved and plateaued at depth 7. Stress test was negative at every depth. Deeper trees memorized the training data without improving generalization.

---

## 3. Optimization Step 2: GridSearchCV (300 combinations)

We used `GridSearchCV` to search across four hyperparameters simultaneously.

### What is GridSearchCV?

GridSearchCV is an automated search that tries every combination of parameters you specify. For each combination it uses **Cross-Validation (CV)** — splitting the training data into 5 folds, training on 4 and testing on 1, rotating 5 times — to get a reliable performance estimate. It then picks the combination with the best average CV score.

### Parameters Tested

| Parameter | Values | What it Controls |
|---|---|---|
| `max_depth` | 3, 5, 7, 9, 11 | Maximum number of levels in the tree |
| `min_samples_split` | 2, 5, 10, 20 | Minimum number of samples required to split a node into two children |
| `min_samples_leaf` | 1, 3, 5, 10, 15 | Minimum number of samples that must end up in each leaf |
| `max_features` | None, sqrt, log2 | How many features the tree considers at each split |

### Glossary of Parameters

**max_depth** — limits how many times the tree can split from root to leaf. A depth of 5 means at most 5 decisions before a prediction. Lower = simpler tree, higher = more complex tree.

**min_samples_split** — a node will only split if it contains at least this many samples. Setting it to 20 means a node with 15 samples becomes a leaf automatically, even if the tree hasn't reached max_depth. Forces the tree to stop splitting on small groups.

**min_samples_leaf** — every final leaf must have at least this many samples. If a split would create a leaf with fewer, that split is rejected. Higher values force the tree to make broader, more generalized predictions instead of memorizing small groups.

**max_features** — at each split, the tree only considers a random subset of features. `None` = all features, `sqrt` = square root of the number of features, `log2` = log base 2. Limiting features adds randomness and can reduce overfitting.

### Result

```
Best params: max_depth=9, min_samples_leaf=5, min_samples_split=2, max_features=None
Internal Test:  MAE = 2.18 EUR | R² = 0.8875
Stress Test:    MAE = 8.70 EUR | R² = -0.3108
```

Internal test improved slightly (R² from 0.8737 to 0.8875), but stress test got slightly worse (-0.2544 to -0.3108). The grid search optimized for the training data's structure, not for robustness against edge cases.

---

## 4. Optimization Step 3: Stress-Test Focused Grid (4320 combinations)

Since all previous grids optimized for internal test performance, we ran a grid specifically designed to improve the stress test by searching for **simpler, more conservative trees**.

### Additional Parameters Tested

| Parameter | Values | What it Controls |
|---|---|---|
| `ccp_alpha` | 0.0, 0.1, 0.5, 1.0, 2.0, 5.0 | Cost-complexity pruning strength |

### What is ccp_alpha (Cost-Complexity Pruning)?

Normal tree-building works **top-down**: grow the tree, limit it with max_depth. Pruning works **bottom-up**: grow the tree fully first, then remove branches that don't contribute enough.

`ccp_alpha` controls how aggressively branches are pruned. At 0.0 nothing is pruned. At higher values, any split that doesn't improve the model by at least `ccp_alpha` gets removed. Think of it as a tax on complexity — every branch must "pay for itself" by improving predictions enough to justify its existence.

### Grid Design

The grid intentionally focused on simplicity:
- Low depths: 2–7 (not 9, 11, 15)
- High min_samples_leaf: 5–50 (forcing bigger, more generalized leaves)
- Aggressive pruning: ccp_alpha up to 5.0

### Results: Top 10 by Stress-Test R²

| Depth | min_leaf | ccp_alpha | Leaves | R² Test | R² Stress | MAE Stress |
|---|---|---|---|---|---|---|
| 7 | 5 | 0.0 | 99 | 0.8728 | -0.3100 | 8.77 |
| 7 | 5 | 0.0 | 99 | 0.8728 | -0.3100 | 8.77 |
| 3 | 30 | 1.0 | 6 | 0.7430 | -0.3471 | 8.69 |
| 3 | 50 | 1.0 | 6 | 0.7430 | -0.3471 | 8.69 |
| 3 | 15 | 1.0 | 6 | 0.7430 | -0.3471 | 8.69 |

### Results: Top 10 by Combined Score (R² Test + R² Stress)

| Depth | min_leaf | Leaves | R² Test | R² Stress | Combined |
|---|---|---|---|---|---|
| 7 | 5 | 99 | 0.8728 | -0.3100 | 0.5628 |
| 7 | 5 | 68 | 0.8681 | -0.3118 | 0.5563 |
| 7 | 10 | 66 | 0.8717 | -0.3548 | 0.5169 |
| 6 | 5 | 63 | 0.8634 | -0.3624 | 0.5010 |

---

## 5. Key Finding: Every Configuration Failed on the Stress Test

Out of **4320 parameter combinations**, not a single one achieved a positive R² on the stress test. The best stress-test R² was -0.31, meaning even the optimal Decision Tree is worse than guessing the mean price.

### Why This Happens

A Decision Tree creates **rigid boundaries**. At each split, it draws one hard line:

```
If meat_type <= 2.5  →  go left  →  predict 16.40 EUR
If meat_type >  2.5  →  go right →  predict 28.70 EUR
```

When a stress-test sample has an unusual combination — missing values imputed with medians, extreme outlier prices, or uncommon feature ranges — it crosses the wrong boundary and lands in a leaf meant for completely different samples. There is no safety net, no second opinion, no correction mechanism.

### The Pruning Paradox

We expected simpler trees to generalize better, but:
- Simple trees (depth 3, 6 leaves): R² test = 0.74, R² stress = -0.35
- Complex trees (depth 7, 99 leaves): R² test = 0.87, R² stress = -0.31

Simpler trees lost accuracy on the internal test without gaining anything on the stress test. The problem is not tree complexity — it's the **structural limitation** of a single tree making one hard decision at each split.

---

## 6. Glossary of All Key Terms

### Model Parameters

| Term | Definition |
|---|---|
| **max_depth** | Maximum levels from root to leaf. Controls tree complexity. |
| **min_samples_split** | Minimum samples needed in a node before it can be split further. |
| **min_samples_leaf** | Minimum samples required in each final leaf node. |
| **max_features** | Number of features considered at each split (None = all, sqrt, log2). |
| **ccp_alpha** | Pruning strength. Higher values remove more branches after growing. |
| **criterion** | The function used to measure split quality (squared_error, friedman_mse, absolute_error). |
| **random_state** | Seed for reproducibility. Same value = same results every run. |

### Evaluation Metrics

| Term | Definition |
|---|---|
| **MAE** | Mean Absolute Error — average prediction error in the same unit as the target (EUR). |
| **RMSE** | Root Mean Squared Error — like MAE but punishes large errors more heavily. |
| **R²** | Coefficient of determination — 1.0 = perfect, 0.0 = no better than guessing the mean, negative = worse than guessing. |
| **Relative Error** | MAE divided by the mean target value, expressed as percentage. |

### Training Concepts

| Term | Definition |
|---|---|
| **GridSearchCV** | Automated search trying every parameter combination with cross-validation. |
| **Cross-Validation (CV)** | Splitting training data into k folds, training on k-1 and testing on 1, rotating k times. |
| **Overfitting** | Model memorizes training data patterns that don't generalize to new data. |
| **Generalization** | Model's ability to perform well on data it hasn't seen during training. |
| **Stratify** | Ensuring each class has the same proportion in train and test sets (classification only). |
| **Imputation** | Filling missing values with estimated values (median, KNN, etc.). |

### Tree Structure

| Term | Definition |
|---|---|
| **Root Node** | The very first split at the top of the tree. |
| **Internal Node** | A node that splits into two children based on a feature condition. |
| **Leaf Node** | A final node that makes a prediction (class vote or mean value). |
| **Depth** | Number of splits from root to the deepest leaf. |
| **Pruning** | Removing branches that don't improve the model enough to justify their complexity. |
| **Feature Importance** | How much each feature contributed to reducing prediction error across all splits. |

---

## 7. Conclusion

The Decision Tree Regressor achieves strong results on clean, in-distribution data (R² = 0.88) but fundamentally cannot handle out-of-distribution data (stress test R² always negative). This is a **structural limitation**, not a tuning problem — 4320 hyperparameter combinations proved that no configuration solves it.

This makes the case for **ensemble methods** like Random Forest: by averaging hundreds of trees trained on random subsets, individual split errors are smoothed out, directly addressing the single tree's fragility.