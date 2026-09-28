[English](README.md) | **Deutsch**

# ML Exercises — Fleischpreis-Datensatz

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20progress-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Decision Tree → Random Forest → XGBoost: R² 0,87 → 0,92 → 0,97 auf sauberen Daten — und das erste positive R² (+0,20) auf Out-of-Distribution-Stressdaten.**

Praktische Machine-Learning-Übungen rund um einen synthetischen Datensatz aus der Fleischbranche. Derselbe Datensatz wird für mehrere Algorithmen verwendet, damit sich jedes Modell, jede Technik und jede Vorverarbeitungsidee unter gleichen Bedingungen vergleichen lässt.

Das Repository dokumentiert den gesamten Lernweg: was ausprobiert wurde, was funktioniert hat, was gescheitert ist und warum. Fehlschläge sind genauso sorgfältig dokumentiert wie Erfolge, weil sie die Grenzen der Daten erklären.

> Hinweis: Code, Notebooks und Zusammenfassungen (`*_summary.md`) sind auf Englisch.

---

## Inhaltsverzeichnis

1. [Der Datensatz](#der-datensatz)
2. [Die zwei Aufgaben](#die-zwei-aufgaben)
3. [Metriken erklärt](#metriken-erklärt)
4. [Repository-Struktur](#repository-struktur)
5. [Ergebnisse im Überblick](#ergebnisse-im-überblick)
6. [Was funktioniert hat](#was-funktioniert-hat)
7. [Sackgassen](#sackgassen)
8. [Warum der Stresstest so schwer ist](#warum-der-stresstest-so-schwer-ist)
9. [Wichtigste Erkenntnisse](#wichtigste-erkenntnisse)
10. [Erste Schritte](#erste-schritte)
11. [Ein neues Experiment hinzufügen](#ein-neues-experiment-hinzufügen)
12. [Konventionen](#konventionen)
13. [Roadmap](#roadmap)
14. [Lizenz](#lizenz)

---

## Der Datensatz

Zwei CSV-Dateien mit identischen Spalten:

| Datei | Zeilen | Zweck |
|---|---|---|
| `meat_price_dataset.csv` | 1.200 | Saubere Trainings- und Testdaten |
| `meat_price_100_stress_test.csv` | 100 | Out-of-Distribution-Daten mit fehlenden Werten, Ausreißern und ungewöhnlichen Merkmalskombinationen |

| Spalte | Typ | Beschreibung |
|---|---|---|
| `meat_type` | int | Fleischkategorie (1 = am günstigsten, 5 = am teuersten) |
| `fat_content_pct` | float | Fettanteil in Prozent |
| `protein_pct` | float | Proteinanteil in Prozent |
| `marbling_score` | int | Marmorierungsgrad |
| `animal_age_months` | int | Alter des Tieres in Monaten |
| `storage_days` | int | Lagertage |
| `organic` | int | Bio-Kennzeichen (0/1) |
| `cut_quality` | int | Qualitätsstufe des Zuschnitts (1–5) |
| `price_eur_per_kg` | float | Preis in EUR pro kg |

### Saubere Daten vs. Stressdaten

Die beiden Dateien decken deutlich unterschiedliche Wertebereiche ab:

| Spalte | Bereich (sauber) | Bereich (Stress) | Fehlend (Stress) | Stresswerte außerhalb des sauberen Bereichs |
|---|---|---|---|---|
| `meat_type` | 1–5 | 1–5 | 9 | 0 |
| `fat_content_pct` | 2–35 | 2–**58,5** | 14 | 3 |
| `protein_pct` | 15–26 | 15,1–25 | 10 | 0 |
| `marbling_score` | 1–10 | **0**–**12** | 11 | 2 |
| `animal_age_months` | 2–71 | 2–**189** | 8 | 2 |
| `storage_days` | 0–20 | 0–**80** | 9 | 2 |
| `organic` | 0/1 | 0/1 | 9 | 0 |
| `cut_quality` | 1–5 | 1–5 | 7 | 0 |
| `price_eur_per_kg` | 2,50–43,33 | 12,14–**65,40** | 9 | 6 |

Saubere Preise: Mittelwert 23,05 EUR, Std.-Abw. 8,18. Stresspreise: Mittelwert 29,97 EUR, Std.-Abw. 9,88.

Der **Stresstest** ist der schwierige Teil dieses Projekts. Jede Spalte enthält fehlende Werte, einige Zeilen enthalten Werte weit außerhalb des sauberen Bereichs, und — am wichtigsten — die Preise liegen systematisch höher als in den sauberen Daten (siehe [Warum der Stresstest so schwer ist](#warum-der-stresstest-so-schwer-ist)). Er misst, wie gut ein Modell über die gelernten Muster hinaus generalisiert.

**Wichtige Eigenschaft der Daten:** Nur `meat_type` (Korrelation 0,82 mit dem Preis) und `cut_quality` (0,39) tragen nennenswertes Signal. Die übrigen sechs Merkmale verhalten sich weitgehend wie Rauschen. Diese Abhängigkeit von einem einzigen Merkmal erklärt die meisten der folgenden Ergebnisse.

---

## Die zwei Aufgaben

| Aufgabe | Zielvariable | Typ | Hauptmetriken |
|---|---|---|---|
| Klassifikation | `meat_type` (1–5) | Mehrklassen | Accuracy, F1 pro Klasse |
| Regression | `price_eur_per_kg` | Stetig | MAE, RMSE, R² |

Jedes Experiment wird zweimal ausgewertet: auf einem internen Test-Split (20 % der sauberen Daten) und auf dem Stresstest-Datensatz.

---

## Metriken erklärt

### Klassifikation

| Metrik | Welche Frage sie beantwortet |
|---|---|
| **Accuracy** | Wie oft liegt das Modell insgesamt richtig? |
| **Precision** | Wenn das Modell Klasse X vorhersagt, wie oft stimmt das? |
| **Recall** | Wie viele aller echten Klasse-X-Fälle hat das Modell gefunden? |
| **F1-Score** | Ausgleich zwischen Precision und Recall (harmonisches Mittel) |

Bei 5 Klassen ergibt reines Raten etwa 20 % Accuracy.

### Regression

| Metrik | Perfekter Wert | Bedeutung |
|---|---|---|
| **MAE** | 0 | Durchschnittlicher absoluter Fehler in EUR — „die Vorhersagen liegen X EUR daneben" |
| **RMSE** | 0 | Wie MAE, aber große Fehler werden stärker bestraft (sie werden vorher quadriert) |
| **R²** | 1,0 | Anteil der Preisvarianz, den das Modell erklärt |

Zwei Faustregeln:

- **R² = 0** bedeutet, das Modell ist nicht besser, als immer den Durchschnittspreis vorherzusagen. **Negatives R²** bedeutet, es ist *schlechter* als das.
- **RMSE deutlich größer als MAE** bedeutet, dass das Modell einige sehr große Fehler macht.

---

## Repository-Struktur

Alle Dateien liegen im Hauptverzeichnis des Repositorys. Die Dateien sind über ihr Namenspräfix nach Algorithmus gruppiert.

```
ml_exercises/
│
├── meat_price_dataset.csv                          # 1.200 Zeilen, saubere Daten
├── meat_price_100_stress_test.csv                  # 100 Zeilen, Grenzfälle + fehlende Werte
│
├── decision_tree_classification_meat_type.ipynb    # DT-Klassifikation (meat_type)
├── decision_tree_regression_price_per_kilo.ipynb   # DT-Regression (Preis)
├── decision_tree_summary.md                        # DT-Ergebnisse + Erklärungen
├── decision_tree_tuning_summary.md                 # DT-Grid-Searches + Parameter-Glossar
│
├── random_forest_classification.ipynb              # RF-Baseline + domänenbasierte Imputation
├── random_forest_target_encoding.ipynb             # RF + Target Encoding (bester Klassifikator)
├── random_forest_regression.ipynb                  # RF-Regressions-Baseline
├── random_forest_regression_grid_search.ipynb      # RF-Grid-Search
├── random_forest_regression_outlier_imputation.ipynb  # KNN-Imputation + Ausreißer-Clipping
├── random_forest_combined_dataset.ipynb            # Training auf sauberen + Stress-Zeilen
├── random_forest_log_price.ipynb                   # log(Preis) als Zielvariable
├── rf_classification_summary.md                    # RF-Klassifikation im Detail
├── rf_regression_summary.md                        # RF-Regression im Detail
│
├── xgb_regression.ipynb                            # NB1: XGBoost-Baseline
├── xgb_regression_clip_fill.ipynb                  # NB2: gruppenweise Imputation + Clipping
├── xgb_regression_one_hot_encoding.ipynb           # NB3: One-Hot-kodiertes meat_type
├── xgb_regression_log_one_hot_encoding.ipynb       # NB4: log-Zielvariable + One-Hot
├── xgb_regression_combined.ipynb                   # NB5: Training auf kombiniertem Datensatz
├── xgb_regression_summary.md                       # XGBoost-Regression im Detail
│
├── README.md                                       # Englisch
├── README.de.md                                    # Deutsch
├── LICENSE
└── .gitignore
```

Jeder Algorithmus hat eine **Zusammenfassung**, die die Experimente im Detail erklärt. Am besten dort beginnen, bevor man die Notebooks öffnet. Das vollständige Glossar der Baum-Hyperparameter (`max_depth`, `min_samples_leaf`, `ccp_alpha`, …) steht in `decision_tree_tuning_summary.md`.

---

## Ergebnisse im Überblick

### Klassifikation — Vorhersage von `meat_type`

| Experiment | Accuracy (Test) | Accuracy (Stress) | Notebook |
|---|---|---|---|
| DT, falsche Zielvariable (`cut_quality`) | 22,5 % | — | `decision_tree_classification_meat_type.ipynb` |
| DT-Baseline (`meat_type`) | 53,0 % | 27,0 % | `decision_tree_classification_meat_type.ipynb` |
| DT + Feature Engineering (Verhältnisse) | 53,0 % | 26,0 % | `decision_tree_classification_meat_type.ipynb` |
| RF-Baseline (KNN-Imputation) | 56,0 % | 24,2 % | `random_forest_classification.ipynb` |
| RF + domänenbasierte Imputation | 56,0 % | **27,5 %** | `random_forest_classification.ipynb` |
| **RF + Target Encoding** | **62,5 %** | 25,0 % | `random_forest_target_encoding.ipynb` |

**F1 pro Klasse — die mittleren Klassen sind das Problem:**

| Klasse | DT | RF-Baseline | RF + Target Encoding |
|---|---|---|---|
| 1 (am günstigsten) | 0,65 | 0,74 | 0,74 |
| 2 | 0,39 | 0,48 | **0,55** |
| 3 | 0,49 | 0,39 | **0,51** |
| 4 | 0,24 | 0,42 | **0,54** |
| 5 (am teuersten) | 0,77 | 0,73 | 0,75 |

### Regression — Vorhersage von `price_eur_per_kg`

| Modell | Experiment | MAE Test | R² Test | MAE Stress | R² Stress |
|---|---|---|---|---|---|
| DT | Baseline (Tiefe 7) | 2,38 | 0,8737 | 8,59 | -0,2544 |
| DT | GridSearchCV (beste Parameter) | 2,18 | 0,8875 | 8,70 | -0,3108 |
| RF | Baseline | 1,80 | 0,9249 | 8,68 | -0,3288 |
| RF | Grid Search (`max_features=0.3`) | — | 0,8422 | 8,46 | -0,2711 |
| RF | KNN-Imputation | 1,80 | 0,9249 | 8,64 | -0,2806 |
| RF | KNN-Imputation + Ausreißer-Clipping | 1,80 | 0,9249 | 8,64 | -0,2806 |
| RF | Training auf kombiniertem Datensatz | 2,15 | 0,8841 | 8,70 | -0,3082 |
| RF | log(Preis) als Zielvariable | 1,80 | 0,9252 | 8,80 | -0,3161 |
| **XGB** | **NB1: Baseline** | **1,06** | **0,9751** | 8,76 | -0,3590 |
| XGB | NB2: Gruppenweise Imputation + Clipping | — | — | 8,36 | -0,3078 |
| XGB | NB3: One-Hot `meat_type` | — | — | — | -0,3495 |
| XGB | NB4: log-Zielvariable + One-Hot | — | — | — | -0,3669 |
| **XGB** | **NB5: Training auf kombiniertem Datensatz** | 1,79 | 0,8637 | **7,38** | **+0,1968** |

MAE-Werte in EUR pro kg.

### Bestes Ergebnis pro Modell

| Modell | Bestes R² Test | Bestes R² Stress |
|---|---|---|
| Decision Tree | 0,8875 | -0,2544 |
| Random Forest | 0,9249 | -0,2711 |
| **XGBoost** | **0,9751** | **+0,1968** |

**Hinweis zum NB5-Ergebnis:** Es ist das erste positive Stresstest-R² im Projekt. Gemessen wurde es auf den Stress-Zeilen innerhalb des 20-%-Test-Splits der kombinierten Daten (etwa 18 Zeilen). Der Wert ist also aussagekräftig, aber verrauscht. Er sollte mit Cross-Validation oder wiederholten Splits bestätigt werden, bevor er als endgültig gilt.

---

## Was funktioniert hat

| Technik | Wirkung | Warum es funktioniert hat |
|---|---|---|
| Richtige Wahl der Zielvariable | Klassifikation 22,5 % → 53 % | Die Korrelationsanalyse zeigte, wo das Signal tatsächlich liegt |
| Random Forest statt Decision Tree | R² 0,87 → 0,92, Accuracy 53 % → 56 % | Das Mitteln über 200 Bäume reduziert die Fehler einzelner Bäume |
| Diversität über `max_features` (RF) | Bestes RF-Stress-R² (-0,27) | Zwingt die Bäume, auch aus schwächeren Merkmalen zu lernen |
| Domänenbasierte Imputation | Klassifikation Stress 24 % → 27,5 % | Fehlende Werte werden mit dem Median desselben `meat_type` gefüllt, nicht mit einem globalen Wert |
| Target Encoding | Klassifikation 56 % → 62,5 % | Die kodierten Merkmale erfassen Gruppenmuster, die die Rohwerte nicht abbilden können |
| XGBoost statt Random Forest | R² 0,92 → 0,97 | Boosting baut die Bäume nacheinander; jeder korrigiert die Fehler der vorherigen |
| Gruppenweise Imputation + Clipping (XGB) | Stress-R² -0,36 → -0,31 | Realistische Füllwerte und keine Extrapolation über den Trainingsbereich hinaus |
| Training auf kombiniertem Datensatz (XGB) | Stress-R² -0,36 → **+0,20** | Das Modell sieht die Stressmuster bereits im Training |

---

## Sackgassen

| Technik | Ergebnis | Warum es gescheitert ist |
|---|---|---|
| Feature Engineering mit Verhältnissen | Kein Gewinn, Stress 27 % → 26 % | Preis (Signal) kombiniert mit Fett/Protein (Rauschen) ergibt Rauschen: *Rauschen × Signal = Rauschen* |
| KNN-Imputation (Klassifikation) | Stress 27 % → 24 % | Der Forest nutzt mehr Merkmale, also haben schlechte Füllwerte mehr Gelegenheit, ihn in die Irre zu führen |
| Einfachere, beschnittene Bäume | R² Test 0,87 → 0,74, Stress unverändert | Siehe „Das Pruning-Paradox" unten |
| Ausreißer-Clipping (RF) | Keine Veränderung | Nur 2–6 Werte pro Merkmal liegen außerhalb des sauberen Bereichs, und die dominanten Merkmale (`meat_type`, `cut_quality`) haben gar keine. Clipping der Merkmale kann außerdem das eigentliche Problem nicht lösen: das verschobene Preisniveau |
| Training auf kombiniertem Datensatz (RF) | Schlechter auf beiden Datensätzen | 100 Stress-Zeilen unter 1.300 wurden als Rauschen behandelt (siehe offene Frage unten) |
| log(Preis) als Zielvariable | Keine Veränderung oder schlechter | Der Preis ist annähernd symmetrisch verteilt, log-Skalierung bringt daher nichts |
| One-Hot-Encoding von `meat_type` (XGB) | Keine Veränderung | Bäume können ordinale Kategorien bereits gut aufteilen; One-Hot zersplittert nur das stärkste Merkmal |
| Hyperparameter-Tuning | Nur minimale Gewinne | 300 + 4.320 DT-Kombinationen und 360 + 375 RF-Kombinationen ergaben nie ein positives Stress-R² |

**Offene Frage — warum ist kombiniertes Training beim RF gescheitert, bei XGBoost aber nicht?**
Ein wahrscheinlicher Unterschied: Das RF-Notebook hat fehlende Preise in den Stress-Zeilen mit dem Median gefüllt, während das XGBoost-Notebook Zeilen ohne Zielwert entfernt hat. Imputierte Zielwerte sind erfundene Labels und könnten dem RF falsche Muster beigebracht haben. Das ist eine Hypothese und noch nicht getestet.

---

## Warum der Stresstest so schwer ist

**1. Die Preise sind verschoben — der Hauptgrund.** Für jede Fleischkategorie liegen die Stresstest-Preise im Durchschnitt höher als in den sauberen Daten:

| `meat_type` | Ø Preis (sauber) | Ø Preis (Stress) | Differenz |
|---|---|---|---|
| 1 | 14,03 | 22,10 | +8,07 |
| 2 | 16,44 | 26,62 | +10,18 |
| 3 | 24,20 | 32,20 | +8,00 |
| 4 | 27,24 | 33,28 | +6,04 |
| 5 | 32,48 | 33,57 | +1,09 |
| **Gesamt** | **23,05** | **29,97** | **+6,92** |

Ein Modell, das nur auf sauberen Daten trainiert wurde, lernt das saubere Preisniveau und sagt die Stresspreise daher im Schnitt rund 7 EUR zu niedrig voraus — egal wie es getunt ist. Das erklärt, warum keine DT- oder RF-Konfiguration ein positives Stress-R² erreicht hat, und warum kombiniertes Training (XGB NB5), bei dem das Modell die höheren Preise schon im Training sieht, als einziger Ansatz erfolgreich war.

Die Verschiebung schadet auch der Klassifikation: In den Stressdaten liegen die Typen 3, 4 und 5 alle bei etwa 33 EUR, der Preis kann sie also nicht mehr trennen.

**2. Starre Grenzen.** Ein Entscheidungsbaum trifft an jedem Split eine harte Entscheidung:

```
If meat_type <= 2.5  →  go left  →  predict 16.40 EUR
If meat_type >  2.5  →  go right →  predict 28.70 EUR
```

Eine Stresstest-Zeile mit einer ungewöhnlichen Kombination überschreitet die falsche Grenze und landet in einem Blatt, das für ganz andere Fälle gedacht ist. Ein einzelner Baum hat keine zweite Meinung und keine Möglichkeit, den Fehler zu korrigieren.

**3. Das Pruning-Paradox.** Eigentlich sollten einfachere Bäume besser generalisieren. Ein Baum mit Tiefe 3 (6 Blätter) erreichte aber ein Stress-R² von -0,35, ein Baum mit Tiefe 7 (99 Blätter) dagegen -0,31. Die Vereinfachung kostete Genauigkeit auf den sauberen Daten und brachte beim Stresstest nichts. Das Problem ist nicht die Komplexität des Modells.

**4. Die Randklassen sind leicht, die mittleren überlappen.** Die Klassen 1 und 5 liegen in klar getrennten Preisbereichen und erreichen einen F1 von etwa 0,75. Die Klassen 2, 3 und 4 überlappen sich im Preis und lassen sich allein über den Preis nicht trennen. Deshalb hat Target Encoding, das neue Trennmöglichkeiten für diese Gruppen schafft, den mittleren Klassen am meisten geholfen (Klasse 4: F1 0,24 → 0,54).

**5. Ein Merkmal dominiert.** Die Feature Importance zeigt, wie stark die Modelle von einem einzigen Merkmal abhängen:

| Modell | Wichtigstes Merkmal | Importance |
|---|---|---|
| Decision Tree (Regression) | `meat_type` | 70,3 % |
| Random Forest (Regression) | `meat_type` | 69,5 % |
| Random Forest (Klassifikation) | `price_eur_per_kg` | 52 % |
| RF + Target Encoding | `price_eur_per_kg` / `price_bin_te` | 33,7 % / 21,6 % |

Wenn dieses dominante Merkmal in einer Stresstest-Zeile fehlt, imputiert wurde oder ungewöhnlich ist, bricht die Vorhersage ein.

**6. XGBoost passt sich eng an.** Die XGBoost-Baseline erreichte einen Train-RMSE von 0,39, aber einen Test-RMSE von 1,30, was auf etwas Overfitting hindeutet. Alle fünf XGBoost-Notebooks nutzten dieselbe, nicht getunte Konfiguration (`n_estimators=150`, `learning_rate=0.08`, `max_depth=5`). Das Tuning steht also noch aus.

---

## Wichtigste Erkenntnisse

1. **Erst die Daten verstehen, dann modellieren.** Eine Korrelationsprüfung hätte schon vor dem ersten Training gezeigt, dass `cut_quality` als Zielvariable nicht lernbar ist.
2. **Bessere Modelle helfen auf sauberen Daten.** Decision Tree → Random Forest → XGBoost verbesserte R² von 0,87 auf 0,97.
3. **Bessere Modelle lösen keine Out-of-Distribution-Daten.** Tausende Hyperparameter-Kombinationen ergaben mit DT und RF nie ein positives Stress-R². Die Stresspreise liegen im Schnitt rund 7 EUR höher — ein Datenproblem, kein Tuning-Problem.
4. **Zuerst die Verteilungen von Trainings- und Testdaten vergleichen.** Ein einfacher Gruppenvergleich der beiden Datensätze zeigt die Preisverschiebung sofort und erklärt die meisten Stresstest-Ergebnisse.
5. **Fachwissen schlägt generische Imputation.** Das Füllen fehlender Werte pro `meat_type`-Gruppe brachte in beiden Aufgaben die besten Stresstest-Ergebnisse.
6. **Target Encoding war die stärkste Feature-Technik** für die Klassifikation.
7. **Beliebte Techniken sind nicht universell.** Log-Transformationen helfen bei schiefen Zielvariablen und One-Hot-Encoding bei linearen Modellen. Beides traf hier nicht zu.
8. **Dem Modell die schwierigen Fälle zu zeigen, hat funktioniert.** Kombiniertes Training mit XGBoost war der einzige Ansatz mit positivem Stress-R².
9. **Beides gleichzeitig zu maximieren, gelingt meist nicht.** Die beste Konfiguration auf sauberen Daten ist selten die beste beim Stresstest.

---

## Erste Schritte

### Voraussetzungen

- Python 3.10 oder neuer
- Jupyter (Notebook, JupyterLab oder VS Code)

### Installation

```bash
git clone https://github.com/Martin-Frei/ml_exercises.git
cd ml_exercises

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### Empfohlene Lesereihenfolge

1. `decision_tree_summary.md` → `decision_tree_tuning_summary.md`
2. `rf_classification_summary.md` → `rf_regression_summary.md`
3. `xgb_regression_summary.md`

Danach die Notebooks in der Reihenfolge öffnen, die in der jeweiligen Zusammenfassung angegeben ist. Jedes Notebook enthält Schritt-für-Schritt-Erklärungen in Markdown-Zellen und lässt sich eigenständig von oben nach unten ausführen.

---

## Ein neues Experiment hinzufügen

Dieses README ist darauf ausgelegt, mit dem Projekt zu wachsen. Für jedes neue Experiment:

1. **Ein neues Notebook** für den neuen Prozess oder Schritt anlegen. Benennung: `<algorithm>_<task>_<technique>.ipynb`, z. B. `xgb_regression_hyperparameter_tuning.ipynb`.
2. Jeden Schritt in Markdown-Zellen erklären.
3. Relative Pfade zu den CSV-Dateien verwenden, niemals rechnerspezifische absolute Pfade.
4. Nach Abschluss eines Experimentblocks `<algorithm>_<task>_summary.md` schreiben.
5. Beide READMEs (Englisch und Deutsch) aktualisieren: Dateien unter [Repository-Struktur](#repository-struktur) ergänzen, Zeilen unter [Ergebnisse im Überblick](#ergebnisse-im-überblick) hinzufügen und [Was funktioniert hat](#was-funktioniert-hat) oder [Sackgassen](#sackgassen) erweitern.

Für einen neuen Algorithmus (z. B. LightGBM, KNN, neuronale Netze) einen neuen Block im Strukturbaum und eine neue Zeile unter „Bestes Ergebnis pro Modell" anlegen.

---

## Konventionen

1. Code, Notebooks, Kommentare, Zusammenfassungen und Dateinamen sind auf **Englisch**. Dieses deutsche README dient der besseren Lesbarkeit.
2. Jeder neue Prozess oder Schritt bekommt **ein eigenes Notebook**. Prozesse werden nie in einem Notebook kombiniert.
3. Jedes Notebook enthält Schritt-für-Schritt-Erklärungen in Markdown-Zellen.

---

## Roadmap

- [x] Decision Tree — Klassifikation und Regression
- [x] Random Forest — Klassifikation und Regression
- [x] XGBoost — Regression (5 Notebooks)
- [ ] XGBoost — Klassifikation
- [ ] XGBoost — Hyperparameter-Tuning auf dem kombinierten Datensatz (mit leakage-freier Stresstest-Auswertung)
- [ ] Robuste Auswertung des Stresstest-Ergebnisses (Cross-Validation, wiederholte Splits)
- [ ] Augmentierung der Stressdaten für die Regression (Noise Injection / Gaußsche Störung — SMOTE ist nur für Klassifikation geeignet)
- [ ] Hypothese RF vs. XGBoost beim kombinierten Training testen
- [ ] `requirements.txt`

---

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz — siehe [LICENSE](LICENSE).

**Autor:** Martin Freimuth — [GitHub](https://github.com/Martin-Frei)