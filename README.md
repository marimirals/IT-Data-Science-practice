# Data Science Practice

Lab assignments for a university data analysis course: Python, NumPy, pandas, matplotlib, k-nearest neighbors and decision trees with scikit-learn.

## Lab 1 — Python, NumPy and pandas (`ИТ_лр1.ipynb`)

- Python basics: Fibonacci sequence, a tokenizer and calculator for arithmetic expressions, string and file name handling with exceptions.
- NumPy broadcasting, loop vs vectorized performance, date parsing in pandas (mixed formats, `dayfirst`).
- **Exploratory analysis of the Gapminder dataset** (142 countries, 1952–2007): missing values and duplicates, descriptive statistics, top and bottom countries by life expectancy, GDP growth by continent, correlation of life expectancy with GDP and log GDP (0.68 vs 0.81), population shares, charts and conclusions.

## Lab 2 — Visualization and k-NN (`ИТ_лр2.ipynb`)

- **CO₂ emissions (Our World in Data):** separating countries from regional aggregates, top emitters in 2024, emission trends of the five largest emitters since 1990. Results are checked against reference values.
- **k-nearest neighbors from scratch** on the Palmer Penguins dataset: predicting sex from body measurements.
  - The distance matrix is computed with NumPy broadcasting, without loops over objects.
  - The implementation matches scikit-learn on 100% of predictions.
  - k is tuned from 1 to 25 on a validation set; best k = 14.
  - Test accuracy is 0.910, against 0.507 for a constant baseline.
  - The notebook also explains why species features add almost nothing: species is fully determined by the four body measurements.
- **Instagram top accounts (Kaggle):** cleaning and type conversion, follower growth by account type and category, growth vs account size, overlaid histograms.

## Lab 3 — Training and evaluating a model (`lab3_Лазуткина.ipynb`)

**Task:** predict a car's region of origin (USA, Europe, Japan) from 6 technical characteristics. Dataset: Auto MPG, 392 cars from 1970–1982. Model: decision tree.

- **Pipeline:** `ColumnTransformer` + `SimpleImputer` + `DecisionTreeClassifier` in a scikit-learn `Pipeline`, so preprocessing is fitted only on training folds (no data leakage).
- **Evaluation setup:** stratified 80/20 train/test split, `GridSearchCV` with 5-fold `StratifiedKFold` over `max_depth` and `min_samples_leaf`, a constant baseline for reference.
- **Results:**

  | Metric | Value |
  |---|---|
  | Accuracy, 5-fold CV | 0.796 ± 0.043 |
  | Accuracy, test | 0.873 |
  | Constant baseline ("always USA"), test | 0.620 |

- **Interpretation:**
  - Engine displacement is the most important feature (importance 0.49), followed by weight and horsepower.
  - The confusion matrix shows that US cars are recognized almost perfectly. European cars are the hardest class: they are most often confused with Japanese cars, because both have small engines.
- **Deployment step:** the whole trained pipeline is saved with `joblib`, loaded back, checked to give identical test predictions, and used to predict a single car in original units.

## Tech stack

Python 3, NumPy, pandas, matplotlib, scikit-learn, joblib, kagglehub, Jupyter.

## How to run

Open the notebooks in Jupyter or Google Colab. Datasets are loaded from public URLs and Kaggle when the notebooks run.

```bash
pip install numpy pandas matplotlib scikit-learn joblib kagglehub jupyter
jupyter notebook
```
