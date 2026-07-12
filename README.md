# Breast Cancer Classification

A small ML project built on the classic Breast Cancer Wisconsin dataset. The goal is to classify a tumor as **benign** or **malignant** based on a set of features computed from digitized images of a breast mass (things like radius, texture, perimeter, smoothness, concavity, etc).

I wanted to keep this pretty end-to-end — from raw CSV to a working prediction function — while still being deliberate about the metric that actually matters for this kind of problem (more on that below).

## Dataset

- **Source:** Breast Cancer Wisconsin dataset (`breast_cancer_dataset.csv`)
- **Shape:** 569 rows, 32 columns
- **Target:** `diagnosis` — originally `B` (benign) / `M` (malignant), mapped to `0` / `1`
- **Class balance:** ~63% benign, ~37% malignant — a bit imbalanced, but not extreme
- All other 30 columns are numeric features (mean, standard error, and "worst" values for radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension)

No missing values, no duplicate rows, no constant or quasi-constant columns — the data was already pretty clean.

## Project structure

The notebook (`brest_cancer_classification.ipynb`) is organized into the following sections:

1. **Setup** — imports and config
2. **Data collection** — loading the CSV
3. **EDA** — checking for missing values, duplicates, class distribution, feature distributions (histograms), outliers (box plots), and a correlation heatmap
4. **Data preprocessing** — dropping the `id` column, train/test split (80/20, stratified on the target)
5. **Model training** — a baseline Logistic Regression model, scaled with `StandardScaler`
6. **Optimization** — a `Pipeline` (scaler → PCA → SVM) tuned with `GridSearchCV`
7. **Evaluation** — accuracy, confusion matrix, and a full classification report
8. **Inference** — a simple `predict_cancer()` function to run the tuned model on a single sample

## Key EDA takeaways

- All features are numerical, so no encoding needed
- Fairly high-dimensional feature space with a lot of correlated features (e.g. radius/perimeter/area move together, as you'd expect geometrically) — a good candidate for PCA or feature selection
- There's a clear separation in feature means between benign and malignant tumors, which is a good sign the model should be able to pick up a strong signal
- Some extreme values show up, but they tend to correspond to malignant cases rather than being noise, so I didn't strip them out
- Given the slight class imbalance and the cost of missing a malignant case, **recall** was chosen as the metric to optimize for, rather than plain accuracy

## Modeling approach

**Baseline:** Logistic Regression on standardized features.
- Train accuracy: **98.7%**
- Test accuracy: **96.5%**

Already a strong baseline, but I wanted to push further and make sure recall on the malignant class specifically was solid.

**Tuned model:** A `Pipeline` combining `StandardScaler → PCA → SVC (RBF kernel)`, tuned with `GridSearchCV` (5-fold stratified cross-validation, scoring on recall).

Grid searched over:
- PCA components: `[10, 15, 20, 25, 30]`
- SVM `C`: `[0.1, 1, 10, 100, 1000]`
- SVM `gamma`: `[0.001, 0.01, 0.1, 1, "scale"]`

**Best parameters:** `n_components=10`, `C=10`, `gamma='scale'`
**Best CV recall:** ~96.5%

## Results

Evaluated on the held-out test set (114 samples):

| Metric | Benign (0) | Malignant (1) |
|---|---|---|
| Precision | 0.9595 | 0.9750 |
| Recall | 0.9861 | 0.9286 |
| F1-score | 0.9726 | 0.9512 |

**Overall accuracy: 96.5%**

The confusion matrix and full classification report are in the notebook. Recall on the malignant class (~93%) is the number I'd keep an eye on most — in a real diagnostic setting, missing a malignant case is far more costly than a false alarm, so this is the metric worth improving further before this could be trusted beyond a learning project.

## Inference

There's a small `predict_cancer()` helper at the end of the notebook that takes a single row of feature values and prints out whether the model predicts benign or malignant, using the tuned pipeline.

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Notes / possible next steps

- This is trained and evaluated on a single 80/20 split — a full nested cross-validation would give a more robust estimate of performance
- Recall for malignant cases could likely be pushed higher with class weighting, a different threshold, or ensembling
- The `id` column is dropped since it carries no predictive value
- This project is for learning/portfolio purposes only — it's not a diagnostic tool
