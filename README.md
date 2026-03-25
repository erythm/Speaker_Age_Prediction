# 🎙️ Speaker Age Prediction

Predicting a speaker's age from acoustic speech features using machine learning regression models.

> **Course Project** — Politecnico di Torino 

---

##  Problem Overview

Estimating a speaker's age from voice is a challenging acoustic task with applications in personalized services and health monitoring. This project builds a regression pipeline that predicts speaker age from:

- **Acoustic features**: pitch, jitter, shimmer, energy, zero-crossing rate, MFCCs, chroma, spectral contrast
- **Metadata**: gender, ethnicity, tempo
- **Linguistic features**: number of words, number of pauses, silence duration

---

## 📁 Project Structure

```
.
├── development.csv             # Labeled training data (2,933 samples)
├── evaluation.csv              # Unlabeled test data (691 samples)
├── code.ipynb                  # Main Jupyter notebook (full pipeline)
├── gradient_boosting_submission.csv   # Final predictions output
└── README.md
```

---

##  Requirements

```bash
pip install pandas numpy scikit-learn librosa matplotlib seaborn
```

**Key libraries:**

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `librosa` | Audio loading and feature extraction |
| `scikit-learn` | Model training, evaluation, hyperparameter tuning |
| `matplotlib` / `seaborn` | Visualization |

---

##  Pipeline

### 1. Data Loading & EDA
- Load `development.csv` and `evaluation.csv`
- Inspect missing values and descriptive statistics
- Visualize age distribution

### 2. Preprocessing
- **One-hot encoding** for `gender` and `ethnicity`
- **Tempo parsing**: strip brackets and convert to float
- **Standard scaling** of all numerical features
- **Log transformation** of target variable (`age`) using `log1p` to reduce skewness
- **Feature alignment**: reindex evaluation set to match training columns

### 3. Audio Feature Extraction (via `librosa`)
For each `.wav` file, the following are extracted and averaged across time frames:
- **13 MFCCs** (Mel-Frequency Cepstral Coefficients)
- **12 Chroma features**
- **7 Spectral Contrast bands**

All extracted features are standardized before modeling.

### 4. Model Selection
Three models were compared on the log-transformed target:

| Model | Validation RMSE (log scale) |
|---|---|
| Ridge Regression | 0.2714 |
| Random Forest Regressor | 0.2710 |
| **Gradient Boosting Regressor** | **0.2561** ✅ |

Gradient Boosting was selected as the final model.

### 5. Hyperparameter Tuning
Grid search with 5-fold cross-validation over:

```python
param_grid = {
    "n_estimators":     [100, 200, 300],
    "learning_rate":    [0.01, 0.05, 0.1],
    "max_depth":        [3, 5, 7],
    "subsample":        [0.7, 0.8, 1.0]
}
```

**Best configuration:**

```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

### 6. Prediction & Submission
- Predictions are made on the evaluation set
- Inverse log transform applied: `np.expm1(predictions)`
- Output saved to `gradient_boosting_submission.csv`

---

##  Results

| Metric | Value |
|---|---|
| Validation RMSE (log scale) | **0.2561** |
| Validation RMSE (original scale) | **9.07 years** |
| Evaluation RMSE (external platform) | **9.641 years** |

---

## 📄 Reference

T. Ganchev et al., *Comparative evaluation of various MFCC implementations on the speaker verification task*, vol. 1. MIT Press Cambridge, 2016.
