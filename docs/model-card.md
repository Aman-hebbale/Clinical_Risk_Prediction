# Model Card — MedPredict Heart Disease Classifier

## Model overview

MedPredict trains four binary classifiers to predict the presence of heart disease
from 13 clinical features. All models share the same data split and preprocessing
pipeline so their metrics are directly comparable.

| Attribute | Value |
|-----------|-------|
| Task | Binary classification (disease present / absent) |
| Input | 13 clinical features (see Dataset section) |
| Output | Binary label (0 / 1) + probability score |
| Framework | scikit-learn 1.4+ / PyTorch 2.x |
| Tracking | MLflow experiment `heart-disease-prediction` |

## Models trained

| Name in code | Algorithm | Key hyperparameters |
|---|---|---|
| `LogisticModel` | Logistic Regression (lbfgs) | C=1.0, max_iter=1000 |
| `RandomForestModel` | Random Forest | n_estimators=200, max_depth=None |
| `SklearnModel` | Gradient Boosting | n_estimators=200, lr=0.05, max_depth=3 |
| `TorchModel` | Feed-forward MLP | hidden=[64,32], dropout=0.3, lr=1e-3, epochs=100 |

The model served by the API (`MODEL_PATH`) defaults to the Gradient Boosting pickle.
To serve a different model, update `MODEL_PATH` and restart the container.

## Training data

- **Dataset**: UCI Heart Disease, Cleveland subset
- **Source**: https://archive.ics.uci.edu/dataset/45/heart+disease
- **Size**: 303 patients, 13 features
- **Target distribution (binary)**: approximately 54 % disease, 46 % no disease
- **Train / test split**: 80 / 20 stratified by label, random seed 42
- **Preprocessing**:
  - Numeric features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`): median imputation, then standard scaling
  - Categorical features (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`): most-frequent imputation, then one-hot encoding
  - Engineered features added before the sklearn pipeline: `age_group` (decade bucket, ordinal) and `chol_age_ratio` (chol / age)

## Performance metrics

The table below reflects metrics logged to MLflow on the held-out 20 % test set
(61 patients). These figures are representative; exact values depend on the
`ucimlrepo` download at the time of training.

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Gradient Boosting | ~0.85 | ~0.85 | ~0.92 |
| Random Forest | ~0.84 | ~0.83 | ~0.91 |
| Logistic Regression | ~0.82 | ~0.81 | ~0.89 |
| PyTorch MLP | ~0.82 | ~0.81 | ~0.88 |

To see exact per-run metrics, open the MLflow UI (`http://localhost:5000`) and
navigate to the `heart-disease-prediction` experiment.

## Known limitations

- **Small dataset**: 303 rows from a single centre. Performance estimates carry
  high variance; confidence intervals are wide.
- **Cleveland subset only**: The UCI repository contains data from four centres
  (Cleveland, Hungary, Switzerland, Long Beach VA). Only Cleveland data has
  complete labels; models trained here may not generalise to other populations.
- **Class near-balance hides minority-class errors**: The binary split is
  approximately 54 / 46. Accuracy is not a misleading metric here, but F1
  and ROC-AUC remain the primary evaluation criteria.
- **Engineered features are uncalibrated**: `chol_age_ratio` provides a signal
  in-distribution but has not been validated against clinical guidelines.
- **No calibration step**: Predicted probabilities are not calibrated via Platt
  scaling or isotonic regression. Treat probability values as indicative rank
  scores rather than precise probability estimates.
- **Static model**: There is no automated retraining pipeline. Distributional
  shift in new patient populations will degrade performance undetected unless
  monitoring is added.

## Ethical considerations

This model is a research and engineering demonstration built on publicly
available data. It must not be used to support actual clinical decisions,
diagnoses, or treatment plans.

- **No clinical validation**: The model has not been validated against
  prospective patient cohorts or reviewed by medical professionals.
- **Population bias**: Training data is from the Cleveland Clinic, 1988. Results
  may not reflect modern patient populations or diverse demographics.
- **Accountability**: Automated predictions for medical use require regulatory
  oversight (e.g., FDA 510(k) clearance in the US, CE marking in the EU). This
  software has none.
- **Privacy**: The API accepts clinical feature values. In any deployment
  involving real patients, all inputs must be treated as protected health
  information (PHI) under applicable law (HIPAA, GDPR, etc.).

## Citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988).
Heart Disease [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X
