# **Melanoma Survival Time Prediction**

A machine learning pipeline to predict **patient survival time** after melanoma diagnosis, with progressive handling of real-world challenges: missing data, censored observations, and limited labelled samples.

## **Problem Statement**

Multiple Myeloma is a haematological malignancy where predicting patient survival time is clinically critical for treatment planning. This project frames the problem as a **regression task over right-censored survival data** — a realistic and challenging setup where some patients have not yet experienced the event of interest at the time of data collection.

The key challenges were threefold:
- **Right-censored targets**: 19.8% of patients are censored, meaning their true survival time is unknown but bounded below. Standard MSE is inappropriate here.
- **Missing feature values**: Several clinical features (Genetic Risk, Comorbidity Index, Treatment Response) had substantial missingness, and Survival Time itself was missing for 40% of patients.
- **Small labelled dataset**: After removing records with missing targets or censorship, only 109 samples remained — well below what most models need to generalise reliably.

The evaluation metric throughout was **Censored MSE (cMSE)**, which penalises a model only when it predicts a survival time *shorter* than a censored patient's lower bound. This aligns with the clinical intuition that underestimating survival for a censored patient is the true error.

## **Dataset**

| Split | Description |
|-------|-------------|
| `train_data.csv` | Patient records with survival time, censorship status, and clinical features |
| `test_data.csv` | Held-out patients for Kaggle submission |

| Feature | Type | Missing |
|---|---|---|
| Age | Numerical | 0 |
| Gender | Categorical | 0 |
| Stage | Ordinal (1–3) | 0 |
| Treatment Type | Categorical | 0 |
| Genetic Risk | Numerical | 85 (21%) |
| Comorbidity Index | Numerical | 45 (11%) |
| Treatment Response | Numerical | 29 (7%) |
| **Survival Time** (target) | Numerical | **160 (40%)** |
| Censored | Binary | 0 |

The missingness correlation analysis (heatmap + hierarchical dendrogram) showed weak associations (≤ 0.2) between variables, suggesting **Missing At Random (MAR)** rather than systematic dropout. Survival Time was the exception — its 40% missingness, combined with its isolation in the dendrogram, indicates a distinct mechanism likely tied to censoring or loss to follow-up.

**Clinical features used:**

| Feature | Description |
|---------|-------------|
| `Age` | Patient age at diagnosis |
| `Gender` | Patient gender |
| `Stage` | Melanoma stage (I–IV) |
| `GeneticRisk` | Genetic risk score |
| `TreatmentType` | Type of treatment received |
| `ComorbidityIndex` | Index of comorbid conditions |
| `TreatmentResponse` | Response to treatment |
| `SurvivalTime` | Target — months of survival (missing if censored) |
| `Censored` | 1 if outcome is unknown, 0 if observed |

## **Project Structure**

```
├── data/
|   ├── sample_submission.csv
│   ├── train_data.csv
│   └── test_data.csv
|
├── output/
│    ├── baseline-submission.csv
│    ├── Nonlinear-submission.csv
│    ├── handle-missing-submission.csv
│    ├── semisupervised-submission.csv
│    └── optional-submission.csv
|
├── notebooks/
│   └── melnoma_survival.ipynb/
|
├── requirements.txt
└── README.md
```

## **Methodology**

### **Task 1 — Baseline (Linear Regression)**

- **Data pipeline**: visualise missing data (missingno), drop columns with missing features, remove censored rows and missing targets
- **Features**: Age, Gender, Stage, TreatmentType (complete cases only)
- **Model**: Linear Regression with StandardScaler
- **Evaluation**: 5-fold cross-validation, MSE/RMSE on held-out test set
- **Custom metric**: cMSE — a censoring-aware variant of MSE that penalises underestimates on censored samples

A **Linear Regression** baseline was built on complete, non-censored samples using only the four fully-observed features (Age, Gender, Stage, Treatment Type). The validation strategy chosen was **5-Fold Cross-Validation** on an 80% training split, reserving 20% as a hold-out test set.

The choice of 5-Fold CV over a simple train/val/test split was deliberate: with only ~160 labelled samples after cleaning, a static validation split would waste 20% of already scarce data. CV allows every sample to contribute to both training and validation across folds, giving a more reliable estimate of generalisation performance.

The baseline achieved CV MSE ≈ 4.52 and Test MSE ≈ 4.41. The y-ŷ plot revealed a characteristic **collapse-to-the-mean** pattern — predictions clustering between 4 and 6 regardless of the true value — confirming that with so few features and a linear model, the signal was too weak to capture the full survival range.

### **Task 2 — Nonlinear Models**

Two model families were evaluated against the linear baseline:

**Polynomial Ridge Regression**: Iterating over degrees 1–5, without regularisation the CV MSE exploded from 4.53 (degree 1) to 344 (degree 5), a textbook demonstration of the bias-variance trade-off. Adding **Ridge regularisation** stabilised the error across all degrees (all CV MSEs between 4.41–4.59), and Degree 4 was selected for its best trade-off (CV MSE = 4.41, Train MSE = 3.79).

**k-Nearest Neighbours (k-NN)**: Sweeping k from 1 to 28, low values (k=1,2) severely overfit (CV MSE > 6.4), while increasing k smoothed predictions. k=10 and k=28 both achieved CV MSE = 4.41; k=28 was ultimately selected for its slightly lower Test MSE (4.39) and better variance control.

Despite the complexity of Polynomial Ridge, the simpler **k-NN (k=28) won on Test MSE**, highlighting that on small datasets, model expressiveness is less important than variance control.

### **Task 3 — Handling Missing Data**

- **Task 3.1**: Five imputation strategies compared (mean, median, mode, constant zero, KNN-5), now using all 7 features
- **Task 3.2**: Models that handle missing data natively — `HistGradientBoostingRegressor` (sklearn) and `CatBoostRegressor` with AFT survival loss
- **Task 3.3**: Full comparison across all Task 3 models; best model selected for Kaggle submission

Rather than dropping incomplete rows (which would reduce the usable dataset by 73%), imputation strategies were evaluated. Five approaches were compared using a pipeline with StandardScaler + LinearRegression:

| Strategy | CV MSE | Test MSE |
|---|---|---|
| **Mean** | 3.724 | **3.386** |
| k-NN (k=5) | 3.691 | 3.411 |
| Median | 3.741 | 3.500 |
| Most Frequent | 3.741 | 3.500 |
| Constant Zero | 3.765 | 3.542 |

**Mean imputation** won — not because it is theoretically superior, but because it introduces the least distortion for a linear model on this data distribution. The improvement from Task 1 (MSE 4.41 → 3.39) is attributable to the 60% increase in usable training data, not the imputation method per se.

Applying imputation to all three model families:
- **Baseline + Mean**: Test MSE 3.39 — best overall, minimal overfitting gap (0.35)
- **k-NN + Mean**: Test MSE 3.93 — decent, but below linear baseline
- **Polynomial Ridge + Mean**: Test MSE 4.50 — clear overfitting of imputed values

This inversion — where the linear model beats k-NN after imputation — suggests that imputed values introduce artificial smoothness that k-NN's local averaging does not handle well.

Two models capable of **natively handling missing values** were also tested:
- **HistGradientBoostingRegressor**: Train MSE 1.60, Test MSE 4.05 — severe overfitting. Despite its native NaN support, the model memorised the training data, a known issue with boosting on small datasets.
- **CatBoost AFT (Accelerated Failure Time)**: Trained on *all* samples (including censored ones) by encoding survival labels as intervals [lower, ∞) for censored patients. This properly exploits the information in censored observations. Evaluated via cMSE on a validation set; however, it still overfit (Train MSE ≈ 0.027, Test MSE ≈ 3.99).

### **Task 4 — Semi-Supervised Learning**

The semi-supervised approach leveraged unlabelled samples (censored patients and those with missing Survival Time) to fit the imputer and scaler on the **full population distribution**, then froze these transformers before training the regression model strictly on labelled samples. This was implemented via a custom `FrozenTransformer` class to prevent data leakage from re-fitting during cross-validation.

**Semi-supervised Linear Regression** (Test MSE 3.387) was virtually indistinguishable from the fully supervised baseline (3.386). The full dataset distribution did not add meaningful information to the preprocessing steps.

**Semi-supervised Isomap** (n_components=3, Test MSE 3.961) attempted to capture non-linear manifold structure from the unlabelled data before regression. The performance degradation confirms that the geometric structure of this patient dataset does not align with survival time in a way that Isomap can recover.

### **Task 5 — Stacking Ensemble**

A stacking architecture combined four base learners (HistGradientBoosting, CatBoost, XGBoost, Linear Regression) with a **Ridge meta-learner** trained on out-of-fold (OOF) predictions. The 5-fold OOF strategy ensures the meta-learner sees unbiased predictions — each base model's OOF predictions on a fold are generated from models trained on the other four folds.

| Metric | Value |
|---|---|
| Local Test MSE | 0.627 |
| **Kaggle Score** | **3.271** |

The extreme local-Kaggle discrepancy (0.627 vs 3.271) is a symptom of **meta-learner overfitting**: with only ~160 labelled samples, the OOF predictions have high variance, and the Ridge meta-learner fitted to them learns noise. Despite this, the Kaggle score of 3.271 represents a genuine improvement over the single-model baseline (3.38), confirming that ensemble diversity helps even when each individual model struggles.

## **Results Summary**

|| Model | Test MSE (local) | Kaggle MSE |
|---|---|---|
| Baseline Linear Regression | 4.41 | 3.53 |
| k-NN (k=28) | 4.39 | 3.44 |
| Baseline + Mean Imputation | 3.39 | 2.89 |
| Semi-supervised LR | 3.39 | — |
| **Stacking Ensemble** | 0.63* | **2.57** |

*\*Overfitted to local split.*

## **Key Learnings**

**On model complexity and data size**: With fewer than 200 labelled samples, every increase in model expressiveness was punished by overfitting. The linear baseline consistently outperformed k-NN, Polynomial Ridge, and gradient boosting on held-out data — not because it is a better model in general, but because it has fewer degrees of freedom to overfit.

**On censored data**: The cMSE metric is not merely a technical detail — it encodes a clinical truth. Underestimating a censored patient's survival is a real error; overestimating is not. Using standard MSE on censored data would bias model selection. The CatBoost AFT approach, despite not winning here, is the correct methodological choice when censored observations are numerous.

**On imputation**: The choice of imputation strategy should be made jointly with the downstream model. Mean imputation added no artificial structure that could confound a linear model. k-NN imputation performed similarly, while the more complex models that were intended to benefit from richer imputed features ended up overfitting to them instead.

**On semi-supervised preprocessing**: Fitting the scaler and imputer on the full population (labelled + unlabelled) is theoretically sound, but in this case the unlabelled data did not shift the feature distribution meaningfully relative to the labelled subset. The gain was negligible.

## **Setup**

```bash
# Create virtual environment
conda create -n melanoma python=3.11
conda activate melanoma

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**

```
pandas==2.2.2
numpy>=1.26.0,<2.0
scikit-learn==1.4.2
scipy==1.13.1
matplotlib
seaborn
missingno
catboost
xgboost
```

## **Running the Notebook**

```bash
jupyter notebook melanoma_survival.ipynb
```
Run cells sequentially — tasks are dependent on variables defined in previous tasks.

## **Key Design Decisions**

**Why drop censored rows in Tasks 1–2?**
For uncensored patients (Censored == 0), MSE equals cMSE, simplifying baseline comparisons without loss of correctness.

**Why use FrozenTransformer in Task 4?**
scikit-learn pipelines refit all steps during cross-validation. FrozenTransformer prevents the imputer and scaler from being re-fitted on each fold's training subset — preserving the semi-supervised property that these transformers were fitted on the full population.

**Why Ridge as meta-learner?**
Ridge regularisation prevents the meta-model from over-weighting a single base learner, especially when base model predictions are correlated.

## **Technologies**

`Python` · `scikit-learn` · `CatBoost` · `XGBoost` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `missingno`