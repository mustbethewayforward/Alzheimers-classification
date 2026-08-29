# Alzheimer’s Classification Rebuild

## Overview

This project is a rebuild of an earlier machine learning classification project using the OASIS-2 longitudinal dataset. The main aim was not just to improve model performance, but to build a cleaner workflow and understand why each modelling decision matters.

The task is binary classification of baseline subjects as **Demented** or **Nondemented** using demographic, cognitive and MRI-derived features. To reduce subject-level leakage from repeated visits, I restricted the main analysis to each subject's first visit. I also moved preprocessing into scikit-learn pipelines, compared several models using stratified cross-validation, added a majority-class dummy baseline, tuned the strongest models and evaluated the selected model on a held-out test set.

The final modelling cohort contains **136 subjects**, and Random Forest was selected using cross-validated F1. On the held-out test set it achieved **0.79 accuracy, 0.73 F1 and 0.785 ROC-AUC**. These results are useful as a small-sample machine learning exercise, but they should not be interpreted as evidence of clinical diagnostic performance.

## Dataset and Cohort

OASIS-2 contains **373 observations from 150 unique subjects**, with some subjects measured at multiple visits. A row-level random split would therefore risk placing different visits from the same person in both training and evaluation data.

I restricted the primary analysis to **Visit 1**, giving one baseline observation per subject:

| Group | Baseline subjects |
| --- | ---: |
| Nondemented | 72 |
| Demented | 64 |
| Converted | 14 |
| **Total** | **150** |

The 14 `Converted` subjects were excluded from the primary binary classification task. At baseline, **13 had CDR = 0.0 and one had CDR = 0.5**. Rather than overriding the original `Group` labels and assigning the whole group to a new class, I kept the target definition simple and limited the main cohort to subjects already labelled Demented or Nondemented.

This leaves **136 subjects: 72 Nondemented and 64 Demented**. Nondemented is encoded as `0` and Demented as `1`.

The raw OASIS-2 dataset is not included in this repository. It should be obtained separately and stored locally as `data/oasis_longitudinal.csv`. See `data/README.md` for the dataset setup instructions.

## Features and Target

The target is the OASIS `Group` label after restricting the cohort to baseline Demented and Nondemented subjects. I kept `Group` as the target rather than creating a new diagnosis directly from CDR. During inspection of the longitudinal data, some subjects who remained labelled Nondemented showed changes in CDR across visits, so I did not treat a single CDR value as interchangeable with the dataset's longitudinal group label.

All predictor values below are taken strictly from **Visit 1**.

| Feature | Description | Measurement used |
| --- | --- | --- |
| `M/F` | Sex | Visit 1 |
| `Age` | Age | Visit 1 |
| `EDUC` | Years of education | Visit 1 |
| `SES` | Socioeconomic status | Visit 1 |
| `MMSE` | Mini-Mental State Examination score | Visit 1 |
| `eTIV` | Estimated total intracranial volume | Visit 1 |
| `nWBV` | Normalized whole-brain volume | Visit 1 |
| `ASF` | Atlas scaling factor | Visit 1 |

`Subject ID` and `MRI ID` were excluded because they are identifiers. CDR was excluded as a predictor because of its close relationship with dementia status and the resulting leakage risk. `Visit`, `MR Delay` and `Hand` were also excluded because they are constant in the final baseline cohort.

Dataset inspection showed a very strong negative correlation between `eTIV` and `ASF` of approximately **r = -0.99**. I retained both in the current model, but their individual contributions should therefore be interpreted cautiously.

## Methodology

### Train-test split

I used a stratified **80/20 train-test split** with `random_state=42`. This produced:

- **108 training subjects**
- **28 test subjects**
- test-set class counts of **15 Nondemented and 13 Demented**

The test set was not used in the final cross-validation or hyperparameter-selection procedure.

### Preprocessing

Preprocessing was implemented using scikit-learn `Pipeline` and `ColumnTransformer` objects so that transformations are fitted separately inside each training fold during cross-validation.

`SES` was the only predictor containing missing values. There were **8 missing SES values**, all in the Demented group:

| Group | Subjects | Missing SES | Missing |
| --- | ---: | ---: | ---: |
| Demented | 64 | 8 | 12.5% |
| Nondemented | 72 | 0 | 0.0% |

Numeric missing values are handled using median imputation inside the pipeline.

Sex (`M/F`) is treated as categorical and encoded using:

```python
OneHotEncoder(drop="if_binary", handle_unknown="ignore")
```

Because `M/F` is binary, `drop="if_binary"` produces a single encoded indicator column rather than two redundant dummy columns.

For Logistic Regression, numeric features are also standardized using `StandardScaler`. Random Forest and Gradient Boosting use median imputation without numeric scaling.

### Cross-validation and model selection

Model comparison uses **5-fold stratified cross-validation** on the 108-subject training set, with shuffled folds and `random_state=42`.

The recorded metrics are accuracy, precision, recall, F1 and ROC-AUC. **F1 for the Demented class was chosen as the main model-selection metric** because it balances precision and recall.

A majority-class `DummyClassifier` was included to provide a simple reference baseline.

## Results

### Cross-validation comparison

Mean ± standard deviation across the five folds:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy, most frequent | 0.528 ± 0.017 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Logistic Regression | 0.852 ± 0.045 | **0.933 ± 0.055** | 0.745 ± 0.100 | 0.823 ± 0.060 | **0.919 ± 0.049** |
| Random Forest | **0.870 ± 0.037** | 0.922 ± 0.074 | **0.802 ± 0.092** | **0.851 ± 0.044** | 0.900 ± 0.040 |
| Gradient Boosting | 0.842 ± 0.039 | 0.899 ± 0.086 | 0.764 ± 0.104 | 0.817 ± 0.054 | 0.877 ± 0.052 |

All three trained classifiers substantially outperformed the majority-class dummy baseline. Random Forest had the highest mean accuracy, recall and F1, while Logistic Regression had the highest precision and ROC-AUC.

### Hyperparameter tuning

Logistic Regression and Random Forest were tuned using `GridSearchCV` with the same 5-fold stratified cross-validation and F1 as the optimization metric.

| Tuned model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression, `C=10` | 0.861 ± 0.067 | 0.892 ± 0.071 | **0.804 ± 0.110** | 0.843 ± 0.077 | **0.920 ± 0.051** |
| Random Forest | **0.870 ± 0.037** | **0.922 ± 0.074** | 0.802 ± 0.092 | **0.851 ± 0.044** | 0.900 ± 0.040 |

Random Forest tuning selected **100 estimators, unrestricted tree depth and a minimum leaf size of 1**, which are the scikit-learn defaults used by the untuned model. Its mean F1 therefore remained unchanged at 0.851.

The tuned Logistic Regression improved from 0.823 to 0.843 mean F1. The gap between the two final candidates was small, but Random Forest was selected because F1 had already been chosen as the primary selection metric.

### Held-out test evaluation

The selected Random Forest was fitted to the full 108-subject training set and then evaluated on the 28-subject held-out test set.

| Metric | Test result |
| --- | ---: |
| Accuracy | **0.786** |
| Precision, Demented | **0.889** |
| Recall, Demented | **0.615** |
| F1, Demented | **0.727** |
| ROC-AUC | **0.785** |

Confusion matrix:

```text
                 Predicted
                 Non-dem.   Demented
Actual Non-dem.    14          1
Actual Demented     5          8
```

The model correctly classified 14 of 15 Nondemented subjects and 8 of 13 Demented subjects. Its precision was high, but recall was noticeably lower because five Demented subjects were missed.

Held-out performance was also below the Random Forest cross-validation estimates, particularly for F1 and ROC-AUC. With only 28 test subjects, individual predictions have a large effect on the reported metrics, so the test results should be treated as an approximate estimate rather than a precise measure of generalisation.

### Permutation importance

After final evaluation, I calculated permutation importance on the held-out test set using F1 as the scoring metric and 20 repeats.

| Feature | Mean importance | SD |
| --- | ---: | ---: |
| MMSE | **0.244** | 0.096 |
| SES | 0.045 | 0.017 |
| EDUC | 0.034 | 0.027 |
| eTIV | 0.033 | 0.035 |
| M/F | 0.031 | 0.021 |
| ASF | 0.017 | 0.029 |
| Age | 0.004 | 0.031 |
| nWBV | -0.002 | 0.039 |

MMSE was by far the most influential feature in this fitted model. Since MMSE is already a cognitive assessment, the current performance should not be interpreted as showing that the MRI-derived variables alone provide strong discrimination.

The importance values are descriptive rather than causal. They are also estimated from only 28 test subjects, and the strong correlation between eTIV and ASF makes their individual importance values harder to separate.

## Limitations

- **Small sample and no external validation.** The final cohort contains only 136 subjects and the test set contains 28. The reported metrics are therefore sensitive to a small number of predictions and do not establish clinical diagnostic performance.
- **Converted subjects were excluded.** This removes 14 of 150 baseline subjects, or about 9.3% of the baseline sample. Other defensible target definitions are possible and may produce different results.
- **MMSE dominates the current model.** Because it is already a cognitive assessment, the results do not show how well the demographic and MRI-derived variables would perform without it.
- **eTIV and ASF are strongly correlated.** Their correlation is approximately `r = -0.99`, so their separate contributions should not be interpreted as independent effects.
- **SES missingness is class-dependent.** All 8 missing SES values occur in the Demented group. Median imputation handles the missing values computationally but does not explain why this pattern exists or whether missingness itself contains information.
- **The test set was exposed once during early development.** Before the final cross-validation workflow was established, an earlier Logistic Regression model was evaluated on this same held-out split. I did not subsequently tune features, hyperparameters or model choice directly in response to that result, but an indirect influence on later decisions cannot be completely ruled out. The test set should therefore not be described as having remained completely unseen throughout the entire project.
- **Permutation importance reuses the test set.** It was calculated only after model selection and did not feed back into training or tuning, but the importance estimates are exploratory and potentially unstable because they come from the same small 28-subject test sample.

## Future Work

The most useful next steps are targeted robustness checks rather than simply adding more algorithms.

First, I would repeat the full workflow **without MMSE** to measure how much discriminatory performance remains when the dominant cognitive feature is removed.

I would also rerun the analysis **without ASF**, while retaining eTIV, to test whether removing one of the highly correlated MRI-derived variables changes model performance or interpretation.

The SES missingness pattern could be investigated by comparing the current median-imputation approach with a model that includes an explicit missingness indicator.

A separate future project could treat `Converted` subjects as a prognostic problem, using baseline information to predict future conversion rather than trying to force them into one of the current baseline classes.

Finally, external validation on a larger independent dataset would be more valuable than continued tuning on the same small OASIS-2 cohort.

## Running the Project

The raw OASIS-2 dataset is not distributed with this repository. After obtaining it separately, save it as:

```text
data/oasis_longitudinal.csv
```

Run the commands below from the **repository root**, since the scripts use a data path relative to the current working directory:

```bash
pip install -r requirements.txt
python src/data_inspection.py
python src/train_model.py
```

`data_inspection.py` reproduces the main dataset checks used in the write-up, including cohort composition, SES missingness by group, baseline CDR values among Converted subjects and the eTIV/ASF correlation.

`train_model.py` performs the baseline cohort filtering, train-test split, preprocessing, dummy-baseline comparison, cross-validation, model comparison, hyperparameter tuning, final held-out evaluation and permutation importance analysis.

## Repository Structure

```text
.
├── data/
│   └── README.md
├── src/
│   ├── data_inspection.py
│   └── train_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

The raw CSV is stored locally under `data/` and excluded from version control.

## Conclusion

This rebuild produced a cleaner and more reproducible machine learning workflow for baseline dementia classification using OASIS-2. Random Forest was selected using cross-validated F1 and achieved **0.79 accuracy, 0.73 F1 and 0.785 ROC-AUC** on the held-out test set.

The most useful part of the rebuild was not finding a more complicated model. It was learning how choices around cohort construction, target definition, preprocessing and evaluation can change what a machine learning result actually means. The final model still has important limitations, particularly the small dataset and its reliance on MMSE, so I treat it as a learning project and a baseline for further experiments rather than a clinically useful classifier.
