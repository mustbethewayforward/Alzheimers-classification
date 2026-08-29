import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, make_scorer, precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier



df = pd.read_csv("data/oasis_longitudinal.csv")
baseline_df = df[df["Visit"]==1].copy()

 #Exclude Converted subjects because this primary model is baseline
 #  dementia classification, not future conversion prediction
model_df = baseline_df[
    baseline_df["Group"].isin(["Demented", "Nondemented"])
 ].copy()


#Define features and target
feature_columns = ["M/F","Age","EDUC", 
                   "SES", "MMSE", "eTIV", "nWBV", "ASF"
]

X = model_df[feature_columns]

y = model_df["Group"].map({
    "Nondemented": 0, 
    "Demented": 1})

#Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.20,
    random_state = 42,
    stratify = y
)

print("Training features:", X_train.shape)
print("Test features:", X_test.shape)

print("\nTraining classes:")
print(y_train.value_counts())

print("\nTest classes:")
print(y_test.value_counts())


#Pipeline setup

numeric_features =["Age",
                   "EDUC",
                   "SES",
                   "MMSE",
                   "eTIV",
                   "nWBV",
                   "ASF"
                   ]

categorical_features =["M/F"]


numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)
categorical_pipeline = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)
log_reg_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42))
    ]
)


#Cross Validation

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

#Evaluate metrics
scoring = { 
    "accuracy":"accuracy",
    "recall":"recall",
    "precision":make_scorer(precision_score, zero_division=0),
    "f1":"f1",
    "roc_auc":"roc_auc"
}

dummy_cv = cross_validate(
    DummyClassifier(strategy="most_frequent"),
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)

print("Dummy Classifier")
for metric in scoring:
    scores = dummy_cv[f"test_{metric}"]
    print(f"{metric}: {scores.mean():.3f} +/- {scores.std():.3f}")

#Evaluate pipeline

log_reg_cv = cross_validate(
    log_reg_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)

#Random forest baseline

rf_numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)
rf_preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", rf_numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", rf_preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=42
        ))
    ]
)

rf_cv = cross_validate(
    rf_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)

print("\nLogistic Regression:")
for metric in scoring:
    scores = log_reg_cv[f"test_{metric}"]
    print(
        f"{metric}: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}"
    )

print("\nRandom Forest:")
for metric in scoring:
    scores = rf_cv[f"test_{metric}"]
    print(
        f"{metric}: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}"
    )

#Gradient Boosting baseline

gb_pipeline = Pipeline(
    steps=[
    ("preprocessor", rf_preprocessor),
    ("classifier", GradientBoostingClassifier(
        random_state=42))
   ]
)

gb_cv = cross_validate(
    gb_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)

print("\nGradient Boosting:")
for metric in scoring:
    scores = gb_cv[f"test_{metric}"]
    print(
        f"{metric}: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}"
    )


#Random Forest tuning
rf_param_grid = {
        "classifier__n_estimators": [100, 300,],
        "classifier__max_depth": [None, 3, 5,],
        "classifier__min_samples_leaf": [1, 2, 4]
    }


rf_grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs= 1,
)

rf_grid_search.fit(X_train, y_train)

print("\nBest Random Forest parameters:")
print(rf_grid_search.best_params_)

print(
    "Best Random Forest CV F1:",
    round(rf_grid_search.best_score_, 3)
)

#Logistic Regression tuning

log_reg_param_grid = {
    "classifier__C":[0.01, 0.1, 1, 10, 100]
}
log_reg_grid_search = GridSearchCV(
    estimator=log_reg_pipeline,
    param_grid=log_reg_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=1
)

log_reg_grid_search.fit(X_train, y_train)

print("\nBest Logistic Regression parameters")
print(log_reg_grid_search.best_params_)

print(
    "Best Logistic Regression CV F1:",
    round(log_reg_grid_search.best_score_, 3)
)

# Tuned model comparison

best_rf = rf_grid_search.best_estimator_
best_log_reg = log_reg_grid_search.best_estimator_

best_rf_cv = cross_validate(
    best_rf,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)

best_log_reg_cv = cross_validate(
    best_log_reg,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)

print("\nTuned Logistic Regression:")
for metric in scoring:
    scores = best_log_reg_cv[f"test_{metric}"]
    print(
        f"{metric}: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}"
    )

print("\nTuned Random Forest:")
for metric in scoring:
    scores = best_rf_cv[f"test_{metric}"]
    print(
        f"{metric}: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}"
    )


#Final test Evaluation

final_test_pred = best_rf.predict(X_test)

print("\nFinal Random Forest test confusion matrix:")
print(confusion_matrix(y_test, final_test_pred))

print("\nFinal Random Forest test classification report:")
print(classification_report(y_test, final_test_pred))

final_test_prob = best_rf.predict_proba(X_test)[:, 1]

final_test_roc_auc = roc_auc_score(y_test, final_test_prob)

print(
    "\nFinal Random Forest test ROC-AUC:",
    round(final_test_roc_auc, 3)
)

#Feature importance

perm_importance = permutation_importance(
    best_rf,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=20,
    random_state=42,
    n_jobs=1
)

importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": perm_importance.importances_mean,
    "importance_std": perm_importance.importances_std
})

importance_df = importance_df.sort_values(
    "importance_mean",
    ascending=False
)

print("\nPermutation importance: ")
print(importance_df)