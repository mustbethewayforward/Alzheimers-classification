import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
#


df = pd.read_csv("data/oasis_longitudinal.csv")
baseline_df = df[df["Visit"]==1].copy()

model_df = baseline_df[baseline_df["Group"].isin(["Demented", "Nondemented"])].copy()

#
#Define features and target
feature_columns = ["M/F","Age","EDUC", 
                   "SES", "MMSE", "eTIV", "nWBV", "ASF"
]

X = model_df[feature_columns]

#nondemented encoded as 0 demented as 1
y = model_df["Group"].map({"Nondemented": 0, "Demented": 1})

#Train test split

# leave 20% of population for final evaluation
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


#
#Preprocessing
#

# Fill missing SES values using median calculated from training data
ses_imputer = SimpleImputer(strategy = "median")
ses_imputer.fit(X_train[["SES"]])
print("Training SES median:", ses_imputer.statistics_[0])

#apply training set median to missing SES values
X_train["SES"] = ses_imputer.transform(X_train[["SES"]]).ravel()
X_test["SES"] = ses_imputer.transform(X_test[["SES"]]).ravel()

print("Missing SES in training:", X_train["SES"].isna().sum())
print("Missing SES in test:", X_test["SES"].isna().sum())

# Encode sex as a numeric binary feature
sex_mapping = {
    "F": 0,
    "M": 1,

}

X_train["M/F"] = X_train["M/F"].map(sex_mapping)
X_test["M/F"] = X_test["M/F"].map(sex_mapping)

print(X_train.dtypes)
print("\nEncoded sex values:")
print(X_train["M/F"].value_counts())

#Check for any remaining missing values
print("\nMissing training values:")
print(X_train.isna().sum())

print("\nMissing test values")
print(X_test.isna().sum())

#
#Feature scaling

# learn scaling parameters from training data only
scaler = StandardScaler()
scaler.fit(X_train)
x_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(x_train_scaled.shape)
print(X_test_scaled.shape)

#
#Logistic regression baseline

#train binary classifier on scaled training data
log_reg = LogisticRegression(random_state=42)
log_reg.fit(x_train_scaled, y_train)

print("Logistic regression trained successfully")

#training set evaluation
y_train_pred = log_reg.predict(x_train_scaled)

#compare predicted labels with the true training labels
print(confusion_matrix(y_train,y_train_pred))

print("\nTraining classification report:")
print(classification_report(y_train, y_train_pred))

#
#Test set evaluation

y_test_pred = log_reg.predict(X_test_scaled)

print(confusion_matrix(y_test,y_test_pred))

print("\n Test classification report:")
print(classification_report(y_test, y_test_pred))

#
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

#Impute missing values then standardise features
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

log_reg_pipeline.fit(X_train, y_train)

pipeline_train_pred = log_reg_pipeline.predict(X_train)

print("\nPipeline training classification report:")
print(classification_report(y_train, pipeline_train_pred))

#
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
    "precision":"precision",
    "f1":"f1",
    "roc_auc":"roc_auc"
}

#Evaluate pipeline

log_reg_cv = cross_validate(
    log_reg_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)

for metric in scoring:
    mean_score = log_reg_cv[f"test_{metric}"].mean()
    print(f"Mean CV {metric}: {mean_score:.3f}")


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

#fold by fold comparison

print("\nFold by fold F1 scores:")
for fold, (lr_score, rf_score) in enumerate(
    zip(log_reg_cv["test_f1"], rf_cv["test_f1"]),
    start=1
):
    print(
        f"Fold {fold}: "
        f"Logistic Regression ={lr_score:.3f}, "
        f"Random Forest = {rf_score:.3f}"

    )


#RF tuning
rf_param_grid = {
        "classifier__n_estimators": [100, 300, 500],
        "classifier__max_depth": [None, 3, 5, 8],
        "classifier__min_samples_leaf": [1, 2, 4]
    }


rf_grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs= 1,
    verbose= 2
)

rf_grid_search.fit(X_train, y_train)

print("\nBest Random Forest parameters:")
print(rf_grid_search.best_params_)

print(
    "Best Random Forest CV F1:",
    round(rf_grid_search.best_score_, 3)
)


#LR tuning

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