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

catergorical_features =["M/F"]

#Impute missing values then standardise features
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)
catergorical_pipeline = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("catergorical", catergorical_pipeline, catergorical_features)
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
