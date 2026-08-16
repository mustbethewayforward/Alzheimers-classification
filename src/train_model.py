import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
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
Y = model_df["Group"].map({"Nondemented": 0, "Demented": 1})

#Train test split

# leave 20% of population for final evaluation
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size = 0.20,
    random_state = 42,
    stratify = Y
)

print("Training features:", X_train.shape)
print("Test features:", X_test.shape)

print("\nTraining classes:")
print(Y_train.value_counts())

print("\nTest classes:")
print(Y_test.value_counts())