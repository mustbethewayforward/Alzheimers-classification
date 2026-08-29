import pandas as pd


df = pd.read_csv("data/oasis_longitudinal.csv")


print(df.shape)
print(df.columns.tolist())

print(df.head())
print(df['Group'].value_counts())
print(df['Subject ID'].nunique())

#count unique participants in each group
print(df.groupby('Group')['Subject ID'].nunique())


#inspect how dementia status changes across visits for converted participants
converted = df[df['Group']=='Converted']
print(
    converted[["Subject ID", "Visit", "Age", "MMSE", "CDR"]
              ].sort_values(["Subject ID","Visit"])
)

print(pd.crosstab(df["Group"], df["CDR"], margins=True))

#inspect nondemented-group visits with a non-zero dementia rating
nondemented_cdr_mismatch = df[(df["Group"] == "Nondemented") & (df["CDR"] > 0)]
print(
    nondemented_cdr_mismatch[
        ["Subject ID", "Visit", "Age", "MMSE", "CDR"]]
)
#History of subjects with conflicting group and CDR
mismatch_subjects = ["OAS2_0005", "OAS2_0017"]
mismatch_history = df[df["Subject ID"].isin(mismatch_subjects)]
print(mismatch_history)

baseline_visits = df[df["Visit"] == 1]

print(baseline_visits.shape)
print(baseline_visits ["Subject ID"].nunique())
print(baseline_visits["Group"].value_counts())

#compare diagnostic groups with dementia ratings at baseline visi
print(pd.crosstab(
    baseline_visits["Group"],
    baseline_visits["CDR"],
    margins=True
))

print(
    baseline_visits[
        baseline_visits["Group"] == "Converted"
    ][ ["Subject ID", "Age", "MMSE", "CDR"]]
)

#prepare modelling data

model_df = baseline_visits[
    baseline_visits["Group"].isin(["Nondemented", "Demented"])
].copy()

print(model_df.shape)
print(model_df["Group"].value_counts())
print(model_df["Subject ID"].nunique())

# inspect features

model_df.info()
print(model_df.isna().sum())

print(model_df[["Visit","MR Delay", "M/F", "Hand"]].nunique())
for column in ["Visit", "MR Delay", "M/F", "Hand"]:
    print(f"\n{column}")
    print(model_df[column].value_counts())


ses_missing_by_group = (
    model_df.groupby("Group")["SES"]
    .agg(
        n_subjects="size",
        n_missing=lambda s: s.isna().sum(),
        pct_missing=lambda s: s.isna().mean() * 100
    )
)

print("\nSES missingness by Group:")
print(ses_missing_by_group)  


print("\neTIV/ASF correlation:")
print(model_df[["eTIV", "ASF"]].corr())


baseline_df = df[df["Visit"]==1].copy()
converted_baseline = baseline_df[baseline_df["Group"] == "Converted"]

print("\nBaseline CDR among Converted subjects:")
print(converted_baseline["CDR"].value_counts().sort_index())