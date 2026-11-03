import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

file_path = r"C:\Users\chalv\OneDrive\Desktop\PREDITIVE\drug_consumption_cleaned.csv"

df = pd.read_csv(file_path)

sns.set_theme(style="whitegrid")

print("=" * 55)
print("DRUG CONSUMPTION DATASET - EDA")
print("=" * 55)

# ------------------------------------------------------------
# 2. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\nDataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all"))

# ------------------------------------------------------------
# 3. DATA QUALITY CHECKS
# ------------------------------------------------------------

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())

print("\nUnique Values in Each Column:")
print(df.nunique())

# Remove duplicate rows if any
df = df.drop_duplicates()

# ------------------------------------------------------------
# 4. IDENTIFY TARGET COLUMN
# ------------------------------------------------------------

# Update this name if your cleaned CSV uses a different target name.
target = "Cannabis"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

print("\nTarget Variable:", target)

# ------------------------------------------------------------
# 5. TARGET VARIABLE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x=target,
    order=df[target].value_counts().index
)

plt.title("Cannabis Consumption Category Distribution")
plt.xlabel("Consumption Category")
plt.ylabel("Number of Respondents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nTarget Category Counts:")
print(df[target].value_counts())

print("\nTarget Category Percentages:")
print((df[target].value_counts(normalize=True) * 100).round(2))

# ------------------------------------------------------------
# 6. NUMERICAL FEATURE DISTRIBUTIONS
# ------------------------------------------------------------

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Exclude ID from feature distributions if present
numeric_features = [
    col for col in numeric_cols
    if col.lower() not in ["id", "id_"]
]

if numeric_features:
    df[numeric_features].hist(
        figsize=(15, 10),
        bins=20,
        edgecolor="black"
    )

    plt.suptitle("Distribution of Numerical Features")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 7. CATEGORICAL FEATURE DISTRIBUTIONS
# ------------------------------------------------------------

categorical_cols = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

categorical_features = [
    col for col in categorical_cols if col != target
]

for col in categorical_features[:8]:
    plt.figure(figsize=(8, 4))

    order = df[col].value_counts().index[:15]

    sns.countplot(data=df, x=col, order=order)

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 8. PERSONALITY FEATURES VS TARGET
# ------------------------------------------------------------

personality_candidates = [
    "Nscore", "Escore", "Oscore",
    "Ascore", "Cscore", "Impulsive", "SS"
]

personality = [
    col for col in personality_candidates
    if col in df.columns
]

if personality:
    for col in personality:
        plt.figure(figsize=(8, 5))

        sns.boxplot(data=df, x=target, y=col)

        plt.title(f"{col} vs {target}")
        plt.xlabel("Cannabis Consumption Category")
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# ------------------------------------------------------------
# 9. CORRELATION HEATMAP
# ------------------------------------------------------------

# Use numerical columns only; correlation does not imply causation.
# Exclude ID because it is an identifier, not a meaningful predictor.
correlation_cols = [
    col for col in numeric_cols
    if col.lower() not in ["id", "id_"]
]

if len(correlation_cols) >= 2:
    plt.figure(figsize=(14, 10))

    sns.heatmap(
        df[correlation_cols].corr(),
        cmap="coolwarm",
        center=0,
        annot=False
    )

    plt.title("Correlation Heatmap of Numerical Variables")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 10. MEAN PERSONALITY SCORES BY TARGET CATEGORY
# ------------------------------------------------------------

if personality:
    mean_scores = df.groupby(target)[personality].mean()

    mean_scores.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Mean Personality Scores by Cannabis Category")
    plt.xlabel("Cannabis Consumption Category")
    plt.ylabel("Mean Score")
    plt.xticks(rotation=45)
    plt.legend(
        title="Personality Traits",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 11. FINAL EDA SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("FINAL EDA SUMMARY")
print("=" * 55)

print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows Remaining:", df.duplicated().sum())

print("\nTarget Variable Distribution:")
print(df[target].value_counts().sort_index())

print("\nNumerical Columns:", len(numeric_cols))
print("Categorical Columns:", len(categorical_cols))

print("\nEDA COMPLETED SUCCESSFULLY!")
