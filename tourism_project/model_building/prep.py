
# For data manipulation
import pandas as pd
import sklearn

# For train-test split
from sklearn.model_selection import train_test_split

# For converting categorical text into numerical representation
from sklearn.preprocessing import LabelEncoder


DATASET_PATH = "tourism_project/data/tourism.csv"

# Load the tourism dataset
df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")


# ---------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------

# Clean Gender values
# Merge "Fe male" with "Female"
df["Gender"] = df["Gender"].replace({
    "Fe male": "Female"
})

# Clean MaritalStatus values
# Merge "Unmarried" with "Single"
df["MaritalStatus"] = df["MaritalStatus"].replace({
    "Unmarried": "Single"
})


# ---------------------------------------------------------
# Drop columns that are not useful for modeling
# ---------------------------------------------------------

# CustomerID is a unique identifier and should not be used
# as a predictive feature.
df.drop(
    columns=["CustomerID"],
    inplace=True
)


# ---------------------------------------------------------
# Encode categorical columns
# ---------------------------------------------------------

label_encoder = LabelEncoder()

categorical_columns = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])


# ---------------------------------------------------------
# Define target variable
# ---------------------------------------------------------

target_col = "ProdTaken"


# ---------------------------------------------------------
# Split into X (features) and y (target)
# ---------------------------------------------------------

X = df.drop(columns=[target_col])
y = df[target_col]


# ---------------------------------------------------------
# Perform train-test split
# ---------------------------------------------------------

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# Save train/test datasets
# ---------------------------------------------------------

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)


print("Data prepared: train/test splits written.")

print(f"Xtrain shape: {Xtrain.shape}")
print(f"Xtest shape: {Xtest.shape}")
print(f"ytrain shape: {ytrain.shape}")
print(f"ytest shape: {ytest.shape}")

print("\nTarget distribution:")
print(y.value_counts())
