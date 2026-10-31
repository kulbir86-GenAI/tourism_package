import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw tourism dataset
df = pd.read_csv(RAW_PATH)

# Clean Gender values
# "Fe male" is a variation of "Female"
df["Gender"] = df["Gender"].replace({
    "Fe male": "Female"
})

# Clean MaritalStatus values
# Merge "Unmarried" into "Single"
df["MaritalStatus"] = df["MaritalStatus"].replace({
    "Unmarried": "Single"
})

# Expected columns
expected_columns = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

# Validate expected columns
missing = [c for c in expected_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"Dataset is missing expected columns: {missing}"
    )

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\nGender values after cleaning:")
print(df["Gender"].value_counts())

print("\nMaritalStatus values after cleaning:")
print(df["MaritalStatus"].value_counts())

print("\nTarget column preview (ProdTaken):")
print(df["ProdTaken"].describe())

print("\nTarget value counts:")
print(df["ProdTaken"].value_counts())
