# for data manipulation
import pandas as pd
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data into numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define the dataset path from Hugging Face Hub
# Update with your username
DATASET_PATH = "hf://datasets/Manish9119/tourism-package-prediction/tourism.csv"

# Load the dataset
print("Loading dataset from Hugging Face Hub...")
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded successfully. Shape: {df.shape}")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)

# Handle any data quality issues (e.g., "Fe Male" -> "Female")
df['Gender'] = df['Gender'].str.strip().str.replace('Fe Male', 'Female')

# Encode categorical variables
label_encoders = {}
categorical_columns = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
                       'MaritalStatus', 'Designation']

for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Define target column
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
print("Splitting data into train and test sets...")
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {Xtrain.shape[0]}")
print(f"Test set size: {Xtest.shape[0]}")

# Save the split datasets locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Saved split datasets locally.")

# Upload processed files to Hugging Face Hub
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

print("Uploading processed datasets to Hugging Face Hub...")
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="Manish9119/tourism-package-prediction",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path}")

print("Data preparation completed successfully!")
