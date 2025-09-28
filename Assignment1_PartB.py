#!/usr/bin/env /opt/anaconda3/bin/python

# Assignment 1 - Part B: Predictive Modelling

print("\n" + "="*100)
print("PART B: PREDICTIVE MODELLING")
print("="*100)

# Import libraries
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

print("Starting Part B: Predictive Modelling")

# ============================================================================
# Part B - 1. Feature Engineering
# ============================================================================
print("\n" + "="*50)
print("PART B - 1: FEATURE ENGINEERING")

# Automatically detect an available data file
data_files = [
    'restaurant_data_processed.csv',  # processed data
    'Zomato_df_final_data.csv',       # raw data
    'restaurant_data.csv',
    'dataset.csv',
    'data.csv'
]

df_clean = None
loaded_file = None

for file in data_files:
    if os.path.exists(file):
        try:
            df_clean = pd.read_csv(file)
            loaded_file = file
            print(f"Dataset loaded from: {file}")
            print(f"  Shape: {df_clean.shape}")
            print(f"  Columns: {list(df_clean.columns)}")
            break
        except Exception as e:
            print(f"  Could not load {file}: {e}")
            continue

if df_clean is None:
    print("Error: No data file found!")
    print("CSV files available in the current directory:")
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            print(f"  - {file}")
    exit(1)

print(f"Using dataset: {loaded_file}")
print(f"Original shape: {df_clean.shape}")

print("Starting feature engineering")

# 1) Handle missing/invalid data (drop or impute)
print("\n1. Handling Missing/Invalid Data")
print("Rationale: Use imputation to preserve data points and maintain dataset size")

missing_before = df_clean.isnull().sum().sum()
print(f"Total missing values before: {missing_before}")

# Show per-column missing counts
print("\nMissing values per column:")
for col in df_clean.columns:
    missing_count = df_clean[col].isnull().sum()
    if missing_count > 0:
        print(f"  {col}: {missing_count} missing")

# Imputation strategy
for col in df_clean.columns:
    if df_clean[col].isnull().any():
        if df_clean[col].dtype in ['int64', 'float64']:
            # Numerical columns: median (robust to outliers)
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            print(f"Numerical column '{col}' imputed with median")
        else:
            # Categorical columns: mode (most frequent value)
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"Categorical column '{col}' imputed with mode")

missing_after = df_clean.isnull().sum().sum()
print(f"Total missing values after: {missing_after}")

# 2) Encode categorical features
print("\n2. Encoding Categorical Features")

categorical_cols = df_clean.select_dtypes(include=['object']).columns
print(f"Categorical columns to encode: {list(categorical_cols)}")

for col in categorical_cols:
    if df_clean[col].nunique() <= 10:  # low cardinality: one-hot encoding
        dummies = pd.get_dummies(df_clean[col], prefix=col, prefix_sep='_')
        df_clean = pd.concat([df_clean, dummies], axis=1)
        print(f"'{col}' - one-hot encoded ({df_clean[col].nunique()} categories)")
    else:  # high cardinality: label encoding
        unique_vals = df_clean[col].unique()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        df_clean[f'{col}_encoded'] = df_clean[col].map(mapping)
        print(f"'{col}' - label encoded ({len(unique_vals)} categories)")

# Drop original categorical columns to avoid duplication
print("\nRemoving original categorical columns...")
df_clean = df_clean.drop(columns=categorical_cols)

# 3) Create useful features
print("\n3. Creating Useful Features")

# Cost bins (if a 'cost' column exists)
if 'cost' in df_clean.columns:
    df_clean['cost_bins'] = pd.cut(
        df_clean['cost'],
        bins=[0, 300, 600, float('inf')],
        labels=['Low', 'Medium', 'High']
    )
    print("Created cost_bins: Low(<300), Medium(300–600), High(>600)")

# Rating bins (if a 'rating_number' column exists)
if 'rating_number' in df_clean.columns:
    df_clean['rating_bins'] = pd.cut(
        df_clean['rating_number'],
        bins=[0, 3.0, 4.0, 5.0],
        labels=['Low', 'Medium', 'High']
    )
    print("Created rating_bins: Low(<3.0), Medium(3.0–4.0), High(>4.0)")

# Value-for-money feature (rating per cost)
if 'cost' in df_clean.columns and 'rating_number' in df_clean.columns:
    df_clean['value_score'] = df_clean['rating_number'] / (df_clean['cost'] / 100 + 1)
    print("Created value_score feature (rating/cost ratio)")

print("\nFeature Engineering Completed!")
print(f"Final dataset shape: {df_clean.shape}")
print(f"Number of features: {len(df_clean.columns)}")

# Save processed dataset
output_file = 'restaurant_data_processed_PartB.csv'
df_clean.to_csv(output_file, index=False)
print(f"Processed data saved to: {output_file}")

# ============================================================================
# Part B - 2. Regression Models
# ============================================================================
print("\n" + "="*80)
print("PART B - 2: REGRESSION MODELS")
print("="*80)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("scikit-learn imported successfully.")

# Prepare features and target variable
print("\nPreparing features and target variable...")

print("Available columns:", df_clean.columns.tolist())

exclude_columns = ['rating_number', 'rating_text', 'rating_bins']
feature_columns = []

for col in df_clean.columns:
    if col not in exclude_columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            feature_columns.append(col)

print(f"Selected {len(feature_columns)} feature columns")

if len(feature_columns) == 0:
    print("Error: No numeric feature columns found.")
    # Fallback: use all columns except excluded ones
    feature_columns = [col for col in df_clean.columns if col not in exclude_columns]
    print(f"Using all available columns as features: {feature_columns}")

# Define X and y
X = df_clean[feature_columns]
y = df_clean['rating_number']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Number of features: {len(feature_columns)}")

# Check missing values
print(f"Missing values in X: {X.isnull().sum().sum()}")
print(f"Missing values in y: {y.isnull().sum()}")

# Handle missing values if any
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    print("Handling missing values by median imputation for X and y...")
    X = X.fillna(X.median())
    y = y.fillna(y.median())

# Split data
print("\nSplitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Model A: Linear Regression (scikit-learn)
print("\n--- Model A: Linear Regression (scikit-learn) ---")
model_a = LinearRegression()
model_a.fit(X_train, y_train)
y_pred_a = model_a.predict(X_test)
mse_a = mean_squared_error(y_test, y_pred_a)
print(f"MSE for Model A: {mse_a:.4f}")

# Model B: Manual Gradient Descent Regression
print("\n--- Model B: Gradient Descent Regression (Manual) ---")

class GradientDescentRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 200 == 0:
                mse = np.mean((y_pred - y) ** 2)
                print(f"  Iteration {i}: MSE = {mse:.4f}")
        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

model_b = GradientDescentRegression(learning_rate=0.01, n_iterations=1000)
model_b.fit(X_train.values, y_train.values)
y_pred_b = model_b.predict(X_test.values)
mse_b = mean_squared_error(y_test, y_pred_b)
print(f"MSE for Model B: {mse_b:.4f}")

# Results summary
print("\n" + "="*100)
print("RESULTS SUMMARY")
print("="*100)
print(f"Model A (scikit-learn Linear Regression): MSE = {mse_a:.4f}")
print(f"Model B (Manual Gradient Descent): MSE = {mse_b:.4f}")

print("\n" + "="*100)
print("Summary:")
print(f"- Dataset used: {loaded_file}")
print(f"- Feature engineering: {df_clean.shape[1]} total features")
print(f"- Regression models: 2 models trained and evaluated")
print(f"- Data split: 80% training, 20% testing")
print(f"- Best performing model: {'Model A' if mse_a < mse_b else 'Model B'}")
print("="*100)
