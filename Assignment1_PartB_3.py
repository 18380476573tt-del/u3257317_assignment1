#!/usr/bin/env /opt/anaconda3/bin/python
"""
Assignment 1 - Part B-3: Classification Models 
"""

# Part B-3 Classification Models Only
print("\n" + "="*100)
print("PART B-3: CLASSIFICATION MODELS ONLY")
print("="*100)

# Import libraries
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

print("Starting Part B-3: Classification Models")

# ============================================================================
# Load and Prepare Data
# ============================================================================
print("\n" + "="*80)
print("DATA LOADING AND PREPARATION")
print("="*80)

# Auto-detect available data files
data_files = [
    'restaurant_data_processed.csv',
    'restaurant_data_processed_PartB.csv',
    'Zomato_df_final_data.csv',
    'restaurant_data.csv',
    'dataset.csv',
    'data.csv'
]

df = None
loaded_file = None

for file in data_files:
    if os.path.exists(file):
        try:
            df = pd.read_csv(file)
            loaded_file = file
            print(f"✓ Dataset loaded from: {file}")
            print(f"  Shape: {df.shape}")
            break
        except Exception as e:
            print(f"  Could not load {file}: {e}")
            continue

if df is None:
    print("Error: No data file found.")
    exit(1)

print(f"Using dataset: {loaded_file}")
print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ============================================================================
# Data Preprocessing for Classification
# ============================================================================
print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# 1. Handle missing values
print("1. Handling missing values...")
missing_before = df.isnull().sum().sum()
print(f"Missing values before: {missing_before}")

# Simple imputation
df_clean = df.copy()
for col in df_clean.columns:
    if df_clean[col].isnull().any():
        if df_clean[col].dtype in ['int64', 'float64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col] = df_clean[col].fillna(mode_val)

missing_after = df_clean.isnull().sum().sum()
print(f"Missing values after: {missing_after}")

# 2. Create binary target
print("\n2. Creating binary classification target...")

# Check rating-related columns
rating_columns = [col for col in df_clean.columns if 'rating' in col.lower()]
print(f"Rating-related columns found: {rating_columns}")

if 'rating_text' in df_clean.columns:
    print("Using 'rating_text' column for classification")
    print("Original rating_text distribution:")
    print(df_clean['rating_text'].value_counts())

    # Map to binary classes
    def simplify_rating(rating):
        if rating in ['Poor', 'Average']:
            return 'Class_1_Poor_Average'
        else:
            return 'Class_2_Good_VeryGood_Excellent'

    df_clean['rating_binary'] = df_clean['rating_text'].apply(simplify_rating)

elif 'rating_number' in df_clean.columns:
    print("Using 'rating_number' column to create binary classification")
    print("Rating number statistics:")
    print(f"Min: {df_clean['rating_number'].min()}, Max: {df_clean['rating_number'].max()}")

    # Build binary from numeric rating
    def create_binary_rating(rating):
        if rating <= 3.0:  # <=3.0 as Poor/Average
            return 'Class_1_Poor_Average'
        else:              # >3.0 as Good/Very Good/Excellent
            return 'Class_2_Good_VeryGood_Excellent'

    df_clean['rating_binary'] = df_clean['rating_number'].apply(create_binary_rating)
    df_clean['rating_text'] = df_clean['rating_number'].apply(
        lambda x: 'Poor' if x <= 2.0 else 'Average' if x <= 3.0 else 'Good' if x <= 4.0 else 'Excellent'
    )
else:
    print("Error: No rating column found for classification.")
    print("Available columns:", df_clean.columns.tolist())
    exit(1)

print("\nBinary class distribution:")
print(df_clean['rating_binary'].value_counts())

# 3. Prepare features
print("\n3. Preparing features...")

# Select numeric features
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
feature_columns = [col for col in numeric_columns if col != 'rating_number']

print(f"Selected {len(feature_columns)} numeric features")
print(f"Feature columns: {feature_columns}")

if len(feature_columns) == 0:
    print("Error: No numeric features found.")
    # Fallback: encode some categorical features
    for col in df_clean.columns:
        if col not in ['rating_text', 'rating_binary']:
            if df_clean[col].dtype == 'object':
                df_clean[f'{col}_encoded'] = pd.factorize(df_clean[col])[0]
                feature_columns.append(f'{col}_encoded')

    print(f"Created {len(feature_columns)} encoded features")

X = df_clean[feature_columns]
y = df_clean['rating_binary']

print(f"Final feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

# ============================================================================
# Classification Models
# ============================================================================
print("\n" + "="*80)
print("CLASSIFICATION MODELS")
print("="*80)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

print("All libraries imported successfully.")

# 1. Standardize and split
print("\n1. Standardizing features and splitting data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training class distribution:\n{y_train.value_counts()}")

# 2. Model 1: Logistic Regression
print("\n" + "="*50)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*50)

logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

print("--- Logistic Regression Results ---")
print(classification_report(y_test, y_pred_logreg))

# Confusion matrix
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_logreg, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Class 1', 'Class 2'],
    yticklabels=['Class 1', 'Class 2']
)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_logreg.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Additional classifiers
print("\n" + "="*50)
print("MODELS 2-4: ADDITIONAL CLASSIFIERS")
print("="*50)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='Class_2_Good_VeryGood_Excellent')
    recall = recall_score(y_test, y_pred, pos_label='Class_2_Good_VeryGood_Excellent')
    f1 = f1_score(y_test, y_pred, pos_label='Class_2_Good_VeryGood_Excellent')

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# 4. Comparison
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

# Add Logistic Regression metrics
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
logreg_precision = precision_score(y_test, y_pred_logreg, pos_label='Class_2_Good_VeryGood_Excellent')
logreg_recall = recall_score(y_test, y_pred_logreg, pos_label='Class_2_Good_VeryGood_Excellent')
logreg_f1 = f1_score(y_test, y_pred_logreg, pos_label='Class_2_Good_VeryGood_Excellent')

results['Logistic Regression'] = {
    'Accuracy': logreg_accuracy,
    'Precision': logreg_precision,
    'Recall': logreg_recall,
    'F1-Score': logreg_f1
}

# Build comparison table
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(4)

print("\nModel Performance Comparison Table:")
print(comparison_df)

# Best models
best_f1_model = comparison_df['F1-Score'].idxmax()
best_accuracy_model = comparison_df['Accuracy'].idxmax()

print(f"\nBest Model by F1-Score: {best_f1_model} (F1 = {comparison_df.loc[best_f1_model, 'F1-Score']:.4f})")
print(f"Best Model by Accuracy: {best_accuracy_model} (Accuracy = {comparison_df.loc[best_accuracy_model, 'Accuracy']:.4f})")

# 5. Visualization
print("\n" + "="*50)
print("RESULTS VISUALIZATION")
print("="*50)

plt.figure(figsize=(12, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
models_list = list(results.keys())

for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    values = [results[model][metric] for model in models_list]
    bars = plt.bar(models_list, values)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    plt.title(f'{metric} Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('classification_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*100)
print("PART B-3 CLASSIFICATION COMPLETED SUCCESSFULLY")
print("="*100)
print("Summary:")
print(f"- Dataset: {loaded_file}")
print(f"- Binary Classes: Class 1 (Poor + Average) vs Class 2 (Good + Very Good + Excellent)")
print(f"- Models Trained: 5 classification models")
print(f"- Best Model: {best_f1_model} (F1-Score: {comparison_df.loc[best_f1_model, 'F1-Score']:.4f})")
print(f"- Files Saved: confusion_matrix_logreg.png, classification_performance_comparison.png")
print("="*100)
