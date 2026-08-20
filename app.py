import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

print("\nLoading dataset...")

data_path = "data/diabetes-dataset.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ==========================================================
# 2. HANDLE MISSING / INVALID VALUES
# ==========================================================

print("\nPreprocessing data...")

columns_with_zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in columns_with_zero_as_missing:
    df[column] = df[column].replace(0, np.nan)


# ==========================================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# ==========================================================
# 4. TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# ==========================================================
# 5. DEFINE MACHINE LEARNING MODELS
# ==========================================================

models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]),

    "XGBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(
            probability=True,
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "MLP": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42
        ))
    ])
}


# ==========================================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================================

results = []

confusion_matrices = {}

best_model = None
best_model_name = None
best_auc = -1


print("\n")
print("=" * 70)
print("TRAINING AND EVALUATING MODELS")
print("=" * 70)


for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Probability
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    confusion_matrices[name] = cm

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

    # Print results
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("Confusion Matrix:")
    print(cm)

    # Find best model
    if roc_auc > best_auc:

        best_auc = roc_auc
        best_model = model
        best_model_name = name


# ==========================================================
# 7. CREATE RESULTS DATAFRAME
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ==========================================================
# 8. SAVE MODEL RESULTS
# ==========================================================

print("\nSaving model comparison results...")

results_df.to_csv(
    "model_results.csv",
    index=False
)

print("model_results.csv created successfully!")


# ==========================================================
# 9. SAVE CONFUSION MATRICES
# ==========================================================

print("\nSaving confusion matrices...")

os.makedirs(
    "results",
    exist_ok=True
)

for model_name, cm in confusion_matrices.items():

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    cm_df.to_csv(
        f"results/{safe_name}_confusion_matrix.csv"
    )


print("Confusion matrices saved successfully!")


# ==========================================================
# 10. SAVE BEST MODEL
# ==========================================================

print("\nSaving best model...")

os.makedirs(
    "models",
    exist_ok=True
)

model_path = "models/diabetes_model.pkl"

joblib.dump(
    best_model,
    model_path
)


# ==========================================================
# 11. FINAL OUTPUT
# ==========================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Best Model     : {best_model_name}")
print(f"Best ROC-AUC   : {best_auc:.4f}")
print(f"Model Location : {model_path}")
print("Results File   : model_results.csv")
print("Results Folder : results/")
print("=" * 70)