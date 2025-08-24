"""
Breast Cancer Detection Project
------------------------------

This script builds a classification model to predict whether a tumour is
benign or malignant based on various measurements computed from a digitised
image of a breast mass.  We use the Wisconsin Breast Cancer dataset,
available through `sklearn.datasets`.  The dataset contains 569 samples,
each described by 30 numeric features (mean, standard error and worst value
for radius, texture, perimeter, area, smoothness, compactness and more).

The workflow consists of:

1. Loading the dataset and examining its structure.
2. Splitting the data into training and testing subsets.
3. Scaling the features using `StandardScaler`.
4. Training a `LogisticRegression` classifier.
5. Evaluating the model using accuracy and a classification report.

Usage
-----
Run this script directly with Python::

    python breast_cancer_detection.py

It will output the accuracy and classification metrics for the two classes
(benign vs malignant).
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


def load_and_preprocess() -> tuple[pd.DataFrame, pd.Series]:
    """Load and standardise the breast cancer dataset.

    Returns
    -------
    X : pd.DataFrame
        Standardised feature matrix.
    y : pd.Series
        Binary labels: 0 indicates malignant, 1 indicates benign.
    """
    data = load_breast_cancer(as_frame=True)
    X_raw: pd.DataFrame = data.data
    y: pd.Series = data.target
    # Standardise the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = pd.DataFrame(X_scaled, columns=X_raw.columns)
    return X, y


def train_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train multiple classifiers and evaluate their performance.

    Returns a dictionary with metrics for Logistic Regression, Random
    Forest and Support Vector Machine classifiers.  For each model we
    compute test accuracy, cross‑validated accuracy and AUC.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    results['logistic_regression'] = {
        'model': lr,
        'test_accuracy': accuracy_score(y_test, y_pred_lr),
        'cross_val_accuracy': cross_val_score(lr, X, y, cv=5).mean(),
        'auc': roc_auc_score(y_test, y_prob_lr),
        'report': classification_report(y_test, y_pred_lr, target_names=["malignant", "benign"]),
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    results['random_forest'] = {
        'model': rf,
        'test_accuracy': accuracy_score(y_test, y_pred_rf),
        'cross_val_accuracy': cross_val_score(rf, X, y, cv=5).mean(),
        'auc': roc_auc_score(y_test, y_prob_rf),
        'report': classification_report(y_test, y_pred_rf, target_names=["malignant", "benign"]),
    }

    # Support Vector Machine with probability estimates
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    y_prob_svm = svm.predict_proba(X_test)[:, 1]
    results['svm'] = {
        'model': svm,
        'test_accuracy': accuracy_score(y_test, y_pred_svm),
        'cross_val_accuracy': cross_val_score(svm, X, y, cv=5).mean(),
        'auc': roc_auc_score(y_test, y_prob_svm),
        'report': classification_report(y_test, y_pred_svm, target_names=["malignant", "benign"]),
    }
    return results


def main() -> None:
    """Entry point."""
    X, y = load_and_preprocess()
    results = train_models(X, y)
    print("Breast Cancer Detection Results")
    print("--------------------------------")
    for name, res in results.items():
        print(f"\nModel: {name.replace('_', ' ').title()}")
        print(f"Test Accuracy: {res['test_accuracy']:.4f}")
        print(f"Cross‑Validated Accuracy (5‑fold): {res['cross_val_accuracy']:.4f}")
        print(f"AUC: {res['auc']:.4f}")
        print("Classification Report:")
        print(res['report'])


if __name__ == "__main__":
    main()
