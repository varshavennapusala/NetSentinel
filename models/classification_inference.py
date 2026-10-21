"""
NetSentinel — Multi-Class Cyber Intrusion Detection Inference Engine
Course: 23CSE301 Machine Learning Capstone
Review 2 — Classification Track (Part A + Part B)

This module provides a production-ready interface for training, evaluating,
and predicting multi-class network threats using the best-performing model
identified in the 10-model benchmark (Tuned Random Forest Classifier).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score


class NetSentinelClassifier:
    """
    Production-ready wrapper for training, evaluating, and serving
    the NetSentinel multi-class intrusion detection pipeline.
    """

    def __init__(self, random_state: int = 42, n_estimators: int = 200, max_depth: int = 20):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_names = None
        self.classes = None

    def fit(self, data_path: str = "data/classification/network_traffic_classification.csv"):
        """
        Loads dataset, performs standard preprocessing with zero data-leakage,
        and trains the model.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

        print(f"Loading classification dataset from: {data_path}")
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()

        # Handle duplicates & infinities
        df = df.drop_duplicates()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Feature matrix & target
        X = df.drop(columns=["Label"])
        y = df["Label"]
        self.feature_names = list(X.columns)

        # Stratified 80:20 split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Scale fitted strictly on train partition
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)

        print("Fitting Random Forest Classifier...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.classes = list(self.model.classes_)

        # Evaluation
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        proba = self.model.predict_proba(X_test)
        roc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")

        print("\n" + "=" * 60)
        print("NetSentinel Classifier Evaluation Summary")
        print("=" * 60)
        print(f"Accuracy:      {acc:.6f} ({acc*100:.2f}%)")
        print(f"Weighted F1:   {f1:.6f}")
        print(f"ROC-AUC (OvR): {roc:.6f}")
        print("=" * 60)
        return {"accuracy": acc, "f1_weighted": f1, "roc_auc": roc}

    def predict(self, sample_features: np.ndarray) -> np.ndarray:
        """
        Predict threat class for incoming raw or scaled feature vectors.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        scaled = self.scaler.transform(sample_features)
        return self.model.predict(scaled)


if __name__ == "__main__":
    classifier = NetSentinelClassifier()
    classifier.fit()
