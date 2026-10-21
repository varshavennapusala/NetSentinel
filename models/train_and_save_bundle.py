"""
NetSentinel — Model Bundle Generation for Ultra-Lightweight Deployment
Trains all 10 regression models, 10 classification models, and clustering models
on representative samples and exports a compressed joblib artifact (< 3 MB).
"""

import os
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_bundle():
    print("[1/4] Training Regression Models (UNSW-NB15 QoS Packet Loss)...")
    reg_csv = "data/regression/UNSW_NB15_training-set.csv"
    if not os.path.exists(reg_csv):
        raise FileNotFoundError(f"Missing {reg_csv}")

    df_reg = pd.read_csv(reg_csv, nrows=10000)
    reg_features = ["sbytes", "dbytes", "spkts", "dpkts", "dur", "sload", "dload", "rate", "sttl", "dttl"]
    X_reg = df_reg[reg_features].copy()
    X_reg["total_bytes"] = X_reg["sbytes"] + X_reg["dbytes"]
    y_reg = df_reg["sloss"].copy()

    reg_scaler = StandardScaler()
    X_reg_scaled = reg_scaler.fit_transform(X_reg)

    # 10 Regression Models
    poly_pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("linear", LinearRegression())
    ])
    # Fit polynomial on top 3 features for compactness and speed
    poly_pipeline.fit(X_reg[["sbytes", "spkts", "dur"]], y_reg)

    reg_models = {
        "Random Forest Regressor (Tuned)": RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1).fit(X_reg_scaled, y_reg),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=5).fit(X_reg_scaled, y_reg),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=30, max_depth=4, random_state=42).fit(X_reg_scaled, y_reg),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, random_state=42).fit(X_reg_scaled, y_reg),
        "Polynomial Regression (Degree 2)": poly_pipeline,
        "Ridge Regression (Tuned)": Ridge(alpha=10.0, random_state=42).fit(X_reg_scaled, y_reg),
        "Linear Regression (OLS)": LinearRegression().fit(X_reg_scaled, y_reg),
        "Support Vector Regressor (SVR)": LinearSVR(C=1.0, random_state=42, max_iter=2500).fit(X_reg_scaled, y_reg),
        "Lasso Regression": Lasso(alpha=0.1, random_state=42).fit(X_reg_scaled, y_reg),
        "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42).fit(X_reg_scaled, y_reg)
    }

    print("[2/4] Training Classification Models (CIC-IDS2017 Threat Detection)...")
    clf_csv = "data/classification/network_traffic_classification.csv"
    if not os.path.exists(clf_csv):
        raise FileNotFoundError(f"Missing {clf_csv}")

    df_clf = pd.read_csv(clf_csv)
    df_clf.columns = df_clf.columns.str.strip()
    df_clf = df_clf.drop_duplicates()
    df_clf.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clf.dropna(inplace=True)

    clf_features = [
        "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Fwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
        "Flow IAT Mean", "Packet Length Mean", "Packet Length Std", "SYN Flag Count", "ACK Flag Count"
    ]
    X_clf = df_clf[clf_features].copy()
    y_clf = df_clf["Label"].copy()

    clf_scaler = StandardScaler()
    X_clf_scaled = clf_scaler.fit_transform(X_clf)

    # 10 Classification Models with Balanced Class Weights for Minority Threat Sensitivity
    clf_models = {
        "Random Forest Classifier (Tuned)": RandomForestClassifier(n_estimators=45, max_depth=14, class_weight="balanced", random_state=42, n_jobs=-1).fit(X_clf_scaled, y_clf),
        "Gradient Boosting Classifier (Tuned)": GradientBoostingClassifier(n_estimators=35, learning_rate=0.12, max_depth=3, random_state=42).fit(X_clf_scaled, y_clf),
        "Bagging Classifier": BaggingClassifier(n_estimators=25, random_state=42).fit(X_clf_scaled, y_clf),
        "Logistic Regression": LogisticRegression(max_iter=600, class_weight="balanced", random_state=42).fit(X_clf_scaled, y_clf),
        "Support Vector Machine (SVC)": LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000).fit(X_clf_scaled, y_clf),
        "MLP Classifier (Neural Net)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42).fit(X_clf_scaled, y_clf),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42).fit(X_clf_scaled, y_clf),
        "AdaBoost Classifier": AdaBoostClassifier(n_estimators=35, learning_rate=0.5, random_state=42).fit(X_clf_scaled, y_clf),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=3).fit(X_clf_scaled, y_clf),
        "Gaussian Naive Bayes": GaussianNB().fit(X_clf_scaled, y_clf)
    }

    print("[3/4] Training Clustering Models (Unsupervised Behavioral Profiling)...")
    km_model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_clf_scaled)
    # Agglomerative clustering sample for fast centroid/linkage representation
    sample_indices = np.random.RandomState(42).choice(len(X_clf_scaled), size=min(1500, len(X_clf_scaled)), replace=False)
    hc_sample_data = X_clf_scaled[sample_indices]
    hc_model = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(hc_sample_data)

    # Compute hierarchical cluster centers approximation
    hc_centroids = np.array([hc_sample_data[hc_model.labels_ == k].mean(axis=0) for k in range(3)])

    print("[4/4] Serializing deployment bundle...")
    bundle = {
        "version": "2.0-deployment",
        "reg_features": reg_features,
        "reg_scaler": reg_scaler,
        "reg_models": reg_models,
        "clf_features": clf_features,
        "clf_scaler": clf_scaler,
        "clf_models": clf_models,
        "clf_classes": list(clf_models["Random Forest Classifier (Tuned)"].classes_),
        "km_model": km_model,
        "hc_centroids": hc_centroids
    }

    out_dir = "models/bundle"
    os.makedirs(out_dir, exist_ok=True)
    bundle_path = os.path.join(out_dir, "model_bundle.joblib")
    joblib.dump(bundle, bundle_path, compress=3)

    size_mb = os.path.getsize(bundle_path) / (1024 * 1024)
    print(f"✅ Bundle successfully created at: {bundle_path}")
    print(f"📦 Size: {size_mb:.2f} MB (Optimized for Streamlit Cloud & Render deployment)")


if __name__ == "__main__":
    generate_bundle()
