"""
NetSentinel — Unsupervised Traffic Behavioral Clustering Inference Engine
Course: 23CSE301 Machine Learning Capstone
Review 2 — Clustering Track

This module provides a production-ready interface for fitting, evaluating,
and visualising unsupervised cluster structures on network telemetry flows
using K-Means and Agglomerative Hierarchical Clustering.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA


class NetSentinelClusterer:
    """
    Production-ready wrapper for unsupervised network traffic clustering.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        self.hc_model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage="ward"
        )
        self.pca_2d = PCA(n_components=2, random_state=self.random_state)
        self.is_fitted = False

    def fit(self, data_path: str = "data/classification/network_traffic_classification.csv"):
        """
        Loads dataset, removes duplicates & labels, scales features,
        and fits both clustering models.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

        print(f"Loading clustering dataset from: {data_path}")
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()

        # Unsupervised protocol: exclude target labels
        df_clean = df.drop_duplicates().copy()
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean.dropna(inplace=True)

        X_num = df_clean.select_dtypes(include=[np.number]).copy()
        print(f"Feature matrix shape for clustering: {X_num.shape}")

        # Standardise features
        X_scaled = self.scaler.fit_transform(X_num)

        # Fit K-Means
        print("Fitting K-Means Clustering...")
        km_labels = self.kmeans_model.fit_predict(X_scaled)

        # Fit Hierarchical Clustering
        print("Fitting Agglomerative Hierarchical Clustering...")
        hc_labels = self.hc_model.fit_predict(X_scaled)

        # Compute Metrics
        sample_size = min(5000, len(X_scaled))
        km_sil = silhouette_score(X_scaled, km_labels, sample_size=sample_size, random_state=self.random_state)
        km_db = davies_bouldin_score(X_scaled, km_labels)
        km_ch = calinski_harabasz_score(X_scaled, km_labels)

        hc_sil = silhouette_score(X_scaled, hc_labels, sample_size=sample_size, random_state=self.random_state)
        hc_db = davies_bouldin_score(X_scaled, hc_labels)
        hc_ch = calinski_harabasz_score(X_scaled, hc_labels)

        self.is_fitted = True

        print("\n" + "=" * 60)
        print("NetSentinel Clustering Evaluation Summary")
        print("=" * 60)
        print(f"K-Means (k={self.n_clusters}):")
        print(f"  Silhouette Score:         {km_sil:.4f}")
        print(f"  Davies-Bouldin Index:     {km_db:.4f}")
        print(f"  Calinski-Harabasz Index:  {km_ch:.1f}")
        print(f"Hierarchical (k={self.n_clusters}, Ward):")
        print(f"  Silhouette Score:         {hc_sil:.4f}")
        print(f"  Davies-Bouldin Index:     {hc_db:.4f}")
        print(f"  Calinski-Harabasz Index:  {hc_ch:.1f}")
        print("=" * 60)

        return {
            "kmeans": {"silhouette": km_sil, "davies_bouldin": km_db, "calinski_harabasz": km_ch},
            "hierarchical": {"silhouette": hc_sil, "davies_bouldin": hc_db, "calinski_harabasz": hc_ch}
        }


if __name__ == "__main__":
    clusterer = NetSentinelClusterer()
    clusterer.fit()
