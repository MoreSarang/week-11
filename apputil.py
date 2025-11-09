"""
apputil.py
----------
Implements helper functions for K-Means clustering exercises.
Follows PEP-8 style guidelines.
"""

from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------
# Global variable: numerical columns from seaborn's diamonds dataset
# ---------------------------------------------------------------------
diamonds = sns.load_dataset("diamonds")

# Select only numerical columns (typically 7 columns)
diamonds_numeric = diamonds.select_dtypes(include=[np.number])

# Make this dataframe global for reuse
DIAMONDS_NUMERIC = diamonds_numeric


# ---------------------------------------------------------------------
# Exercise 1: K-Means on any numerical NumPy array
# ---------------------------------------------------------------------
def kmeans(X, k):
    """
    Perform k-means clustering on a numerical NumPy array X.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    k : int
        Number of clusters

    Returns
    -------
    centroids : np.ndarray
        Cluster centroids of shape (k, n_features)
    labels : np.ndarray
        Cluster assignment for each sample, shape (n_samples,)
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X)

    centroids = model.cluster_centers_
    labels = model.labels_

    return centroids, labels


# ---------------------------------------------------------------------
# Exercise 2: K-Means on the first n rows of the diamonds dataset
# ---------------------------------------------------------------------
def kmeans_diamonds(n, k):
    """
    Run k-means clustering on the first n rows of the numeric
    columns in the seaborn diamonds dataset.

    Parameters
    ----------
    n : int
        Number of rows from the diamonds dataset to use
    k : int
        Number of clusters

    Returns
    -------
    centroids : np.ndarray
        Cluster centroids of shape (k, n_features)
    labels : np.ndarray
        Cluster assignment for each sample, shape (n_samples,)
    """
    data_subset = DIAMONDS_NUMERIC.head(n).to_numpy()
    centroids, labels = kmeans(data_subset, k)
    return centroids, labels


# ---------------------------------------------------------------------
# Exercise 3: Average runtime measurement for K-Means on diamonds
# ---------------------------------------------------------------------
def kmeans_timer(n, k, n_iter=5):
    """
    Measure the average runtime (in seconds) of running
    kmeans_diamonds(n, k) for n_iter times.

    Parameters
    ----------
    n : int
        Number of samples (rows)
    k : int
        Number of clusters
    n_iter : int, optional
        Number of repetitions (default=5)

    Returns
    -------
    float
        Average runtime across n_iter runs (in seconds)
    """
    times = []
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        elapsed = time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    return avg_time
# your code here
