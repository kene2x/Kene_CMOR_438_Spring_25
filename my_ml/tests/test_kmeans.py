import numpy as np
import pytest
from my_ml.k_means import KMeans

def test_kmeans_initialization():
    # Test default initialization
    kmeans = KMeans()
    assert kmeans.num_clusters == 3
    assert kmeans.max_iterations == 100
    assert kmeans.centroids is None
    
    # Test custom parameters
    kmeans = KMeans(num_clusters=5, max_iterations=50)
    assert kmeans.num_clusters == 5
    assert kmeans.max_iterations == 50

def test_distance_calculation():
    kmeans = KMeans()
    point = np.array([1, 1])
    center = np.array([4, 5])
    
    # Test Euclidean distance
    distance = kmeans._euclidean_distance(point, center)
    assert np.isclose(distance, 5.0)  # sqrt((4-1)^2 + (5-1)^2)

def test_assign_label():
    kmeans = KMeans(num_clusters=2)
    kmeans.centroids = np.array([[0, 0], [2, 2]])
    point = np.array([0.1, 0.1])
    
    # Point should be assigned to closest center
    label = kmeans._closest_centroid(point)
    assert label == 0  # Closer to [0, 0] than [2, 2]

def test_cluster_assignment():
    kmeans = KMeans(num_clusters=2)
    kmeans.centroids = np.array([[0, 0], [2, 2]])
    X = np.array([[0.1, 0.1], [1.9, 1.9]])
    
    # Test cluster assignments
    labels = kmeans._get_labels(X)
    assert len(labels) == 2
    assert labels[0] == 0  # Closer to first center
    assert labels[1] == 1  # Closer to second center

def test_center_updates():
    kmeans = KMeans(num_clusters=2)
    X = np.array([[0, 0], [0, 2], [2, 0], [2, 2]])
    labels = np.array([0, 0, 1, 1])
    
    # Test center updates
    new_centers = kmeans._compute_centroids(X, labels)
    assert new_centers.shape == (2, 2)
    assert np.allclose(new_centers[0], [0, 1])  # Center of first cluster
    assert np.allclose(new_centers[1], [2, 1])  # Center of second cluster

def test_fit_predict():
    # Create simple dataset with clear clusters
    X = np.array([[0, 0], [0.1, 0], [0, 0.1],  # Cluster 1
                  [2, 2], [2.1, 2], [2, 2.1]])  # Cluster 2
    
    kmeans = KMeans(num_clusters=2, max_iterations=10)
    labels = kmeans.fit(X)
    
    # Test basic properties
    assert len(labels) == len(X)
    assert len(np.unique(labels)) == 2
    assert kmeans.centroids is not None
    assert kmeans.centroids.shape == (2, 2)
    
    # Test prediction
    new_point = np.array([0.05, 0.05])
    pred = kmeans.predict(np.array([new_point]))[0]
    assert pred == labels[0]  # Should be assigned to same cluster as nearby points