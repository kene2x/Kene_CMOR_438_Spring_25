import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

__all__ = ["KMeans"]

class KMeans:
	def __init__(self, num_clusters=3, max_iterations=100):
		self.num_clusters = num_clusters
		self.max_iterations = max_iterations
		self.centroids = None
		
	def _euclidean_distance(self, a, b):
		return np.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))
	
	def _closest_centroid(self, sample):
		dists = np.array([self._euclidean_distance(sample, centroid) for centroid in self.centroids])
		return np.argmin(dists)
	
	def _get_labels(self, data):
		return [self._closest_centroid(d) for d in data]
	
	def _compute_centroids(self, data, labels):
		dims = data.shape[1]
		updated_centroids = np.zeros((self.num_clusters, dims))
		label_counts = np.zeros(self.num_clusters)
		
		for idx, sample in enumerate(data):
			group = labels[idx]
			updated_centroids[group] += sample
			label_counts[group] += 1
		
		label_counts = np.where(label_counts == 0, 1, label_counts)
		updated_centroids = updated_centroids / label_counts[:, np.newaxis]
		return updated_centroids
	
	def fit(self, data):
		start_idxs = np.random.choice(data.shape[0], self.num_clusters, replace=False)
		self.centroids = data[start_idxs]
		
		for _ in range(self.max_iterations):
			cluster_labels = self._get_labels(data)
			new_centroids = self._compute_centroids(data, cluster_labels)
			
			if np.allclose(self.centroids, new_centroids):
				break
			
			self.centroids = new_centroids
		
		return self._get_labels(data)
	
	def predict(self, samples):
		return self._get_labels(samples)