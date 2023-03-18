import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load the data
data = pd.read_csv('data.csv')

# Normalize the data
data_norm = (data - data.mean()) / data.std()

# Choose the number of nearest neighbors to consider
k = 10

# Train the k-NN model
model = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data_norm)

# Find the k nearest neighbors for each data point
distances, indices = model.kneighbors(data_norm)

# Calculate the average distance to the k nearest neighbors for each data point
avg_distances = np.mean(distances, axis=1)

# Calculate the threshold for anomaly detection
threshold = np.percentile(avg_distances, 95)

# Identify anomalies
anomalies = data[avg_distances > threshold]

# Print the anomalies
print(anomalies)
