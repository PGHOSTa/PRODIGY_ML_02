import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace 'your_dataset.csv' with your file name)
file_path = 'C:/Users/DELL/Desktop/prodigy/archive/Mall_Customers.csv' 
data = pd.read_csv(file_path)

# Inspect the dataset
print("Dataset Overview:")
print(data.head())
print("\nColumns in Dataset:")
print(data.columns)

sns.pairplot(data)

# Select numerical features for clustering (exclude non-numeric columns like IDs)
numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nSelected Numerical Features: {numerical_features}")

# Standardize the selected features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numerical_features])

# Determine the optimal number of clusters using the elbow method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Apply K-Means with the optimal number of clusters (e.g., k=3)
optimal_k = int(input("Enter the optimal number of clusters (k) based on the elbow plot: "))
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualize the first two features (if at least two exist)
if len(numerical_features) >= 2:
    plt.figure(figsize=(8, 5))
    for cluster in range(optimal_k):
        cluster_data = data[data['Cluster'] == cluster]
        plt.scatter(cluster_data[numerical_features[0]], cluster_data[numerical_features[1]], label=f'Cluster {cluster}')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel(numerical_features[0])
    plt.ylabel(numerical_features[1])
    plt.legend()
    plt.show()
else:
    print("Not enough features for a 2D visualization.")

# Save the dataset with cluster labels
output_file = 'clustered_data.csv'
data.to_csv(output_file, index=False)
print(f"Clustered dataset saved to {output_file}")
