import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

#df = pd.read_csv(r'C:\Users\Yousuf Traders\Desktop\task 4\Mall_Customers.csv')
df = pd.read_csv('Mall_Customers.csv')

print(df.info())
print(df.describe())

#for working on numerical features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#find optimal clusters by using elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow method for optimal clusters')
plt.show()

#applying K-Means clustering
optimal_clusters = 5 
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

#applying hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=optimal_clusters, affinity='euclidean', linkage='ward')
df['Hierarchical_Cluster'] = hierarchical.fit_predict(X_scaled)

sns.scatterplot(x=df['Annual income'], y=df['Spending score '], hue=df['KMeans_Cluster'], palette='viridis')
plt.title('Segmentation using k-means algorithm')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.show()

#dendrogram for hierarchical clustering
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix)
plt.title('Dendrogram for hierarchical clustering')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

print(df.head())
