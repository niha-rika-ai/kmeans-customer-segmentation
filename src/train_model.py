import pandas as pd
import joblib
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
customers = pd.read_csv("data/customers.csv")

# Features for clustering
X = customers[["AnnualIncome", "SpendingScore", "PurchaseFrequency", "PurchaseAmount"]]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Cluster labels
customers["Cluster"] = kmeans.labels_
print(customers.groupby("Cluster").mean())

# Evaluate model
score = silhouette_score(X_scaled, kmeans.labels_)
print("Silhouette Score:", round(score, 2))

# Save clustered dataset
customers.to_csv("data/customers_clustered.csv", index=False)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(kmeans, "model/kmeans_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("K-Means model trained successfully!")