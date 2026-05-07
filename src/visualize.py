import pandas as pd
import matplotlib.pyplot as plt

# Load clustered dataset
customers = pd.read_csv("data/customers_clustered.csv")

# Create scatter plot
plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    customers["AnnualIncome"],
    customers["SpendingScore"],
    c=customers["Cluster"],
    cmap="viridis"
)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")

plt.colorbar(scatter)

plt.savefig("assets/customer_clusters.png")

print("Cluster visualization saved successfully!")