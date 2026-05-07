import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ── 1. Load data ──────────────────────────────────────────────────────────────

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

data = pd.read_csv(os.path.join(BASE_DIR, "data", "Mall_Customers.csv"))
print(data.head())
print(data.info())
print("Missing values:\n", data.isnull().sum())

# ── 2. Features ───────────────────────────────────────────────────────────────

X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# ── 3. Elbow method — find optimal k ─────────────────────────────────────────

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, n_init=10, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "results", "elbow.png"), dpi=150)
plt.show()

# ── 4. Train final model (k=5 from elbow) ────────────────────────────────────

k_means = KMeans(n_clusters=5, n_init=10, random_state=42)
y_pred = k_means.fit_predict(X)

# ── 5. Visualise clusters ─────────────────────────────────────────────────────

colors = ['red', 'blue', 'yellow', 'green', 'orange']
labels = [f'Cluster {i+1}' for i in range(5)]

plt.figure(figsize=(9, 6))
for i in range(5):
    plt.scatter(
        X.iloc[y_pred == i, 0],
        X.iloc[y_pred == i, 1],
        s=80, c=colors[i], label=labels[i]
    )
plt.scatter(
    k_means.cluster_centers_[:, 0],
    k_means.cluster_centers_[:, 1],
    s=200, c='black', marker='X', label='Centroids'
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Segments")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "results", "clusters.png"), dpi=150)
plt.show()

# ── 6. Test prediction ────────────────────────────────────────────────────────

test_cases = [[80, 90], [40, 10], [60, 50]]
for tc in test_cases:
    cluster = k_means.predict([tc])[0]
    print(f"Income={tc[0]}k$, Spending={tc[1]} → Cluster {cluster + 1}")

# ── 7. Save model ─────────────────────────────────────────────────────────────

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
model_path = os.path.join(BASE_DIR, "models", "kmeans_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(k_means, f)

print(f"\nModel saved → {model_path}")
