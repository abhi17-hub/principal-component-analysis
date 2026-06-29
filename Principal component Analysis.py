# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load built-in dataset
data = load_breast_cancer()

# Create DataFrame
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (2 Principal Components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# New DataFrame
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Target"] = y

print(pca_df.head())

# Explained Variance Ratio
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Scatter Plot
plt.figure(figsize=(8,6))

for target in [0, 1]:
    plt.scatter(
        pca_df[pca_df["Target"] == target]["PC1"],
        pca_df[pca_df["Target"] == target]["PC2"],
        label=data.target_names[target]
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Breast Cancer Dataset")
plt.legend()
plt.grid(True)

plt.show()


import pickle

with open(r"D:\Starting new\scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

with open(r"D:\Starting new\pca_model.pkl", "wb") as file:
    pickle.dump(pca, file)

print("Models saved successfully!")