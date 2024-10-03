import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime


# Custom KMeans algorithm
def custom_kmeans(data, n_clusters, max_iterations=300, tolerance=1e-4):
    # Initialize centroids randomly from the data points
    np.random.seed(0)
    centroids = data[np.random.choice(data.shape[0], n_clusters, replace=False)]

    for i in range(max_iterations):
        # Assign data points to the nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=-1)
        clusters = np.argmin(distances, axis=1)

        # Calculate new centroids based on the current clusters
        new_centroids = np.array([data[clusters == j].mean(axis=0) for j in range(n_clusters)])

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) <= tolerance):
            break

        centroids = new_centroids

    return clusters, centroids


# Load the dataset
df = pd.read_csv("E:/Space/Neen/4/ثاني/animals.csv")

# Check the expiration date if available
if 'expiration_date' in df.columns:
    # Read the expiration date column
    expiration_date = df['expiration_date'].iloc[0]
    # Convert the expiration date to a date object
    expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
    # Get the current date
    current_date = datetime.now()
    # Check the expiration date
    if expiration_date < current_date:
        print("Warning: The data is expired. Please update the dataset.")
    else:
        print("The data is valid and not expired.")
else:
    print("No expiration date found in the dataset.")


# Feature Engineering
# Calculate Weight to Speed Ratio
df['Weight_to_Speed_Ratio'] = df['Weight (kg)'] / df['Top Speed (km/h)']
# Encode Social Behavior
social_behavior_mapping = {'Solitary': 0, 'Semi-social': 1, 'Social': 2}
df['Social_Behavior_Score'] = df['Social Behavior'].map(social_behavior_mapping)
# Encode Diet
diet_mapping = {'Herbivore': 0, 'Omnivore': 1, 'Carnivore': 2}
df['Diet_Score'] = df['Diet'].map(diet_mapping)
# Encode Habitat
habitat_mapping = {'Grasslands': 0, 'Forests': 1, 'Deserts': 2, 'Water': 3, 'Arctic': 4}
df['Habitat_Score'] = df['Habitat'].map(habitat_mapping)


# Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns


# Impute missing values for numeric columns
imputer = SimpleImputer(strategy='mean')
df_imputed_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)


# Combine the imputed numeric columns with the non-numeric columns
df_imputed = pd.concat([df_imputed_numeric, df[non_numeric_cols]], axis=1)


# Define numerical and categorical columns
numerical_cols = df_imputed.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df_imputed.select_dtypes(include=['object']).columns


# Create transformers for numerical and categorical columns
numerical_transformer = Pipeline([
    ('scaler', StandardScaler())  # Standardize numerical columns
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(drop='first'))  # Apply One-Hot Encoding to categorical columns
])


# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])


# Create the TruncatedSVD object
svd = TruncatedSVD(n_components=2)


# Create the full pipeline
data_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('svd', svd)
])


# Fit the pipeline to the data and transform the data
transformed_data = data_pipeline.fit_transform(df_imputed)


# Output the transformed data
print("Transformed data:")
print(transformed_data)


# Plot the explained variance ratio
print("Explained Variance Ratio:", svd.explained_variance_ratio_)


# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):  # Test 1 to 10 clusters
    clusters, _ = custom_kmeans(transformed_data, n_clusters=i)
    wcss.append(np.sum(np.min(np.linalg.norm(transformed_data[:, np.newaxis] - _, axis=2),
                              axis=1)))  # Append the WCSS for each number of clusters


# Plot the elbow graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.axvline(x=4, color='red', linestyle='--')  # Highlight k=4
plt.show()


# Perform K-means clustering with the optimal number of clusters
optimal_clusters = 4  # Adjust as needed
clusters, centroids = custom_kmeans(transformed_data, n_clusters=optimal_clusters)


# Cluster names
cluster_names = {
    0: ' Aquatic animals ',
    1: ' Mammals ',
    2: 'vertebrates',
    3: 'Predators'
}


# Add cluster labels to dataframe
df_imputed['Cluster'] = clusters


# Visualize the zoo layout based on clusters
plt.figure(figsize=(10, 8))
for cluster in range(optimal_clusters):
    plt.scatter(transformed_data[clusters == cluster, 0], transformed_data[clusters == cluster, 1],
                label=f'{cluster}: {cluster_names[cluster]}')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', label='Centroids')
plt.title('Zoo Layout Based on Clusters')
plt.xlabel('Truncated SVD Component 1')
plt.ylabel('Truncated SVD Component 2')
plt.legend()
plt.grid(True)
plt.show()


# Print the clusters
print("Clusters:")
for i in range(optimal_clusters):
    print(f"\nCluster {i} ({cluster_names[i]}):")
    print(df_imputed[df_imputed['Cluster'] == i])


# Save the dataframe with cluster labels to a new CSV file
df_imputed.to_csv('animals_with_clusters.csv', index=False)

print("\nThe clusters have been labeled and the updated dataframe has been saved to 'animals_with_clusters.csv'.")
