import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def agglomerative_clustering(X, n_clusters):
    distances = squareform(pdist(X))
    np.fill_diagonal(distances, np.inf)
    
    clusters = {i: [i] for i in range(len(X))}
    
    while len(clusters) > n_clusters:
        min_dist = np.inf
        to_merge = None
        for i in clusters:
            for j in clusters:
                if i < j:
                    dist = np.min([distances[p1, p2] for p1 in clusters[i] for p2 in clusters[j]])
                    if dist < min_dist:
                        min_dist = dist
                        to_merge = (i, j)
        
        clusters[to_merge[0]].extend(clusters[to_merge[1]])
        del clusters[to_merge[1]]
        
        for i in clusters:
            if i != to_merge[0]:
                dists = [distances[p1, p2] for p1 in clusters[to_merge[0]] for p2 in clusters[i]]
                new_dist = np.min(dists)
                for p1 in clusters[to_merge[0]]:
                    for p2 in clusters[i]:
                        distances[p1, p2] = new_dist
                        distances[p2, p1] = new_dist
    
    labels = np.zeros(len(X), dtype=int)
    for cluster_id, points in clusters.items():
        for point in points:
            labels[point] = cluster_id
    
    return labels

# Cargar datos sin encabezado
video_features_df = pd.read_csv('video_features_id.csv', header=None)

# Asumir que la primera columna es youtube_id y el resto son características
column_names = ['youtube_id'] + [f'feature_{i}' for i in range(1, video_features_df.shape[1])]
video_features_df.columns = column_names

# Limpiar valores NaN
video_features_df = video_features_df.dropna()

# Cargar archivos de entrenamiento, prueba y validación
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
val_df = pd.read_csv('val.csv')
sample_submission_df = pd.read_csv('sample_submission.csv')

# Unir features con etiquetas del conjunto de entrenamiento
train_features_df = video_features_df.merge(train_df, on='youtube_id')

# Separar features y etiquetas
X_train = train_features_df.drop(columns=['youtube_id', 'label'])
y_train = train_features_df['label']

# Reducción de dimensionalidad usando PCA
pca = PCA(n_components=50)  # Reducir a 50 componentes principales
X_pca = pca.fit_transform(X_train)

# Reducción de dimensionalidad usando UMAP
umap = UMAP(n_components=2)
X_umap = umap.fit_transform(X_train)

# Clustering usando AgglomerativeClustering de sklearn
agglo_sklearn = AgglomerativeClustering(n_clusters=20)
agglo_labels_sklearn = agglo_sklearn.fit_predict(X_pca)

# Clustering usando el algoritmo aglomerativo implementado
agglo_labels_custom = agglomerative_clustering(X_pca, n_clusters=20)

# Comparar los resultados
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agglo_labels_sklearn, cmap='viridis', s=5)
plt.title('Agglomerative Clustering (sklearn)')
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agglo_labels_custom, cmap='viridis', s=5)
plt.title('Agglomerative Clustering (custom)')
plt.show()
