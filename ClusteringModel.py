#from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from Build_PCA import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import pickle
import seaborn as sns
    

if __name__=="__main__":
    data = pd.read_csv("LUSCexpfile.csv", sep = ";")
    data.info()
    data = data.T
    data.columns = data.iloc[0]
    data = data[1:]
    data = data.rename(columns = {np.nan:'Class'})
    data.columns.name = None
    X = data.drop(columns=['Class'])    
    feature_names = X.columns.tolist()
    
    with open('./models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)


    # Define the threshold for dropping columns (e.g., 70% zeros)
    threshold = 0.7
    zero_proportion = (data == 0).mean()
    columns_to_drop = zero_proportion[zero_proportion > threshold].index
    data_cleaned1 = data.drop(columns=columns_to_drop)


    # Selecting features using random forest model
    y = data_cleaned1['Class']
    x = data_cleaned1.drop(columns=['Class'])
    model = RandomForestClassifier(random_state=33)
    model.fit(x, y)
    importances = model.feature_importances_
    importance_threshold = 0.01  
    features_to_drop = x.columns[importances < importance_threshold]
    data_cleaned2 = data_cleaned1.drop(columns=features_to_drop)


    data = data_cleaned2
    selected_features = data.drop(columns=['Class']).columns.tolist()
    selected_features

    with open('./models/selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)


    # collect the tumor samples
    tumor_Data = data[data['Class'] == 'tumor']
    scaler = StandardScaler()
    tumor_scaled = scaler.fit_transform(tumor_Data.drop("Class", axis = 1).values.astype(np.float32))
    
    # convert it back to dataframe
    tumor = pd.DataFrame(tumor_scaled, columns=tumor_Data.drop("Class", axis=1).columns)
    tumor = np.array(tumor)
    print(type(tumor))

    # Reduce dimension using PCA
    pca = PCA(n_components = 2)
    tumor_pca = pca.fit_transform(tumor)
    print(f"Before pca: {tumor.shape}")
    print(f"After PCA: {tumor_pca.shape}")

    joblib.dump(pca, './models/pca_model.pkl')


    # K-means Clustering
    sil_scores = []
    k_range = range(2, 10)  # Try clustering from 2 to 10 clusters
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(tumor_pca)
        sil_score = silhouette_score(tumor_pca, labels)
        sil_scores.append(sil_score)
    plt.plot(k_range, sil_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for K-means clustering')
    plt.show()


    #To find the appropriate k-value using elbow method
    sum_of_square_distance=[]
    for k in range(2,15):
        km = KMeans(n_clusters=k, init="k-means++", max_iter=100, random_state=42)
        km = km.fit(tumor_pca)
        sum_of_square_distance.append(km.inertia_)
    plt.figure(figsize=(7,4))
    plt.plot(range(2,15),sum_of_square_distance)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel("Sum of Squared Distance")
    plt.savefig('./plots/elbow_method_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    #K-means clustering
    kmeans = KMeans(n_clusters=2, init="k-means++", max_iter=200, random_state=42)
    cluster = kmeans.fit(tumor_pca)

    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(tumor_pca, cluster)
    print(f'Silhouette Score: {silhouette_avg}')

    from sklearn.metrics import davies_bouldin_score
    db_index = davies_bouldin_score(tumor_pca, cluster)
    print(f"Davies-Bouldin Index: {db_index}")

    with open("./models/clustering_metrics.txt", "w") as file:
        file.write(f"Silhouette Score: {silhouette_avg}\n")
        file.write(f"Davies-Bouldin Index: {db_index}\n")
    file.close()

    #visualize clustering
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=tumor_pca[:, 0], y=tumor_pca[:, 1], hue=cluster, palette='viridis')
    plt.title('K-Means Clustering Results')
    plt.savefig('./plots/clustering.png', dpi=300, bbox_inches='tight')
    plt.show()


    import joblib
    from sklearn.cluster import KMeans
    joblib.dump(cluster, './models/kmeans_model.pkl')

