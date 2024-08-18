
import csv
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from os import path, makedirs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from gaia.config import Config


class Clustering:
    def __init__(self, name):
        config = Config()
        self.path_experiment = config.experiments
        self.points = config.body_parts

        if name:
            self.path_experiment = path.join(self.path_experiment, name)
            if not path.exists(self.path_experiment):
                makedirs(self.path_experiment)
        
        logging.basicConfig(level=logging.INFO)
        
        # Initialize CSV file for metrics
        self.csv_file = path.join(self.path_experiment, "cluster_metrics.csv")
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Item", "Method", "Num Clusters", "Silhouette Score"])

    def correlate_clusters_with_kinematics(self, data_kinematics, clusters):
        data_kinematics['cluster'] = clusters

        grouped_kinematics = data_kinematics.groupby('cluster').agg(['mean', 'std'])

        logging.info("Means and standard deviations of kinematic variables by cluster:")
        logging.info(grouped_kinematics)

        for var in data_kinematics.columns[:-1]:
            plt.figure()
            sns.histplot(data=data_kinematics, x=var, hue='cluster', element="step", stat="density", common_norm=False)
            plt.title(f'Kinematic Variable: {var}')
            plt.xlabel(var)
            plt.ylabel('Density')
            plt.legend(title='Cluster')
            plt.savefig(f'{self.path_experiment}/clusters_{var}.png')
            plt.clf()

    def run_clustering_kmeans(self, data, method="pca", n_clusters=3, n_components=2):
        correlation_data = pd.read_csv(f'{self.path_experiment}/cross_correlation_results.csv')

        for _, row in correlation_data.iterrows():
            var1 = row.get("imu")
            var2 = row.get("kinematic")
            lag = row.get("lag")

            item = f"{var1}_{var2}_{method}"

            lag = int(lag)
            shifted_data = data[[var1, var2]].copy()

            if "shift" in method:
                if lag < 0:
                    shifted_data[var2] = shifted_data[var2].shift(-lag)
                else:
                    shifted_data[var1] = shifted_data[var1].shift(lag)
            
            shifted_data = shifted_data.dropna()

            if shifted_data.shape[0] > 1:
                if "pca" in method:
                    pca = PCA(n_components=n_components)
                    imu_pca = pca.fit_transform(shifted_data)
                elif "tsne" in method:
                    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
                    imu_pca = tsne.fit_transform(shifted_data)

                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(imu_pca)

                logging.info(f'Clusters for {item}: {clusters}')

                plt.figure()
                sns.scatterplot(x=imu_pca[:, 0], y=imu_pca[:, 1], hue=clusters, palette='viridis')
                plt.title(f"Clusters of {item} data after {method.upper()}")
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.savefig(f'{self.path_experiment}/clusters_{item}.png')
                plt.clf()

                silhouette_avg = silhouette_score(imu_pca, clusters)
                logging.info(f'Silhouette Score for {item}: {silhouette_avg:.2f}')

                with open(self.csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([item, method, n_clusters, silhouette_avg])
