
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
    """
    A class for computing and visualizing clustering results from IMU data,
    and correlating these clusters with kinematic data.
    
    Example:
        clustering = Clustering('experiment_name')
        clusters = clustering.run_kmeans(data)
        clustering.correlate_clusters_with_kinematics(kinematic_data, clusters)
    """

    def __init__(self, name):
        """
        Initializes the Clustering class with the configuration settings,
        and sets up directories for saving plots.

        Parameters:
        - name: str, the name of the experiment for saving figures.
        """
        config = Config()
        self.path_experiment = config.figures
        self.points = config.body_parts

        # Set up a directory for the current experiment's plots
        if name:
            self.path_experiment = path.join(self.path_experiment, name)
            if not path.exists(self.path_experiment):
                makedirs(self.path_experiment)
        
        logging.basicConfig(level=logging.INFO)

    def correlate_clusters_with_kinematics(self, data_kinematics, clusters):
        """
        Correlates the clusters obtained from IMU data with kinematic data.

        Parameters:
        - data_kinematics: DataFrame, the kinematic data to be analyzed.
        - clusters: array-like, the cluster labels for each observation.
        """
        data_kinematics['cluster'] = clusters

        grouped_kinematics = data_kinematics.groupby('cluster').agg(['mean', 'std'])

        logging.info("Means and standard deviations of kinematic variables by cluster:")
        logging.info(grouped_kinematics)

        for var in data_kinematics.columns[:-1]:  # Exclude the 'cluster' column
            plt.figure()
            sns.histplot(data=data_kinematics, x=var, hue='cluster', element="step", stat="density", common_norm=False)
            plt.title(f'Kinematic Variable: {var}')
            plt.xlabel(var)
            plt.ylabel('Density')
            plt.legend(title='Cluster')
            plt.savefig(f'{self.path_experiment}/clusters_{var}.png')
            plt.clf()

    def run_clustering_kmeans(self, data, method="pca", n_clusters=3, n_components=2):
        """
        Runs the K-Means clustering algorithm on IMU data,
        adjusts for lag, reduces dimensionality using PCA,
        and visualizes the clusters.

        Parameters:
        - data_file: str, path to the CSV file containing the variables, lags, and correlations.
        - n_clusters: int, the number of clusters to form.
        - n_components: int, number of principal components to keep.
        """
        correlation_data = pd.read_csv(f'{self.path_experiment}/cross_correlation_results.csv')

        for _, row in correlation_data.iterrows():
            var1 = row.get("imu")
            var2 = row.get("kinematic")
            lag = row.get("lag")

            item = f"{var1}_{var2}_{method}"

            # Aplica a defasagem (lag) nos dados
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
                plt.title(f"Clusters of {item} data after PCA")
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.savefig(f'{self.path_experiment}/clusters_{item}.png')
                plt.clf()

                silhouette_avg = silhouette_score(imu_pca, clusters)
                logging.info(f'Silhouette Score for {item}: {silhouette_avg:.2f}')
            else:
                logging.warning(f"Dados insuficientes após aplicação da defasagem para {item}.")
