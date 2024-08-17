import numpy as np
import pandas as pd
from os import path, makedirs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from gaia.config import Config
from sklearn.metrics import silhouette_score

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

    def run_kmeans(self, data, n_clusters=4, pca_components=3):
        """
        Runs the K-Means clustering algorithm on IMU data,
        reduces dimensionality using PCA, and visualizes the clusters.

        Parameters:
        - data: DataFrame, the input data containing IMU measurements.
        - n_clusters: int, the number of clusters to form.
        - pca_components: int, number of principal components to keep in PCA.
        """
        for item, imu_data in self.points.items():
            combined_imu_data = data[imu_data]

            if combined_imu_data.shape[1] > 1:
                try:
                    pca = PCA(n_components=pca_components)
                    imu_pca = pca.fit_transform(combined_imu_data)

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

                except Exception as e:
                    logging.error(f"Error in clustering {item}: {e}")
            else:
                logging.warning(f"Combined IMU data for {item} has only one column. PCA is not applicable.")
