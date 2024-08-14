import numpy as np
import pandas as pd
from os import path, makedirs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from .config import Config

class Clustering:
    """
    A class for computing and visualizing clustering results from IMU data,
    and correlating these clusters with kinematic data.
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

    def correlate_clusters_with_kinematics(self, data_kinematics, clusters):
        """
        Correlates the clusters obtained from IMU data with kinematic data.

        Parameters:
        - data_kinematics: DataFrame, the kinematic data to be analyzed.
        - clusters: array-like, the cluster labels for each observation.

        This method adds the cluster labels to the kinematic data,
        computes the mean and standard deviation of the kinematic variables
        for each cluster, and visualizes the distribution of each kinematic
        variable across clusters.
        """
        # Add clusters to the corresponding kinematic data
        data_kinematics['cluster'] = clusters

        # Calculate mean and standard deviation for kinematic variables in each cluster
        grouped_kinematics = data_kinematics.groupby('cluster').mean()
        grouped_kinematics_std = data_kinematics.groupby('cluster').std()

        print("Means of kinematic variables by cluster:")
        print(grouped_kinematics)

        print("\nStandard deviations of kinematic variables by cluster:")
        print(grouped_kinematics_std)

        # Compare clusters with kinematic variables
        for var in data_kinematics.columns[:-1]:  # Exclude the 'cluster' column
            plt.figure()
            plt.title(f'Kinematic Variable: {var}')
            for cluster in sorted(data_kinematics['cluster'].unique()):
                plt.hist(data_kinematics[data_kinematics['cluster'] == cluster][var], 
                         alpha=0.5, label=f'Cluster {cluster}')
            plt.legend()
            plt.xlabel(var)
            plt.ylabel('Frequency')
            plt.savefig(f'{self.path_experiment}/clusters_bluba_{var}.png')
            plt.clf()

    def run_kmeans(self, data):
        """
        Runs the K-Means clustering algorithm on IMU data,
        reduces dimensionality using PCA, and visualizes the clusters.

        Parameters:
        - data: DataFrame, the input data containing IMU measurements.

        This method iterates over the body parts defined in the configuration,
        performs PCA on the IMU data, applies K-Means clustering, and saves
        the cluster visualizations. It also calculates the Silhouette Score for
        each clustering result.
        """
        for item in self.points:
            imu_data = self.points[item]

            # Combine all relevant IMU columns into a single DataFrame
            combined_imu_data = data[imu_data]

            # Check if combined data has more than one column
            if combined_imu_data.shape[1] > 1:
                # Apply PCA to reduce dimensionality
                pca = PCA(n_components=3)
                imu_pca = pca.fit_transform(combined_imu_data)

                # Apply K-Means clustering
                kmeans = KMeans(n_clusters=4, random_state=42)
                clusters = kmeans.fit_predict(imu_pca)

                print(clusters)

                # Visualize clusters (after PCA)
                plt.scatter(imu_pca[:, 0], imu_pca[:, 1], c=clusters, cmap='viridis')
                plt.title(f"Clusters of {item} data after PCA")
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.savefig(f'{self.path_experiment}/clusters_combined_{item}.png')
                plt.clf()

                # Calculate the Silhouette Score
                from sklearn.metrics import silhouette_score
                silhouette_avg = silhouette_score(imu_pca, clusters)
                print(f'Silhouette Score for {item}: {silhouette_avg:.2f}')
            else:
                print("Combined IMU data has only one column. PCA is not applicable.")

            # Uncomment the line below to correlate clusters with kinematics
            # if item == "imu":
            #     self.correlate_clusters_with_kinematics(data, clusters)
