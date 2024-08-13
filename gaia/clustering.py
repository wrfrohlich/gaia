import numpy as np
import pandas as pd
from os import path, makedirs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from .config import Config

class Clustering:
    """
    A class for computing and visualizing various correlation metrics between data series.
    """

    def __init__(self, name):
        """
        Initializes the Correlation class with Config and sets up directories for saving plots.
        """
        config = Config()
        self.path_experiment = config.figures
        self.points = config.body_parts

        if name:
            self.path_experiment = path.join(self.path_experiment, name)
            if not path.exists(self.path_experiment):
                makedirs(self.path_experiment)

    def correlate_clusters_with_kinematics(self, data_kinematics, clusters):
        # Adiciona os clusters aos dados cinemáticos correspondentes
        data_kinematics['cluster'] = clusters

        # Calcula a média e a variância para as variáveis cinemáticas em cada cluster
        grouped_kinematics = data_kinematics.groupby('cluster').mean()
        grouped_kinematics_std = data_kinematics.groupby('cluster').std()

        print("Médias das variáveis cinemáticas por cluster:")
        print(grouped_kinematics)

        print("\nDesvios padrão das variáveis cinemáticas por cluster:")
        print(grouped_kinematics_std)

        # Comparação dos clusters com as variáveis cinemáticas
        for var in data_kinematics.columns[:-1]:  # Excluindo a coluna 'cluster'
            plt.figure()
            plt.title(f'Variável Cinemática: {var}')
            for cluster in sorted(data_kinematics['cluster'].unique()):
                plt.hist(data_kinematics[data_kinematics['cluster'] == cluster][var], 
                        alpha=0.5, label=f'Cluster {cluster}')
            plt.legend()
            plt.xlabel(var)
            plt.ylabel('Frequência')
            plt.savefig(f'{self.path_experiment}/clusters_bluba_{var}.png')
            plt.clf()

    def run_kmeans(self, data):
        for item in self.points:
            imu_data = self.points[item]

            # Combine todas as colunas IMU relevantes em um único DataFrame
            combined_imu_data = data[imu_data]

            # Verifique se os dados combinados têm mais de uma coluna
            if combined_imu_data.shape[1] > 1:
                # Aplicar PCA para reduzir a dimensionalidade
                pca = PCA(n_components=3)
                imu_pca = pca.fit_transform(combined_imu_data)

                # Aplicar K-Means
                kmeans = KMeans(n_clusters=4, random_state=42)
                clusters = kmeans.fit_predict(imu_pca)

                print(clusters)

                # Visualizar os clusters (após PCA)
                plt.scatter(imu_pca[:, 0], imu_pca[:, 1], c=clusters, cmap='viridis')
                plt.title(f"Clusters dos dados {item} após PCA")
                plt.xlabel("Componente Principal 1")
                plt.ylabel("Componente Principal 2")
                plt.savefig(f'{self.path_experiment}/clusters_combined_{item}.png')
                plt.clf()


                from sklearn.metrics import silhouette_score

                # Calcular a Silhouette Score
                silhouette_avg = silhouette_score(imu_pca, clusters)
                print(f'Silhouette Score: {silhouette_avg:.2f}')
            else:
                print("Os dados IMU combinados têm apenas uma coluna. PCA não é aplicável.")

            if item == "imu":
                self.correlate_clusters_with_kinematics(data, clusters)



