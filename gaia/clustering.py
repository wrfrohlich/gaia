
import csv
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from os import path, makedirs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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

        metrics_list = []

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
                plt.title(f"Clusters of {item} data after PCA")
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.savefig(f'{self.path_experiment}/clusters_{item}.png')
                plt.clf()

                silhouette_avg = silhouette_score(imu_pca, clusters)
                calinski_harabasz = calinski_harabasz_score(imu_pca, clusters)
                davies_bouldin = davies_bouldin_score(imu_pca, clusters)

                logging.info(f'Silhouette Score for {item}: {silhouette_avg:.2f}')
                logging.info(f'Calinski-Harabasz Score for {item}: {calinski_harabasz:.2f}')
                logging.info(f'Davies-Bouldin Index for {item}: {davies_bouldin:.2f}')

                metrics_list.append({
                    'Variable Pair': item,
                    'Silhouette Score': silhouette_avg,
                    'Calinski-Harabasz Score': calinski_harabasz,
                    'Davies-Bouldin Index': davies_bouldin
                })

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(f'{self.path_experiment}/clustering_metrics.csv', index=False)

    def run_clustering_kmeans_grouped(self, data, method="pca", n_clusters=3):
        correlation_report = pd.read_csv(f'{self.path_experiment}/cross_correlation_report.csv', sep=";")
        correlation_data = pd.read_csv(f'{self.path_experiment}/cross_correlation_results.csv')

        metrics_list = []

        for _, row in correlation_report.iterrows():
            cleaned_list = []
            values = []
            for col in row:
                values.append(col)
            cleaned_list = [x for x in values if  x == x]
            shifted_data = data[cleaned_list].copy()

            item = f"{values[0]}_{method}"
            corrs = correlation_data[correlation_data["kinematic"] == values[0]]

            if "shift" in method:
                for imu in cleaned_list[1:]:
                    lag = corrs[corrs["imu"] == imu].lag.item()
                    correlation_data
                    if lag != 0:
                        shifted_data[imu] = shifted_data[imu].shift(-lag)
            
            shifted_data = shifted_data.dropna()

            if shifted_data.shape[0] > 1:
                if "pca" in method:
                    pca = PCA(n_components=len(cleaned_list))
                    imu_pca = pca.fit_transform(shifted_data)
                elif "tsne" in method:
                    tsne = TSNE(n_components=(cleaned_list), perplexity=30, n_iter=300, random_state=42)
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
                calinski_harabasz = calinski_harabasz_score(imu_pca, clusters)
                davies_bouldin = davies_bouldin_score(imu_pca, clusters)

                logging.info(f'Silhouette Score for {item}: {silhouette_avg:.2f}')
                logging.info(f'Calinski-Harabasz Score for {item}: {calinski_harabasz:.2f}')
                logging.info(f'Davies-Bouldin Index for {item}: {davies_bouldin:.2f}')

                metrics_list.append({
                    'Variable Pair': item,
                    'Silhouette Score': silhouette_avg,
                    'Calinski-Harabasz Score': calinski_harabasz,
                    'Davies-Bouldin Index': davies_bouldin
                })

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(f'{self.path_experiment}/clustering_metrics.csv', index=False)

    def analyze_cross_correlation(self):
        """
        Analyzes correlations from a CSV file and generates a report.
        """
        self.important_points = [
            "c7_x", "c7_y", "c7_z",
            "r_should_x", "r_should_y", "r_should_z", "l_should_x", "l_should_y", "l_should_z",
            "sacrum_s_x", "sacrum_s_y", "sacrum_s_z",
            "r_asis_x", "r_asis_y", "r_asis_z", "l_asis_x", "l_asis_y", "l_asis_z",
            "MIDASIS_x", "MIDASIS_y", "MIDASIS_z",
            "r_knee_1_x", "r_knee_1_y", "r_knee_1_z", "l_knee_1_x", "l_knee_1_y", "l_knee_1_z",
            "r_mall_x", "r_mall_y", "r_mall_z", "l_mall_x", "l_mall_y", "l_mall_z",
            "r_heel_x", "r_heel_y", "r_heel_z", "l_heel_x", "l_heel_y", "l_heel_z",
            "r_met_x", "r_met_y", "r_met_z", "l_met_x", "l_met_y", "l_met_z",
            "SHO_x", "SHO_y", "SHO_z",
            "PO_x", "PO_y", "PO_z",
            'r_force_x', 'r_force_y', 'r_force_z',
            'l_force_x', 'l_force_y', 'l_force_z'
        ]

        data = pd.read_csv(f'{self.path_experiment}/cross_correlation_results.csv')

        if 'imu' not in data.columns or 'kinematic' not in data.columns:
            raise ValueError("The CSV file does not contain the expected 'imu' and 'kinematic' columns.")

        correlation_report = {}
        for kinematic in self.important_points:
            filtered_df = data[data["kinematic"] == kinematic]
            filtered_df = filtered_df.sort_values(by=["imu"]).reset_index(drop=True)

            if not filtered_df.empty:
                correlation_report[kinematic] = filtered_df["imu"].tolist()

        if correlation_report:
            df_report = pd.DataFrame.from_dict(correlation_report, orient='index')
            df_report = df_report.fillna("NaN")
            output_path = f'{self.path_experiment}/cross_correlation_report.csv'
            df_report.to_csv(output_path, sep=';', index=True, header=True)

            print(f"Correlation report exported to: {output_path}")
            return df_report
        else:
            print("No data available for the correlation report.")
            return pd.DataFrame()
