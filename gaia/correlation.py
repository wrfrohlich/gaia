import os
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from scipy.signal import correlate
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.stats.inter_rater as irr
from .config import Config

class Correlation:
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
                os.makedirs(self.path_experiment)

    def get_higher_corr(self, data, level=0.7, method="pearson"):
        """
        Computes and saves high correlation values between body points and IMUs.
        """
        try:
            level = float(level)
        except ValueError:
            raise ValueError(f"Level should be a number, but got {level}")

        correlations_list = []

        for points in self.points:
            if points == "imu":
                continue
            columns = list(self.points[points]) + self.points.get("imu", [])
            missing_cols = [col for col in columns if col not in data.columns]
            if missing_cols:
                continue
            
            df = data[columns]
            df = df.loc[:, ~df.columns.duplicated()]
            corr_matrix = df.corr(method=method)
            
            for x in corr_matrix.columns:
                if x in self.points.get("imu", []):
                    for y in corr_matrix.index:
                        if y not in self.points.get("imu", []):
                            correlation_value = corr_matrix.loc[y, x]
                            if abs(correlation_value) > level and x != y:
                                correlations_list.append({
                                    'group': points,
                                    'imu': x,
                                    'point': y,
                                    'corr': correlation_value
                                })

        correlations_df = pd.DataFrame(correlations_list)
        correlations_df = correlations_df.sort_values(by=["imu", "corr"], ascending=[True, False]).reset_index(drop=True)
        correlations_df.to_csv(f'{self.path_experiment}/correlations.csv', index=False)

        return correlations_df

    def analyze_correlation(self):
        """
        Analyzes correlations from a CSV file and generates a report.
        """
        imus = self.points.get("imu", [])
        data = pd.read_csv(f'{self.path_experiment}/correlations.csv')

        if 'imu' not in data.columns or 'point' not in data.columns:
            raise ValueError("The CSV file does not contain the expected 'imu' and 'point' columns.")

        correlation_report = {}
        for imu in imus:
            filtered_df = data[data["imu"] == imu]
            filtered_df = filtered_df.sort_values(by=["point"]).reset_index(drop=True)

            if not filtered_df.empty:
                correlation_report[imu] = filtered_df["point"].tolist()

        if correlation_report:
            df_report = pd.DataFrame.from_dict(correlation_report, orient='index')
            df_report = df_report.fillna("")
            output_path = f'{self.path_experiment}/correlation_report.csv'
            df_report.to_csv(output_path, sep=';', index=True, header=True)

            print(f"Correlation report exported to: {output_path}")
            return df_report
        else:
            print("No data available for the correlation report.")
            return pd.DataFrame()

    def corr_matrix(self, data):
        """
        Computes and saves Pearson, Spearman, or Kendall correlation matrices for the given data.
        """
        for points in self.points:
            columns = list(self.points[points]) + self.points.get("imu", [])
            missing_cols = [col for col in columns if col not in data.columns]
            if missing_cols:
                continue
            df = data[columns]
            self.gen_corr_matrix(df, name=points)

    def corr_matrix_special(self, data, name=""):
        """
        Computes and saves Pearson, Spearman, and Kendall correlation matrices for the given data.
        """
        self.gen_corr_matrix(data, name=name)

    def gen_corr_matrix(self, data, name, method="pearson"):
        """
        Generates and saves correlation matrices (Pearson, Spearman, Kendall) for the given data.
        """
        corr_matrix = data.corr(method=method)
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title(f'correlation_matrix_{method}_{name}')
        plt.savefig(f'{self.path_experiment}/matrix_{name}.png')
        plt.clf()

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
        plt.title(f'triangular_correlation_{method}_{name}')
        plt.savefig(f'{self.path_experiment}/trig_matrix_{name}.png')
        plt.clf()

    def cross_correlation(self, data, report):
        """
        Computes and plots cross-correlation for all pairs of columns in the DataFrame, excluding the 'time' column.
        """
        for idx in report:
            for imu in report.index:
                if report[idx][imu]:
                    self.print_cross_corr(data, imu, report[idx][imu])

    def print_cross_corr(self, df, value_a, value_b, max_lag=100):
        """
        Computes and plots cross-correlation between two data series with more control.
        """
        corr, lags = self.calculate_cross_correlation(df[value_a], df[value_b])
        
        # Limitar o número de lags para os valores especificados
        limited_lags = (lags >= -max_lag) & (lags <= max_lag)
        lags = lags[limited_lags]
        corr = corr[limited_lags]
        plt.figure(figsize=(10, 6))
        plt.plot(lags, corr, marker='o')
        plt.title(f'Cross-Correlation: {value_a} vs {value_b}')
        plt.xlabel('Lags')
        plt.ylabel('Correlation')
        plt.grid(True)
        plt.axhline(0, color='black', lw=1)
        plt.savefig(f'{self.path_experiment}/cross_corr_{value_a}_vs_{value_b}.png')
        plt.clf()

    def calculate_cross_correlation(self, series_a, series_b):
        """
        Calculates the cross-correlation between two series for a given lag.
        """
        series_a = series_a - np.mean(series_a)
        series_b = series_b - np.mean(series_b)
        corr = correlate(series_a, series_b, mode='full')
        corr /= (np.std(series_a) * np.std(series_b) * len(series_a))
        lags = np.arange(-(len(series_a) - 1), len(series_a))
        return corr, lags

    def find_best_cross_correlation_lag(self, series_a, series_b, max_lag=100):
        """
        Finds the best lag (within a range) that gives the highest cross-correlation between two series.
        """
        best_lag = 0
        best_corr = 0

        corr, lags = self.calculate_cross_correlation(series_a, series_b)

        for i, lag in enumerate(lags):
            if -max_lag <= lag <= max_lag:
                current_corr = corr[i]
                if abs(current_corr) > abs(best_corr):
                    best_corr = current_corr
                    best_lag = lag

        return best_lag, best_corr

    def cross_correlation_analysis(self, data, report):
        """
        Analyzes cross-correlation for all pairs of series, finds the best lag for each pair, and saves the results in a CSV.
        """
        results = []
        for idx in report:
            for imu in report.index:
                if report[idx][imu]:
                    best_lag, best_corr = self.find_best_cross_correlation_lag(data[imu], data[report[idx][imu]])
                    results.append({
                        'imu': imu,
                        'kinematic': report[idx][imu],
                        'lag': best_lag,
                        'corr': best_corr
                    })
                    self.print_cross_corr(data, imu, report[idx][imu])
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=["imu", "corr"], ascending=[True, False]).reset_index(drop=True)
        results_df.to_csv(f'{self.path_experiment}/cross_correlation_results.csv', index=False)

        return results_df

    def cross_correlation_exploratory(self, data, criterion=0.7, best=False, max_lag=100):
        """
        Analyzes cross-correlation for all pairs of series, finds the best lag for each pair, and saves the results in a CSV.
        """
        requirement = criterion if best else 0.0
        results = []
        for imu in self.points.get("imu", []):
            for kinematic in data.columns:
                if kinematic in self.points.get("imu", []):
                    continue
                best_lag, best_corr = self.find_best_cross_correlation_lag(data[imu], data[kinematic])
                if abs(best_corr) >= requirement and abs(best_lag) < max_lag:
                    if abs(best_corr) >= criterion:
                        self.print_cross_corr(data, imu, kinematic)
                    results.append({
                        'imu': imu,
                        'kinematic': kinematic,
                        'lag': best_lag,
                        'corr': best_corr
                    })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=["imu", "corr"], ascending=[True, False]).reset_index(drop=True)
        results_df.to_csv(f'{self.path_experiment}/cross_correlation_exploratory_results.csv', index=False)

        return results_df

    def intraclass_correlation_exploratory(self, df):
        """
        Analyzes cross-correlation for all pairs of series, finds the best lag for each pair, and saves the results in a CSV.
        """

        # Inicializar uma lista para armazenar os dados transformados
        data = []

        # Preencher os dados do IMU
        imu_columns = [
            'acc_x'
        ]

        for col in imu_columns:
            print(len(df[col]))
            for index, value in enumerate(df[col]):
                data.append([col, "IMU", value])

        # Preencher os dados cinemáticos (Kinematics)
        kinematic_columns = [
            "c7_x"
        ]

        for col in kinematic_columns:
            print(len(df[col]))
            for index, value in enumerate(df[col]):
                data.append([col, "Kinematic", value])

        df_combined = pd.DataFrame(data, columns=["body_parts", "type", "value"])
        df_combined['value'].fillna(df_combined['value'].mean(), inplace=True)


        icc_result = pg.intraclass_corr(data=df_combined, targets='body_parts', raters='type', ratings='value', nan_policy='omit')
        print(icc_result)