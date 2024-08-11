import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os import path
from scipy.signal import correlate
from scipy.stats import pearsonr, spearmanr, kendalltau

from .config import Config

class Correlation:
    """
    A class for computing and visualizing various correlation metrics between data series.

    Attributes
    ----------
    path_experiment : str
        Directory path for saving correlation matrices and cross-correlation plots.
    points : dict
        Dictionary containing body parts and IMU points as defined in the configuration file.

    Methods
    -------
    __init__(name)
        Initializes the Correlation class and creates necessary directories for saving plots.
    get_higher_corr(data, level=0.7, method="pearson")
        Computes and saves high correlation values between body points and IMUs.
    analyze_correlation()
        Analyzes correlations from a CSV file and generates a report.
    corr_matrix(data)
        Computes and saves Pearson, Spearman, or Kendall correlation matrices for the given data.
    corr_matrix_special(data, name="")
        Computes and saves correlation matrices with a specific name.
    gen_corr_matrix(data, name, method="pearson")
        Generates and saves correlation matrices for the given data.
    cross_correlation(data)
        Computes and plots cross-correlation for all pairs of columns in the DataFrame.
    print_cross_corr(df, value_a, value_b)
        Computes and plots cross-correlation between two data series.
    calculate_cross_correlation(series_a, series_b, lag)
        Calculates the cross-correlation between two series for a given lag.
    cross_correlation_uniq(series1, series2)
        Computes the cross-correlation between two data series.
    """

    def __init__(self, name):
        """
        Initializes the Correlation class with Config and sets up directories for saving plots.

        Parameters
        ----------
        name : str
            A name for the experiment to create a specific directory for saving results.
        """
        config = Config()
        self.path_experiment = config.figures
        self.points = config.body_parts

        if name != "":
            self.path_experiment = path.join(self.path_experiment, name)
            if not path.exists(self.path_experiment):
                os.makedirs(self.path_experiment)

    def get_higher_corr(self, data, level=0.7, method="pearson"):
        """
        Computes and saves high correlation values between body points and IMUs.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing the data for which to compute correlation matrices.
        level : float, optional
            The correlation level threshold above which the correlations are considered significant (default is 0.7).
        method : str, optional
            The method used for computing the correlation ('pearson', 'spearman', 'kendall') (default is 'pearson').

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the correlations that meet the threshold.
        """
        try:
            level = float(level)
        except ValueError:
            raise ValueError(f"Level should be a number, but got {level}")

        correlations_list = []

        for points in self.points:
            if points == "imu":
                continue
            columns = list(self.points[points])
            columns.extend(self.points["imu"])
            missing_cols = [col for col in columns if col not in data.columns]
            if missing_cols:
                continue
            
            df = data[columns]
            df = df.loc[:, ~df.columns.duplicated()]
            corr_matrix = df.corr(method=method)
            
            for x in corr_matrix.columns:
                if x in self.points["imu"]:
                    for y in corr_matrix.index:
                        if y not in self.points["imu"]:
                            correlation_value = corr_matrix.loc[y, x]
                            if abs(correlation_value) > level and x != y:
                                correlations_list.append({
                                    'group': points,
                                    'imu': x,
                                    'point': y,
                                    'corr': correlation_value
                                })

        correlations_df = pd.DataFrame(correlations_list)
        correlations_df = correlations_df.sort_values(by=["imu", "corr"], ascending=[True, False])
        correlations_df = correlations_df.reset_index(drop=True)

        correlations_df.to_csv(f'{self.path_experiment}/correlations.csv', index=False)

        return correlations_df

    def analyze_correlation(self):
        """
        Analyzes correlations from a CSV file and generates a report.

        This method reads the correlations.csv file, analyzes the correlations, and generates
        a report showing which body points have significant correlations with each IMU.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the correlation report, with IMUs as rows and correlated points as columns.
        """
        imus = self.points["imu"]
        data = pd.read_csv(f'{self.path_experiment}/correlations.csv')

        correlation_report = {}

        if 'imu' not in data.columns or 'point' not in data.columns:
            raise ValueError("The CSV file does not contain the expected 'imu' and 'point' columns.")

        for imu in imus:
            filtered_df = data[data["imu"] == imu]
            filtered_df = filtered_df.sort_values(by=["point"], ascending=True).reset_index(drop=True)

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
            return pd.DataFrame()  # Return an empty DataFrame if there is no data

    def corr_matrix(self, data):
        """
        Computes and saves Pearson, Spearman, or Kendall correlation matrices for the given data.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing the data for which to compute correlation matrices.

        Returns
        -------
        None
            This method does not return any values. It generates and saves correlation matrix plots.
        """
        for points in self.points:
            columns = list(self.points[points])
            columns.extend(self.points["imu"])
            missing_cols = [col for col in columns if col not in data.columns]
            if missing_cols:
                continue
            df = data[columns]
            self.gen_corr_matrix(df, name=points)

    def corr_matrix_special(self, data, name=""):
        """
        Computes and saves Pearson, Spearman, and Kendall correlation matrices for the given data.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing the data for which to compute correlation matrices.
        name : str, optional
            A name to identify the correlation matrix (default is an empty string).

        Returns
        -------
        None
            This method does not return any values. It generates and saves correlation matrix plots.
        """
        self.gen_corr_matrix(data, name=name)

    def gen_corr_matrix(self, data, name, method="pearson"):
        """
        Generates and saves correlation matrices (Pearson, Spearman, Kendall) for the given data.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing the data for which to compute correlation matrices.
        name : str
            A name to identify the correlation matrix.
        method : str, optional
            The method used for computing the correlation ('pearson', 'spearman', 'kendall') (default is 'pearson').

        Returns
        -------
        None
            This method does not return any values. It generates and saves correlation matrix plots.
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

    def cross_correlation(self, data):
        """
        Computes and plots cross-correlation for all pairs of columns in the DataFrame, excluding the 'time' column.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the data series.

        Returns
        -------
        None
            This method does not return any values. It generates and saves cross-correlation plots.
        """
        columns = data.columns[data.columns != 'time']
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]
                self.print_cross_corr(data, col1, col2)

    def print_cross_corr(self, df, value_a, value_b):
        """
        Computes and plots cross-correlation between two data series.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data series.
        value_a : str
            The first data series for cross-correlation.
        value_b : str
            The second data series for cross-correlation.

        Returns
        -------
        None
            This method does not return any values. It generates and saves cross-correlation plots.
        """
        plt.xcorr(df[value_a], df[value_b], maxlags=100, usevlines=True, normed=True, lw=2)
        plt.grid(True)
        plt.title(f'cross_correlation {value_a} vs {value_b}')
        plt.savefig(f'{self.path_experiment}/cross_corr_{value_a}_vs_{value_b}.png')
        plt.clf()

    def calculate_cross_correlation(self, series_a, series_b, lag):
        """
        Calculates the cross-correlation between two series for a given lag.

        Parameters
        ----------
        series_a : pd.Series
            The first data series.
        series_b : pd.Series
            The second data series.
        lag : int
            The lag value for which to compute the cross-correlation.

        Returns
        -------
        float
            The cross-correlation value at the specified lag.
        """
        return np.corrcoef(series_a[:-lag], series_b[lag:])[0, 1]

    def cross_correlation_uniq(self, series1, series2):
        """
        Computes the cross-correlation between two data series.

        Parameters
        ----------
        series1 : pd.Series
            The first data series.
        series2 : pd.Series
            The second data series.

        Returns
        -------
        np.ndarray
            An array of cross-correlation values for different lags.
        """
        return correlate(series1, series2)
