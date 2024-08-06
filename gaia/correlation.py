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
    config : Config
        An instance of the Config class.
    path_matrix : str
        Directory path for saving correlation matrix plots.
    path_cross : str
        Directory path for saving cross-correlation plots.

    Methods
    -------
    __init__()
        Initializes the Correlation class and creates necessary directories for saving plots.
    corr_matrix(data)
        Computes and saves correlation matrices (Pearson, Spearman, Kendall) for the given data.
    corr(x, y)
        Computes Pearson, Spearman, and Kendall correlation coefficients between two series.
    cross_correlation_uniq(series1, series2)
        Computes cross-correlation between two series.
    cross_correlation(merged_data)
        Computes and saves cross-correlation plots for all pairs of columns in the merged data.
    """

    def __init__(self, name):
        """
        Initializes the Correlation class with Config and sets up directories for saving plots.
        """
        config = Config()
        self.path_experiment = config.figures

        if name != "":
            self.path_experiment = path.join(self.path_experiment, name)
            if not path.exists(self.path_experiment):
                os.makedirs(self.path_experiment)

    def corr_matrix(self, data):
        """
        Computes and saves Pearson, Spearman, and Kendall correlation matrices for the given data.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing the data for which to compute correlation matrices.

        Returns
        -------
        None
            This method does not return any values. It generates and saves correlation matrix plots.
        """
        self.gen_corr_matrix(data_upper, name="upper_body")
        self.gen_corr_matrix(data_lower_01, name="lower_body_01")
        self.gen_corr_matrix(data_lower_02, name="lower_body_02")

        # Create a new DataFrame with vector magnitudes and other metrics
        df = pd.DataFrame()
        df["roll"] = data["roll"]
        df["pitch"] = data["pitch"]
        df["yaw"] = data["yaw"]
        df["acc"] = self.get_vector(data, "acc")
        df["gyro"] = self.get_vector(data, "gyro")
        df["r should"] = self.get_vector(data, "r should")
        df["l should"] = self.get_vector(data, "l should")
        df["sacrum"] = self.get_vector(data, "sacrum s")
        df["PO"] = self.get_vector(data, "PO")
        df["r knee 1"] = self.get_vector(data, "r knee 1")
        df["l knee 1"] = self.get_vector(data, "l knee 1")
        df["r mall"] = self.get_vector(data, "r mall")
        df["l mall"] = self.get_vector(data, "l mall")
        df["r heel"] = self.get_vector(data, "r heel")
        df["l heel"] = self.get_vector(data, "l heel")
        df["r met"] = self.get_vector(data, "r met")
        df["l met"] = self.get_vector(data, "l met")

        self.gen_corr_matrix(df, name="vectors")

        columns = df.columns[df.columns != 'time']
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]
                self.print_cross_correlation(df, col1, col2)


    def gen_corr_matrix(self, data, method="pearson"):
        """
        Generates and saves correlation matrices (Pearson, Spearman, Kendall) for the given data.

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
        corr_matrix_pearson = data.corr(method=method)

        # Plot and save Pearson correlation matrix
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix_pearson, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title(f'Correlation Matrix - Pearson {name}')
        plt.savefig(f'{self.path_matrix}/.png')
        plt.clf()


    def print_cross_correlation(self, df, value_a, value_b):
        """
        Computes and plots cross-correlation between two data series.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data series.
        value_a : str
            The first data series name.
        value_b : str
            The second data series name.

        Returns
        -------
        None
            This method does not return any values. It generates and saves cross-correlation plots.
        """
        value_A = df[value_a]
        value_B = df[value_b]
        cross_corrs = [self.cross_correlation(value_A, value_B, lag) for lag in range(-10, 10, 1)]

        # Plot cross-correlation
        plt.figure(figsize=(12, 6))
        plt.plot(range(-10, 10, 1), cross_corrs, marker='o')
        plt.xlabel('Lag')
        plt.ylabel('Cross-Correlation')
        plt.title(f'Cross-Correlation between {value_a} and {value_b}')
        plt.savefig(f'{self.path_cross}/{value_a}_and_{value_b}.png')
        plt.clf()

    def cross_correlation(self, a, b, lag=0):
        """
        Computes the cross-correlation between two data series at a specified lag.

        Parameters
        ----------
        a : array-like
            The first data series.
        b : array-like
            The second data series.
        lag : int, optional
            The lag at which to compute the cross-correlation (default is 0).

        Returns
        -------
        float
            The cross-correlation value.
        """
        return np.corrcoef(a[:-lag or None], b[lag:])[0, 1]

    def get_vector(self, data, label):
        """
        Computes the magnitude of vectors for each row in the DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the data.
        label : str
            The prefix of the columns representing the vector components.

        Returns
        -------
        pd.Series
            A Series containing the magnitudes of the vectors.
        """
        value = data.apply(lambda row: self.create_vector(row, label), axis=1)
        value = value.apply(np.linalg.norm)
        return value

    def create_vector(self, row, prefix):
        """
        Creates a 3D vector from row data.

        Parameters
        ----------
        row : pd.Series
            A row of data from the DataFrame.
        prefix : str
            The prefix of the columns representing the vector components.

        Returns
        -------
        np.array
            A 3D vector.
        """
        return np.array([row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_z"]])


    def corr(self, x, y):
        """
        Computes Pearson, Spearman, and Kendall correlation coefficients between two series.

        Parameters
        ----------
        x : array-like
            The first data series.
        y : array-like
            The second data series.

        Returns
        -------
        None
            This method does not return any values. It prints the correlation coefficients.
        """
        pearson_corr, _ = pearsonr(x, y)
        spearman_corr, _ = spearmanr(x, y)
        kendall_corr, _ = kendalltau(x, y)
        
        print(f'Pearson correlation: {pearson_corr}')
        print(f'Spearman correlation: {spearman_corr}')
        print(f'Kendall correlation: {kendall_corr}')

    def cross_correlation_uniq(self, series1, series2):
        """
        Computes the cross-correlation between two data series.

        Parameters
        ----------
        series1 : array-like
            The first data series.
        series2 : array-like
            The second data series.

        Returns
        -------
        tuple
            A tuple containing the lags and the cross-correlation values.
        """
        correlation = correlate(series1, series2, mode='full')
        lags = np.arange(-len(series1) + 1, len(series1))
        return lags, correlation
