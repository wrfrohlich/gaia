import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    def __init__(self):
        """
        Initializes the Correlation class with Config and sets up directories for saving plots.
        """
        self.config = Config()
        self.path_matrix = 'figures/matrix'
        self.path_cross = 'figures/cross'

        # Create directories if they do not exist
        if not os.path.exists(self.path_matrix):
            os.makedirs(self.path_matrix)

        if not os.path.exists(self.path_cross):
            os.makedirs(self.path_cross)

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
        data_upper = data[[
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "roll", "pitch", "yaw",
            "r should.X", "r should.Y", "r should.Z",
            "l should.X", "l should.Y", "l should.Z",
            "sacrum s.X", "sacrum s.Y", "sacrum s.Z",
            "PO.X", "PO.Y", "PO.Z"
        ]]
        data_lower_01 = data[[
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "roll", "pitch", "yaw",
            "r knee 1.X", "r knee 1.Y", "r knee 1.Z",
            "l knee 1.X", "l knee 1.Y", "l knee 1.Z",
            "r mall.X", "r mall.Y", "r mall.Z",
            "l mall.X", "l mall.Y", "l mall.Z"
        ]]
        data_lower_02 = data[[
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "roll", "pitch", "yaw",
            "r heel.X", "r heel.Y", "r heel.Z",
            "l heel.X", "l heel.Y", "l heel.Z",
            "r met.X", "r met.Y", "r met.Z",
            "l met.X", "l met.Y", "l met.Z"
        ]]
        self.gen_corr_matrix(data_upper, name="upper_body")
        self.gen_corr_matrix(data_lower_01, name="lower_body_01")
        self.gen_corr_matrix(data_lower_02, name="lower_body_02")

    def gen_corr_matrix(self, data, name=""):
        corr_matrix_pearson = data.corr(method='pearson')
        corr_matrix_spearman = data.corr(method='spearman')
        corr_matrix_kendall = data.corr(method='kendall')

        # Plot and save Pearson correlation matrix
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix_pearson, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix - Pearson')
        plt.savefig(f'{self.path_matrix}/corr_matrix_pearson_{name}.png')
        plt.clf()

        # Plot and save Spearman correlation matrix
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix_spearman, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix - Spearman')
        plt.savefig(f'{self.path_matrix}/corr_matrix_spearman_{name}.png')
        plt.clf()

        # Plot and save Kendall correlation matrix
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix_kendall, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix - Kendall')
        plt.savefig(f'{self.path_matrix}/corr_matrix_kendall_{name}.png')
        plt.clf()

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

    def cross_correlation(self, merged_data):
        """
        Computes and saves cross-correlation plots for all pairs of columns in the merged data.

        Parameters
        ----------
        merged_data : pd.DataFrame
            A DataFrame containing the merged data with columns to compute cross-correlation.

        Returns
        -------
        None
            This method does not return any values. It generates and saves cross-correlation plots.
        """
        columns = merged_data.columns[merged_data.columns != 'time']
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]

                lags, correlation = self.cross_correlation_uniq(merged_data[col1], merged_data[col2])

                min_len = min(len(lags), len(correlation))
                lags = lags[:min_len]
                correlation = correlation[:min_len]

                plt.figure(figsize=(14, 5))
                plt.plot(lags, correlation)
                plt.title(f'Cross Correlation between {col1} and {col2}')
                plt.xlabel('Lags')
                plt.ylabel('Correlation')
                plt.savefig(f'{self.path_cross}/cross_corr_{col1}_{col2}.png')
                plt.clf()
