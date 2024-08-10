import os
import numpy as np
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
        self.points = config.body_parts

        if name != "":
            self.path_experiment = path.join(self.path_experiment, name)
            if not path.exists(self.path_experiment):
                os.makedirs(self.path_experiment)

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
        name : str, optional
            A name to identify the correlation matrix (default is an empty string).

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

        # Compute cross-correlation
        cross_corrs = [self.calculate_cross_correlation(value_A, value_B, lag) for lag in range(-10, 10)]

        # Plot cross-correlation
        plt.figure(figsize=(12, 6))
        plt.plot(range(-10, 10), cross_corrs, marker='o')
        plt.xlabel('Lag')
        plt.ylabel('Cross-Correlation')
        plt.title(f'Cross-Correlation between {value_a} and {value_b}')
        plt.savefig(f'{self.path_cross}/{value_a}_and_{value_b}.png')
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
            The lag value for which to calculate the cross-correlation.

        Returns
        -------
        float
            The cross-correlation value at the specified lag.
        """
        if lag < 0:
            return series_a[:lag].corr(series_b[-lag:])
        else:
            return series_a[lag:].corr(series_b[:-lag])

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
