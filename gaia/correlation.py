import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import correlate
from scipy.stats import pearsonr, spearmanr, kendalltau

class Correlation():
    def __init__(self):
        self.path_matrix = 'figures/mean'
        self.path_cross = 'figures/cross'

        if not os.path.exists(self.path_matrix):
            os.makedirs(self.path_matrix)

        if not os.path.exists(self.path_cross):
            os.makedirs(self.path_cross)

    def corr_matrix(self, data):
        corr_matrix_pearson = data.corr(method='pearson')
        corr_matrix_spearman = data.corr(method='spearman')
        corr_matrix_kendall = data.corr(method='kendall')

        print(corr_matrix_pearson)

        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix_pearson, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix - Pearson')
        plt.savefig(f'{self.path_matrix}/corr_matrix_corr_matrix_pearson.png')

        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix_spearman, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix - spearman')
        plt.savefig(f'{self.path_matrix}/corr_matrix_corr_matrix_spearman.png')

        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix_kendall, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix - kendall')
        plt.savefig(f'{self.path_matrix}/corr_matrix_corr_matrix_kendall.png')

    def corr(self, x, y):
        pearson_corr, _ = pearsonr(x, y)
        print(_)
        spearman_corr, _ = spearmanr(x, y)
        print(_)
        kendall_corr, _ = kendalltau(x, y)
        print(_)

        print(pearson_corr)
        print(spearman_corr)
        print(kendall_corr)

    def cross_correlation_uniq(self, series1, series2):
        correlation = correlate(series1, series2, mode='full')
        lags = np.arange(-len(series1) + 1, len(series1))
        return lags, correlation

    def cross_correlation(self, merged_data):
        columns = merged_data.columns[merged_data.columns != 'time']
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]

                lags, correlation = self.cross_correlation_uniq(merged_data[col1], merged_data[col2])

                min_len = min(len(lags), len(correlation))
                lags = lags[:min_len]
                correlation = correlation[:min_len]
                print(correlation)
                # plt.figure(figsize=(14, 5))
                # plt.plot(lags, correlation)
                # plt.title('Correlação Cruzada')
                # plt.xlabel('Deslocamento')
                # plt.ylabel('Correlação')
                # plt.savefig(f'{self.path_cross}/cross_corr_{col1}_{col2}.png')
