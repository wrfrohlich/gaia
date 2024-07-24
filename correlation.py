from pandas import merge_asof
from scipy.stats import pearsonr, spearmanr, kendalltau

import seaborn as sns
import matplotlib.pyplot as plt

class Correlation():
    def corr_matrix(self, data):
        corr_matrix_pearson = data.corr(method='pearson')
        corr_matrix_spearman = data.corr(method='spearman')
        corr_matrix_kendall = data.corr(method='kendall')
        
        print(corr_matrix_pearson)
        print(corr_matrix_spearman)
        print(corr_matrix_kendall)

        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix_pearson, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix - Pearson')
        plt.savefig('figures/corr_matrix_corr_matrix_pearson.png')

        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix_spearman, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix - spearman')
        plt.savefig('figures/corr_matrix_corr_matrix_spearman.png')

        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix_kendall, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix - kendall')
        plt.savefig('figures/corr_matrix_corr_matrix_kendall.png')

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