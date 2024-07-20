from scipy.stats import pearsonr, spearmanr, kendalltau

x = [1, 2, 3]
y = [1, 5, 6]

pearson_corr, _ = pearsonr(x, y)
print(_)
spearman_corr, _ = spearmanr(x, y)
print(_)
kendall_corr, _ = kendalltau(x, y)
print(_)

print(pearson_corr)
print(spearman_corr)
print(kendall_corr)