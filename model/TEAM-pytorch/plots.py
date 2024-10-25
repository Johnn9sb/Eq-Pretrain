import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from scipy.stats import norm


def calibration_plot(y_pred, y_true, bins=100, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)

    y_true = y_true.reshape(-1, 1)
    prob = np.sum(
        y_pred[:, :, 0] * (1 - norm.cdf((y_true - y_pred[:, :, 1]) / y_pred[:, :, 2])),
        axis=-1, keepdims=True)
    sns.distplot(prob, norm_hist=True, bins=bins, hist_kws={'range': (0, 1)}, kde=False, ax=ax)
    ax.axhline(1., linestyle='--', color='r')
    ax.set_xlim(0, 1)
    ax.set_ylim(0)
    return ax

thresholds = 0
def true_predicted(y_true, y_pred, agg='mean', quantile=True, ms=None, ax=None, time='error'):
    pga_threshold = thresholds
    select_indices = y_true > pga_threshold
    y_true = y_true[select_indices]
    y_pred = y_pred[select_indices]
    
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    quantile = False
    if quantile:
        c_quantile = np.sum(y_pred[:, :, 0] * (1 - norm.cdf((y_true.reshape(-1, 1) - y_pred[:, :, 1]) / y_pred[:, :, 2])),
                            axis=-1, keepdims=False)
    else:
        c_quantile = None

    y_pred_point = np.sum(y_pred[:, :, 0] * y_pred[:, :, 1], axis=1)
    errors = np.abs(y_pred_point - y_true) 
    normalized_errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors)) 

    cmap = plt.get_cmap('coolwarm')
    colors = cmap(normalized_errors)

    limits = (np.min(y_true) - 0.5, np.max(y_true) + 0.5)
    ax.plot(limits, limits, 'k-', zorder=1)
    if ms is None:
        cbar = ax.scatter(y_true, y_pred_point, c=colors, cmap=cmap, zorder=2)  #(x軸, y軸, c:資料點的顏色, cmap:顏色地圖)
    else:
        cbar = ax.scatter(y_true, y_pred_point, c=colors, cmap=cmap, s=ms, zorder=2) 

    ax.set_xlabel('$y_{true}$')
    ax.set_ylabel('$y_{pred}$')

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    
    ax.set_xticklabels(['0', '1', '10'])
    ax.set_yticklabels(['0', '1', '10'])
    
    r2 = metrics.r2_score(y_true, y_pred_point)
    ax.text(min(np.min(y_true), limits[0]), max(np.max(y_pred_point), limits[1]), f"Time: {time}\n$R^2:$ {r2:.2f}", va='top')

    return ax, cbar