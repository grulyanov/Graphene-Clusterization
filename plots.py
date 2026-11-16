"""Графики сайта. Каждая функция возвращает matplotlib.figure.Figure.

Размеры, маркеры, цвета, подписи и заголовки — как в Graphene_unified_analysis_filtered.ipynb
(номер ячейки в docstring). Стиль: sns.set(); font_scale = 1 до ячейки 57 ноутбука и 1.7 после.
Рисуем через Figure (без pyplot), чтобы графики не мешали друг другу в Streamlit.
"""
from __future__ import annotations

import io

import numpy as np
import scipy.cluster.hierarchy as sch
import seaborn as sns
from matplotlib.figure import Figure

import analysis as A

FS_EARLY = 1.0   # ячейки до 57
FS_LATE = 1.7    # ячейка 57 и все графики после неё
MARKERS = ('o', 's', '^', 'D', 'P', 'X', 'v', '<', '>')


def _style(font_scale):
    sns.set_theme(font_scale=font_scale)   # то же, что sns.set(font_scale=...)


def _colors():
    import matplotlib as mpl
    return mpl.rcParams['axes.prop_cycle'].by_key()['color']


def line_colors():
    """Цвета прямых RANSAC по порядку (палитра seaborn «deep», как в sns.set())."""
    return sns.color_palette('deep').as_hex()


def to_png(fig, dpi=600):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    return buf.getvalue()


def to_svg(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Кластеризация (font_scale = 1)
# ---------------------------------------------------------------------------
def raw_scatter(sorption, x, y_axis_label, title_name, sorption_col_label):
    """Ячейка 8."""
    _style(FS_EARLY)
    fig = Figure(figsize=(10, 8))
    ax = fig.subplots()
    ax.scatter(sorption, x, s=35, edgecolor='k')
    ax.set_xlabel(sorption_col_label, fontsize=14)
    ax.set_ylabel(y_axis_label, fontsize=14)
    ax.set_title(f'Sorption vs {title_name} data', fontsize=14)
    return fig


def elbow(ks, wcss):
    """Ячейка 11."""
    _style(FS_EARLY)
    fig = Figure(figsize=(10, 8))
    ax = fig.subplots()
    sns.pointplot(x=ks, y=wcss, ax=ax)
    ax.set_title('The Elbow curve')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')
    return fig


def dendrogram(Z, labels, color_threshold):
    """Ячейка 12."""
    _style(FS_EARLY)
    fig = Figure(figsize=(12, 10))
    ax = fig.subplots()
    sch.dendrogram(Z, labels=labels, orientation='left', leaf_font_size=10,
                   color_threshold=color_threshold, ax=ax)
    for label in ax.get_yticklabels():
        label.set_color(A.AUTO_GROUP_COLORS[A.auto_group(label.get_text())])
    ax.set_title('Dendrogram')
    ax.set_xlabel('Euclidean distances')
    fig.tight_layout()
    return fig


def cluster_scatter(X, labels, method_title, x_label, y_label, centers=None):
    """Ячейки 15, 20, 30, 34."""
    _style(FS_EARLY)
    fig = Figure(figsize=(8, 6))
    ax = fig.subplots()
    labels = np.asarray(labels)
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        label = 'Noise' if cluster == -1 else f'Cluster {cluster + 1}'
        ax.scatter(X[mask, 0], X[mask, 1], s=45, label=label, edgecolors='black')
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], s=90, label='Centroids', c='black',
                   edgecolors='yellow', marker='X')
    ax.set_title(f'Clusters of substances ({method_title})')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    return fig


def adjacency_hist(A_matrix, suptitle='Adjacency Matrix Value Distribution', figsize=(12, 5)):
    """Ячейки 26, 50, 52."""
    _style(FS_EARLY)
    fig = Figure(figsize=figsize)
    ax1, ax2 = fig.subplots(1, 2)
    bin_values, _, _ = ax1.hist(A_matrix.flatten(), bins=50)
    ax2.hist(A_matrix.flatten(), bins=50)
    ax2.set_ylim([0, bin_values[9]])
    ax1.set_xlabel('Value')
    ax2.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Adjacency Matrix Values')
    ax2.set_title('Area of small values (first 50 bins)')
    fig.suptitle(suptitle)
    return fig


def laplacian(eigenvalues_sorted):
    """Ячейка 27 (с защитой от малого числа веществ)."""
    _style(FS_EARLY)
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    values = eigenvalues_sorted[1:21]
    x = np.arange(0, len(values))
    ax.plot(x, values, 'bo-')
    ax.set_xticks(x)
    ax.set_xlabel('The index of the eigenvalue')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Laplacian Eigenvalues')
    return fig


# ---------------------------------------------------------------------------
# Последовательный RANSAC (ячейка 43, часть рисования)
# ---------------------------------------------------------------------------
def draw_sequential(ax, X, y, models, *, sizes=None, legend_fontsize=10, text_offset=None,
                    ci_level=None, tau_factor=1.0):
    X_arr = np.asarray(X, dtype=float).reshape(-1, 1)
    y_arr = np.asarray(y, dtype=float).ravel()
    sizes = [80] * len(MARKERS) if sizes is None else sizes
    colors = _colors()
    y_range = np.ptp(y_arr) if len(y_arr) and np.ptp(y_arr) != 0 else 1.0
    offset = (0.025 if text_offset is None else text_offset) * y_range
    for iteration, m in enumerate(models):
        model = m['model']
        inliers_idx = np.asarray(m['inliers_idx'])
        x_line = np.linspace(X_arr.min(), X_arr.max(), 200).reshape(-1, 1)
        color = colors[iteration % len(colors)]
        marker = MARKERS[iteration % len(MARKERS)]
        if ci_level:
            band = A.line_confidence_band(X_arr[inliers_idx].ravel(), y_arr[inliers_idx],
                                          x_line.ravel(), ci_level)
            if band is not None:
                ax.fill_between(x_line.ravel(), band[0], band[1], color=color, alpha=0.15,
                                linewidth=0)
        ax.plot(x_line.ravel(), model.predict(x_line), linestyle='--', linewidth=2, color=color,
                label=(f'$slope$={m["slope"]:.3f}, $R^2_{{in}}$={m["r2"]:.2f}, '
                       + f'$Q^2_{{LOO}}$={m["q2"]:.2f}' + chr(10)
                       + rf'$\tau$={A.format_tau_units(m["threshold"], tau_factor)}'))
        ax.scatter(X_arr[inliers_idx].ravel(), y_arr[inliers_idx], marker=marker,
                   s=sizes[iteration % len(sizes)], color=color, edgecolors='black',
                   linewidths=1.2, label=f'$n_{{inliers}}$={len(inliers_idx)}')
    all_inliers = sorted({int(i) for m in models for i in m['inliers_idx']},
                         key=lambda index: (y_arr[index], index))
    rank_map = {index: rank for rank, index in enumerate(all_inliers, start=1)}
    for index in all_inliers:
        ax.text(float(X_arr[index].item()), float(y_arr[index] + offset), str(rank_map[index]),
                fontsize=9, ha='center', va='bottom', color='black')
    if models:
        ax.legend(fontsize=legend_fontsize, loc='best')


def descriptor_ransac(data, descriptor, label, models, y_label, *, ci_level=None, tau_factor=1.0,
                      ycol=A.SORPTION_MMOL):
    """Ячейки 59, 60, 70–78 (analyze_descriptor): 10×8, font_scale 1.7."""
    _style(FS_LATE)
    fig = Figure(figsize=(10, 8))
    ax = fig.subplots()
    sns.scatterplot(ax=ax, x=data[descriptor], y=data[ycol], s=40, edgecolors='black', color='grey')
    draw_sequential(ax, data[descriptor], data[ycol], models, sizes=[80] * 9, legend_fontsize=11,
                    ci_level=ci_level, tau_factor=tau_factor)
    ax.set_xlabel(label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(f'Dependence of sorption on {label}')
    fig.tight_layout()
    return fig


def descriptor_pair(results, labels, y_label, suptitle, *, ci_level=None, tau_factor=1.0,
                    ycol=A.SORPTION_MMOL):
    """Ячейка 57: два дескриптора рядом, 16×10, легенда 12."""
    _style(FS_LATE)
    fig = Figure(figsize=(16, 10))
    axes = fig.subplots(1, 2)
    for ax, (descriptor, res), label in zip(axes, results.items(), labels):
        data = res['data']
        sns.scatterplot(ax=ax, x=data[descriptor], y=data[ycol], s=40, edgecolors='black', color='grey')
        draw_sequential(ax, data[descriptor], data[ycol], res['models'], sizes=[80] * 9,
                        legend_fontsize=12, ci_level=ci_level, tau_factor=tau_factor)
        ax.set_xlabel(label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def cluster_ransac(groups, method_name, x_label, y_label, *, ci_level=None, tau_factor=1.0):
    """Ячейки 44–47 (analyze_cluster_groups): сетка 8n×7m, font_scale 1."""
    _style(FS_EARLY)
    n = max(1, len(groups))
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig = Figure(figsize=(8 * ncols, 7 * nrows))
    axes = np.atleast_1d(fig.subplots(nrows, ncols)).ravel()
    for ax, g in zip(axes, groups):
        data = g['data']
        sns.scatterplot(ax=ax, x=data[x_label], y=data[y_label], s=40, edgecolors='black', color='grey')
        draw_sequential(ax, data[x_label], data[y_label], g['models'], sizes=[80] * 9,
                        legend_fontsize=9, ci_level=ci_level, tau_factor=tau_factor)
        ax.set_title(f'{method_name}: Cluster {g["cluster"]}')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    for ax in axes[len(groups):]:
        ax.axis('off')
    fig.suptitle(f'{method_name} clusters with Sequential RANSAC')
    fig.tight_layout()
    return fig


def ransac_plane(res, x_col, y_col, z_col, tau_factor=1.0):
    """Ячейка 62 (ransac_plane_labels): 3D, 8×10, font_scale 1.7."""
    _style(FS_LATE)
    data = res['data']
    tuned = res['tuned']
    model = tuned['model']
    mask = model.inlier_mask_
    a, b = model.estimator_.coef_
    c = model.estimator_.intercept_
    x = data[x_col].to_numpy(float)
    y = data[y_col].to_numpy(float)
    z = data[z_col].to_numpy(float)
    fig = Figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=30, c='lightgray', edgecolors='k', linewidths=0.35)
    ax.scatter(x[mask], y[mask], z[mask], s=55, c='royalblue', edgecolors='k', linewidths=0.8,
               label='inliers')
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 25), np.linspace(y.min(), y.max(), 25))
    ax.plot_surface(xx, yy, a * xx + b * yy + c, alpha=0.28, linewidth=0, antialiased=True,
                    cmap='viridis')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(rf'$S={a:.2f}{x_col}{b:+.2f}{y_col}{c:+.2f}$' + chr(10)
                 + rf'$R^2_{{in}}={tuned["r2"]:.2f}, Q^2_{{LOO}}={tuned["q2"]:.2f}, '
                 + rf'\tau={A.format_tau_units(tuned["threshold"], tau_factor)}$')
    for rank, index in enumerate(np.flatnonzero(mask), start=1):
        ax.text(x[index], y[index], z[index], str(rank), fontsize=8)
    ax.legend(loc='best')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Валидация (font_scale = 1.7)
# ---------------------------------------------------------------------------
def collinearity(corr, evr):
    """Ячейка 80: корреляционная матрица + объяснённая дисперсия PCA, 15×6."""
    _style(FS_LATE)
    fig = Figure(figsize=(15, 6))
    axes = fig.subplots(1, 2)
    comps = np.arange(1, len(evr) + 1)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True, ax=axes[0])
    axes[0].set_title('Descriptor correlation matrix')
    axes[1].bar(comps, evr)
    axes[1].plot(comps, np.cumsum(evr), marker='o')
    axes[1].set_xticks(comps)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel('Principal component')
    axes[1].set_ylabel('Explained variance')
    axes[1].set_title('PCA explained variance')
    fig.tight_layout()
    return fig


def permutation_hists(results, xlabel=r'$S_{\mathrm{corr}}$ under permutation'):
    """Ячейка 82: гистограммы нулевого распределения (2 в ряд; в ноутбуке 2×2, 14×10)."""
    _style(FS_LATE)
    n = max(1, len(results))
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig = Figure(figsize=(14 if ncols == 2 else 7, 5 * nrows))
    axes = np.atleast_1d(fig.subplots(nrows, ncols)).ravel()
    for ax, (descriptor, r) in zip(axes, results.items()):
        ax.hist(r['null'], bins=30)
        ax.axvline(r['observed'], linestyle='--', linewidth=2, color='black')
        ax.set_title(f'{descriptor}: p={r["p"]:.4f}')
        ax.set_xlabel(xlabel)
    for ax in axes[len(results):]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def pls_observed_predicted(y, predictions, q2, rmsecv, unit_label='mmol/g'):
    """Ячейка 84: 8×7."""
    _style(FS_LATE)
    fig = Figure(figsize=(8, 7))
    ax = fig.subplots()
    ax.scatter(y, predictions, s=50, edgecolors='black')
    limits = [min(y.min(), predictions.min()), max(y.max(), predictions.max())]
    ax.plot(limits, limits, '--', color='black')
    for rank, index in enumerate(np.argsort(y), start=1):
        ax.text(y[index], predictions[index], str(rank), fontsize=9, ha='center', va='bottom')
    ax.set_xlabel(f'Observed sorption ({unit_label})')
    ax.set_ylabel(f'LOO-predicted sorption ({unit_label})')
    ax.set_title(f'Nested LOO PLS: Q²={q2:.3f}, RMSECV={rmsecv:.3f}')
    fig.tight_layout()
    return fig


def y_randomization_hist(null_q2, observed_q2, p):
    """НОВОЕ (в стиле ячейки 82): распределение Q² при перемешанной сорбции."""
    _style(FS_LATE)
    fig = Figure(figsize=(8, 6))
    ax = fig.subplots()
    ax.hist(null_q2, bins=30)
    ax.axvline(observed_q2, linestyle='--', linewidth=2, color='black')
    ax.set_title(f'y-randomization: p={p:.4f}')
    ax.set_xlabel(r'$Q^2_{LOO}$ (nested PLS) under permutation')
    fig.tight_layout()
    return fig
