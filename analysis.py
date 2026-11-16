"""Расчёты сайта.

Функции из Graphene_unified_analysis_filtered.ipynb перенесены без изменения логики
(номер ячейки ноутбука указан в docstring). Глобальные переменные ноутбука (df, X_scaled,
ax, ...) заменены аргументами, а рисование отделено от расчёта: графики строит plots.py
по результатам этих функций. Новые расчёты, которых нет в ноутбуке, помечены «НОВОЕ».
"""
from __future__ import annotations

import math
import unicodedata
import warnings
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings('ignore')

NAME = 'Name'
SORPTION_MMOL = 'Sorption, mmol/g'

# ---------------------------------------------------------------------------
# Ячейка 5: исключение зондов по точному нормализованному имени
# ---------------------------------------------------------------------------
EXCLUDED_PROBE_ALIASES = {
    'n-Butylamine': ['n-Butylamine', '1-Butylamine', 'Butylamine',
                     'Butan-1-amine', '1-Aminobutane', '1-бутиламин', 'н-бутиламин'],
    'Morpholine': ['Morpholine', 'Морфолин'],
    'Piperidine': ['Piperidine', 'Пиперидин'],
    'Phosphorus tribromide': ['Phosphorus tribromide', 'PBr3',
                              'Трибромид фосфора', 'Phosphorus(III) bromide'],
    'Thionyl chloride': ['Thionyl chloride', 'SOCl2', 'Тионилхлорид', 'Хлористый тионил'],
}


def probe_key(value) -> str:
    """Ячейка 5: нормализация имени (NFKC, casefold, только буквы и цифры)."""
    text = unicodedata.normalize('NFKC', str(value)).casefold()
    return ''.join(ch for ch in text if ch.isalnum())


_EXCLUDED_KEYS = {probe_key(a) for aliases in EXCLUDED_PROBE_ALIASES.values() for a in aliases}


def notebook_excluded_mask(names: Iterable) -> np.ndarray:
    """True для веществ, исключённых в ноутбуке filtered (точное совпадение, не подстрока)."""
    return np.array([probe_key(n) in _EXCLUDED_KEYS for n in names], dtype=bool)


# ---------------------------------------------------------------------------
# Ячейка 12: группы для цвета подписей дендрограммы
# ---------------------------------------------------------------------------
import re

AUTO_GROUP_PATTERNS = {
    'oxygenated': r'Dimethylformamide|N-Methyl-2-pyrrolidone|Citraconic anhydride|Furfural|Acetylacetone|Acetic anhydride|Water|Heavy water|Hydrogen peroxide|Ethyl acetate|Acetic acid|Tetrahydrofuran|Dibutyl phthalate|1,4-Dioxane',
    'nitrogenated': r'N-Methyl-2-pyrrolidone|Acetonitrile|Dimethylformamide|Pyridine|Pyrrolidine|n-Butylamine|Dicyclohexylamine|N,N,N\',N\'-Tetramethylethylenediamine|Pyridine-d5|Morpholine|Pyridine|N-Methylaniline|Pyridazine|Et3N|Triethylenetetramine|Pyridine|Aniline|Piperidine|Morpholine|N,N-Dimethylformamide|Diethylamine|Pivalonitrile|Benzonitrile',
    'sulfur': r'Dimethyl sulfoxide|Thionyl chloride|Thiophos',
    'phosphorus': r'Tri-n-butyl phosphate|Phosphorus tribromide',
}
AUTO_GROUP_COLORS = {'oxygenated': 'red', 'nitrogenated': 'blue', 'sulfur': '#FFA600',
                     'phosphorus': '#F311EF', 'other': 'black'}


def auto_group(name: str) -> str:
    for group, pattern in AUTO_GROUP_PATTERNS.items():
        if re.search(pattern, name):
            return group
    return 'other'


# ---------------------------------------------------------------------------
# Кластеризация (ячейки 11–37)
# ---------------------------------------------------------------------------
def scale(X):
    return StandardScaler().fit_transform(X)


def elbow_wcss(X_scaled, k_max=10, seed=42):
    """Ячейка 11."""
    ks = list(range(1, min(k_max, len(X_scaled)) + 1))
    wcss = []
    for k in ks:
        model = KMeans(n_clusters=k, init='k-means++', n_init=50, random_state=seed)
        model.fit(X_scaled)
        wcss.append(model.inertia_)
    return ks, wcss


def _k_range(n, k_min=2, k_max=8):
    return list(range(k_min, min(k_max, n - 1) + 1))


def hierarchical_candidates(X_scaled, linkages=('ward', 'complete', 'average'), k_min=2, k_max=8):
    """Ячейка 12: перебор linkage × k, выбор max(silhouette, −DB, k, linkage)."""
    rows = []
    for linkage_name in linkages:
        for k in _k_range(len(X_scaled), k_min, k_max):
            labels = AgglomerativeClustering(n_clusters=k, linkage=linkage_name).fit_predict(X_scaled)
            rows.append((silhouette_score(X_scaled, labels), -davies_bouldin_score(X_scaled, labels),
                         k, linkage_name))
    return rows


def fit_hierarchical(X_scaled, k, linkage):
    return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X_scaled)


def dendrogram_linkage(names, X_raw, method):
    """Ячейка 12: дубликаты имён убираются (keep='last'), признаки масштабируются заново."""
    frame = pd.DataFrame(X_raw, columns=['a', 'b'])
    frame.insert(0, NAME, list(names))
    removal = frame.drop_duplicates(subset=[NAME], keep='last')
    import scipy.cluster.hierarchy as sch
    Z = sch.linkage(StandardScaler().fit_transform(removal[['a', 'b']]), method=method)
    return Z, removal[NAME].astype(str).to_numpy()


def dendrogram_threshold(Z, k, n_leaves):
    if 1 < k < n_leaves:
        return (Z[-k, 2] + Z[-k + 1, 2]) / 2
    return None


def kmeans_candidates(X_scaled, k_min=2, k_max=8, seed=42):
    """Ячейка 14."""
    rows = []
    for k in _k_range(len(X_scaled), k_min, k_max):
        labels = KMeans(n_clusters=k, init='k-means++', n_init=50, random_state=seed).fit_predict(X_scaled)
        rows.append((silhouette_score(X_scaled, labels), -davies_bouldin_score(X_scaled, labels), k))
    return rows


def fit_kmeans(X_scaled, k, seed=42):
    model = KMeans(n_clusters=k, init='k-means++', n_init=50, random_state=seed)
    labels = model.fit_predict(X_scaled)
    return labels, model.cluster_centers_


def adjacency_matrix(X_scaled):
    """Ячейки 24, 49, 51: гауссово ядро с σ = медиане попарных расстояний."""
    sigma = np.median(pdist(X_scaled, metric='euclidean'))
    A = -np.square(X_scaled[:, None] - X_scaled[None, :]).sum(axis=-1)
    A = np.exp(A / (2 * sigma ** 2))
    np.fill_diagonal(A, 0)
    return A, float(sigma)


def laplacian_eigenvalues(A):
    """Ячейки 27–28."""
    D = np.diag(np.sum(A, axis=1))
    L = np.eye(A.shape[0]) - np.linalg.inv(np.sqrt(D)) @ A @ np.linalg.inv(np.sqrt(D))
    eigenvalues, _ = eigh(L)
    ev = eigenvalues[np.argsort(eigenvalues)]
    gaps = np.diff(ev[1:11])
    eigengap_k = int(np.argmax(gaps) + 2) if len(gaps) else None
    return ev, gaps, eigengap_k


SPECTRAL_GAMMAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def spectral_candidates(X_scaled, gammas=SPECTRAL_GAMMAS, k_min=2, k_max=8, seed=42):
    """Ячейка 29."""
    rows = []
    for gamma in gammas:
        for k in _k_range(len(X_scaled), k_min, k_max):
            try:
                labels = SpectralClustering(n_clusters=k, affinity='rbf', gamma=gamma,
                                            assign_labels='kmeans', n_init=30,
                                            random_state=seed).fit_predict(X_scaled)
                if len(np.unique(labels)) < 2:
                    continue
                rows.append((silhouette_score(X_scaled, labels), -davies_bouldin_score(X_scaled, labels),
                             k, gamma))
            except ValueError:
                pass
    return rows


def fit_spectral(X_scaled, k, gamma, seed=42):
    return SpectralClustering(n_clusters=k, affinity='rbf', gamma=gamma, assign_labels='kmeans',
                              n_init=30, random_state=seed).fit_predict(X_scaled)


DBSCAN_EPS_GRID = tuple(float(v) for v in np.round(np.linspace(0.15, 1.50, 28), 2))


def dbscan_candidates(X_scaled, eps_grid=DBSCAN_EPS_GRID, min_samples_range=range(3, 9),
                      max_noise=0.45, noise_penalty=0.20):
    """Ячейка 33."""
    rows = []
    for eps in eps_grid:
        for min_samples in min_samples_range:
            labels = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(X_scaled)
            mask = labels != -1
            unique = np.unique(labels[mask])
            if mask.sum() < 3 or len(unique) < 2 or (1 - mask.mean()) > max_noise:
                continue
            counts = pd.Series(labels[mask]).value_counts()
            if counts.min() < 2:
                continue
            silhouette = silhouette_score(X_scaled[mask], labels[mask])
            noise_fraction = 1 - mask.mean()
            rows.append((silhouette - noise_penalty * noise_fraction, silhouette, -noise_fraction,
                         eps, min_samples))
    return rows


def fit_dbscan(X_scaled, eps, min_samples):
    return DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(X_scaled).labels_


def clustering_metrics(X_scaled, method, labels, parameters, noise=False):
    """Ячейка 37 (+ число кластеров и шумовых точек)."""
    labels = np.asarray(labels)
    mask = labels != -1 if noise else np.ones(len(labels), dtype=bool)
    n_clusters = len(np.unique(labels[mask]))
    row = {'Method': method, 'Parameters': parameters, 'Clusters': n_clusters,
           'Noise_points': int((~mask).sum()), 'Noise_fraction': 1 - mask.mean()}
    if n_clusters >= 2 and n_clusters < mask.sum():
        row.update({'Silhouette': silhouette_score(X_scaled[mask], labels[mask]),
                    'Calinski-Harabasz': calinski_harabasz_score(X_scaled[mask], labels[mask]),
                    'Davies-Bouldin': davies_bouldin_score(X_scaled[mask], labels[mask])})
    else:
        row.update({'Silhouette': np.nan, 'Calinski-Harabasz': np.nan, 'Davies-Bouldin': np.nan})
    return row


def cluster_df(names, x_values, y_values, labels, x_label, y_label):
    """Ячейка 16 (подписи столбцов берутся из метаданных, а не зашиты в код)."""
    labels = np.asarray(labels)
    cluster_names = np.where(labels == -1, 'Noise', (labels + 1).astype(str))
    return pd.DataFrame({
        NAME: list(names),
        'Cluster': cluster_names,
        x_label: np.round(np.asarray(x_values, dtype=float), 2),
        y_label: np.round(np.asarray(y_values, dtype=float), 3),
    })


def ari_matrix(label_sets: dict) -> pd.DataFrame:
    """НОВОЕ: попарное сходство разбиений (Adjusted Rand Index)."""
    names = list(label_sets)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            M.loc[a, b] = adjusted_rand_score(label_sets[a], label_sets[b])
    return M


def kruskal_by_cluster(y, labels):
    """НОВОЕ: медианы сорбции по кластерам и критерий Краскела–Уоллиса (шум не учитывается)."""
    y = np.asarray(y, dtype=float)
    labels = np.asarray(labels)
    groups, rows = [], []
    for c in sorted(set(labels.tolist())):
        if c == -1:
            continue
        vals = y[labels == c]
        groups.append(vals)
        rows.append({'Cluster': str(c + 1), 'n': len(vals), 'Median': float(np.median(vals)),
                     'Min': float(vals.min()), 'Max': float(vals.max())})
    p = np.nan
    H = np.nan
    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
        try:
            H, p = stats.kruskal(*groups)
        except ValueError:
            pass
    return pd.DataFrame(rows), float(H), float(p)


def contingency_test(a, b, n_mc=10000, seed=42):
    """НОВОЕ: таблица сопряжённости, χ² и точный критерий Фишера.
    Для 2×2 — scipy.fisher_exact; для больших таблиц — Монте-Карло-версия критерия
    Фишера–Фримена–Холтона (перестановки при фиксированных маргиналах)."""
    table = pd.crosstab(pd.Series(a, name='a'), pd.Series(b, name='b'))
    out = {'table': table, 'chi2': np.nan, 'chi2_p': np.nan, 'fisher_p': np.nan, 'fisher_kind': ''}
    if table.shape[0] < 2 or table.shape[1] < 2:
        return out
    chi2, p, _, _ = stats.chi2_contingency(table.values, correction=False)
    out['chi2'], out['chi2_p'] = float(chi2), float(p)
    if table.shape == (2, 2):
        out['fisher_p'] = float(stats.fisher_exact(table.values)[1])
        out['fisher_kind'] = 'точный (2×2)'
    else:
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        observed = _table_logprob(table.values)
        rng = np.random.default_rng(seed)
        count = 0
        rows_index = pd.Index(table.index)
        cols_index = pd.Index(table.columns)
        for _ in range(n_mc):
            perm = rng.permutation(b_arr)
            t = np.zeros(table.shape)
            np.add.at(t, (rows_index.get_indexer(a_arr), cols_index.get_indexer(perm)), 1)
            if _table_logprob(t) <= observed + 1e-9:
                count += 1
        out['fisher_p'] = (count + 1) / (n_mc + 1)
        out['fisher_kind'] = f'Монте-Карло, {n_mc} перестановок'
    return out


def _table_logprob(t):
    t = np.asarray(t, dtype=float)
    r, c, n = t.sum(1), t.sum(0), t.sum()
    lg = lambda x: np.sum([math.lgamma(v + 1) for v in np.ravel(x)])
    return lg(r) + lg(c) - math.lgamma(n + 1) - lg(t)


# ---------------------------------------------------------------------------
# RANSAC (ячейки 40, 43)
# ---------------------------------------------------------------------------
TAU_GRID_MMOL_G = np.array([0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50,
                            2.75, 3.00, 4.00, 5.00, 6.00])


def format_tau(value):
    return f'{float(value):.2f}'.rstrip('0').rstrip('.')


def format_tau_units(value, factor=1.0):
    """Формат τ с учётом масштаба сетки (для г/г сетка ×0.1 → нужен лишний знак)."""
    if factor >= 1:
        return format_tau(value)
    return f'{float(value):.3f}'.rstrip('0').rstrip('.')


def loo_q2(X, y):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    if len(y) <= X.shape[1] + 1:
        return -np.inf
    predictions = np.empty(len(y), dtype=float)
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        model = LinearRegression().fit(X[train], y[train])
        predictions[i] = model.predict(X[i:i + 1])[0]
    denominator = np.sum((y - y.mean()) ** 2)
    if denominator <= np.finfo(float).eps:
        return -np.inf
    return float(1 - np.sum((y - predictions) ** 2) / denominator)


def stabilize_ransac_model(model, X, y, threshold, max_iterations=100):
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    mask = np.asarray(model.inlier_mask_, dtype=bool).copy()
    seen = set()
    for _ in range(max_iterations):
        signature = mask.tobytes()
        if signature in seen or mask.sum() <= X_arr.shape[1]:
            return None
        seen.add(signature)
        estimator = LinearRegression().fit(X_arr[mask], y_arr[mask])
        updated_mask = np.abs(y_arr - estimator.predict(X_arr)) <= float(threshold)
        if np.array_equal(updated_mask, mask):
            model.estimator_ = estimator
            model.inlier_mask_ = mask
            return model
        mask = updated_mask
    return None


def optimize_ransac_hyperparameters(
    X, y, thresholds=None, min_samples=None, min_inliers=None,
    min_train_r2=0.60, min_cv_r2=0.50, max_trials=2000,
    stop_probability=0.999, require_positive=False, random_state=42, tau_grid=None
):
    """Ячейка 40 без изменений; tau_grid — сетка τ (по умолчанию TAU_GRID_MMOL_G)."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
    X_arr = X_arr[valid]
    y_arr = y_arr[valid]
    n_samples, n_features = X_arr.shape
    min_inliers = max(n_features + 3, int(np.ceil(0.20 * n_samples))) if min_inliers is None else int(min_inliers)
    if n_samples < min_inliers:
        return None
    default_grid = TAU_GRID_MMOL_G if tau_grid is None else np.asarray(tau_grid, dtype=float)
    threshold_grid = default_grid if thresholds is None else np.atleast_1d(thresholds).astype(float)
    if min_samples is None:
        min_samples_grid = range(n_features + 1, n_features + 4)
    else:
        min_samples_grid = [int(min_samples)]
    best = None
    for current_min_samples in min_samples_grid:
        if current_min_samples >= n_samples:
            continue
        for current_threshold in threshold_grid:
            try:
                model = RANSACRegressor(
                    estimator=LinearRegression(),
                    min_samples=current_min_samples,
                    residual_threshold=float(current_threshold),
                    max_trials=max_trials,
                    stop_probability=stop_probability,
                    loss='absolute_error',
                    random_state=random_state
                ).fit(X_arr, y_arr)
            except ValueError:
                continue
            model = stabilize_ransac_model(model, X_arr, y_arr, float(current_threshold))
            if model is None:
                continue
            mask = model.inlier_mask_
            n_inliers = int(mask.sum())
            if n_inliers < min_inliers:
                continue
            coefficients = np.asarray(model.estimator_.coef_, dtype=float)
            if require_positive and (coefficients.size != 1 or coefficients[0] <= 0):
                continue
            r2 = float(model.estimator_.score(X_arr[mask], y_arr[mask]))
            q2 = loo_q2(X_arr[mask], y_arr[mask])
            if not np.isfinite(q2) or r2 < min_train_r2 or q2 < min_cv_r2:
                continue
            coverage = n_inliers / n_samples
            score = q2 * coverage
            ranking = (score, coverage, q2, r2, -float(current_threshold), -current_min_samples)
            if best is None or ranking > best['ranking']:
                best = {
                    'model': model,
                    'threshold': float(current_threshold),
                    'min_samples': int(current_min_samples),
                    'min_inliers': min_inliers,
                    'n_inliers': n_inliers,
                    'coverage': coverage,
                    'r2': r2,
                    'q2': q2,
                    'score': score,
                    'ranking': ranking
                }
    return best


def sequential_ransac_fit(
    X, y, threshold=None, n_iterations=5, min_r2=0.50, max_trials=20000,
    stop_probability=0.999, min_samples=None, min_cv_r2=0.50, min_inliers=4,
    require_positive=False, random_state=42, tau_grid=None
):
    """Ячейка 43 (sequential_ransac) — расчётная часть без рисования.
    Возвращает список прямых с теми же полями, что и в ноутбуке, плюс 'ranks'."""
    X_arr = np.asarray(X, dtype=float).reshape(-1, 1)
    y_arr = np.asarray(y, dtype=float).ravel()
    remaining_idx = np.arange(len(y_arr))
    models = []
    for iteration in range(n_iterations):
        if len(remaining_idx) < min_inliers:
            break
        tuned = optimize_ransac_hyperparameters(
            X_arr[remaining_idx], y_arr[remaining_idx],
            thresholds=None if threshold is None else [threshold],
            min_samples=min_samples, min_inliers=min_inliers,
            min_train_r2=min_r2, min_cv_r2=min_cv_r2,
            max_trials=max_trials, stop_probability=stop_probability,
            require_positive=require_positive, random_state=random_state, tau_grid=tau_grid
        )
        if tuned is None:
            break
        model = tuned['model']
        inliers_idx = remaining_idx[model.inlier_mask_]
        models.append({
            'model': model,
            'inliers_idx': inliers_idx.copy(),
            'r2': tuned['r2'],
            'q2': tuned['q2'],
            'score': tuned['score'],
            'slope': float(model.estimator_.coef_[0]),
            'intercept': float(model.estimator_.intercept_),
            'threshold': tuned['threshold'],
            'min_samples': tuned['min_samples'],
            'min_inliers': tuned['min_inliers']
        })
        remaining_idx = remaining_idx[~model.inlier_mask_]
    all_inliers = sorted({int(i) for m in models for i in m['inliers_idx']},
                         key=lambda index: (y_arr[index], index))
    rank_map = {index: rank for rank, index in enumerate(all_inliers, start=1)}
    for m in models:
        m['ranks'] = [rank_map[int(i)] for i in m['inliers_idx']]
    return models, rank_map


def descriptor_tables(data, descriptor, models, ycol=SORPTION_MMOL):
    """Ячейка 43 (descriptor_tables)."""
    inlier_rows = []
    inlier_indices = set()
    for line_no, model in enumerate(models, start=1):
        q_corr = model['r2'] * len(model['inliers_idx']) / len(data)
        for index, rank in zip(model['inliers_idx'], model['ranks']):
            inlier_indices.add(int(index))
            inlier_rows.append({
                'Line': line_no,
                'Plot_Number': int(rank),
                'Name': data.iloc[index][NAME],
                'Sorption': data.iloc[index][ycol],
                descriptor: data.iloc[index][descriptor],
                'Slope': round(model['slope'], 3),
                'Intercept': round(model['intercept'], 3),
                'R2': round(model['r2'], 3),
                'Q2_LOO': round(model['q2'], 3),
                'n_inliers': len(model['inliers_idx']),
                'Inlier_ratio': round(len(model['inliers_idx']) / len(data), 3),
                'Residual_threshold': model['threshold'],
                'Min_samples': model['min_samples'],
                'Q_corr': round(q_corr, 3),
                'C_corr': round(1 - q_corr, 3)
            })
    outlier_indices = [i for i in range(len(data)) if i not in inlier_indices]
    outlier_indices = sorted(outlier_indices, key=lambda i: (data.iloc[i][ycol], i))
    outlier_rows = [{'Outlier_Number': rank, 'Name': data.iloc[i][NAME],
                     'Sorption': data.iloc[i][ycol], descriptor: data.iloc[i][descriptor]}
                    for rank, i in enumerate(outlier_indices, start=1)]
    inliers = (pd.DataFrame(inlier_rows).sort_values('Plot_Number').reset_index(drop=True)
               if inlier_rows else pd.DataFrame())
    outliers = pd.DataFrame(outlier_rows)
    return inliers, outliers


def descriptor_data(frame, descriptor, ycol=SORPTION_MMOL):
    """Ячейка 43 (analyze_descriptor): dropna и сортировка по сорбции (mergesort).
    Возвращает также имена веществ, выпавших из-за пустых значений."""
    data = frame[[NAME, ycol, descriptor]].copy()
    data[descriptor] = pd.to_numeric(data[descriptor].replace('—', np.nan), errors='coerce')
    data[ycol] = pd.to_numeric(data[ycol], errors='coerce')
    missing = data.loc[data[[ycol, descriptor]].isna().any(axis=1), NAME].astype(str).tolist()
    data = data.dropna().sort_values(ycol, kind='mergesort').reset_index(drop=True)
    return data, missing


def default_min_inliers(n, fraction=0.20, floor=4):
    return max(int(floor), int(np.ceil(fraction * n)))


def analyze_descriptor(frame, descriptor, ycol=SORPTION_MMOL, *, min_fraction=0.20, min_floor=4,
                       n_iterations=5, min_r2=0.50, min_cv_r2=0.50, max_trials=20000,
                       threshold=None, require_positive=False, seed=42, tau_grid=None):
    """Ячейка 43 (analyze_descriptor), расчётная часть."""
    data, missing = descriptor_data(frame, descriptor, ycol)
    min_inliers = default_min_inliers(len(data), min_fraction, min_floor)
    result = {'data': data, 'missing': missing, 'min_inliers': min_inliers, 'models': [],
              'error': None}
    if len(data) < min_inliers or len(data) < 3:
        result['error'] = (f'Для RANSAC по «{descriptor}» нужно не менее {min_inliers} веществ '
                           f'со значением, сейчас {len(data)}.')
        return result
    models, _ = sequential_ransac_fit(
        data[descriptor], data[ycol], threshold=threshold, n_iterations=n_iterations,
        min_r2=min_r2, max_trials=max_trials, min_cv_r2=min_cv_r2, min_inliers=min_inliers,
        require_positive=require_positive, random_state=seed, tau_grid=tau_grid)
    result['models'] = models
    result['inliers'], result['outliers'] = descriptor_tables(data, descriptor, models, ycol)
    return result


def analyze_cluster_groups(cluster_table, x_label, y_label, *, seed=42, tau_grid=None,
                           n_iterations=5, min_fraction=0.20, min_floor=4, min_r2=0.50,
                           min_cv_r2=0.50, max_trials=20000, require_positive=True):
    """Ячейка 43 (analyze_cluster_groups), расчётная часть: RANSAC ε → сорбция в каждом кластере."""
    clusters = [v for v in sorted(cluster_table['Cluster'].astype(str).unique()) if v != 'Noise']
    out = []
    for cluster in clusters:
        data = cluster_table.loc[cluster_table['Cluster'].astype(str) == cluster].copy()
        data = data.sort_values(y_label, kind='mergesort').reset_index(drop=True)
        min_inliers = default_min_inliers(len(data), min_fraction, min_floor)
        models = []
        if len(data) >= min_inliers:
            models, _ = sequential_ransac_fit(
                data[x_label], data[y_label], n_iterations=n_iterations, min_r2=min_r2,
                max_trials=max_trials, min_cv_r2=min_cv_r2, min_inliers=min_inliers,
                require_positive=require_positive, random_state=seed, tau_grid=tau_grid)
        report = data.rename(columns={y_label: '__y__'})
        inliers, outliers = descriptor_tables(report, x_label, models, ycol='__y__')
        if not inliers.empty:
            inliers.insert(0, 'Cluster', cluster)
        if not outliers.empty:
            outliers.insert(0, 'Cluster', cluster)
        out.append({'cluster': cluster, 'data': data, 'models': models,
                    'inliers': inliers, 'outliers': outliers})
    return out


def ransac_plane(frame, x_col, y_col, z_col=SORPTION_MMOL, min_inliers=5, seed=42, tau_grid=None,
                 min_train_r2=0.60, min_cv_r2=0.50, max_trials=2000):
    """Ячейка 62 (ransac_plane_labels), расчётная часть."""
    data = frame[[NAME, x_col, y_col, z_col]].dropna().sort_values(z_col, kind='mergesort').reset_index(drop=True)
    result = {'data': data, 'tuned': None, 'error': None}
    if len(data) < min_inliers:
        result['error'] = (f'Для плоскости {x_col} – {y_col} нужно не менее {min_inliers} веществ '
                           f'с обоими значениями, сейчас {len(data)}.')
        return result
    X_plane = data[[x_col, y_col]].to_numpy(float)
    z = data[z_col].to_numpy(float)
    tuned = optimize_ransac_hyperparameters(X_plane, z, min_inliers=min_inliers, random_state=seed,
                                            tau_grid=tau_grid, min_train_r2=min_train_r2,
                                            min_cv_r2=min_cv_r2, max_trials=max_trials)
    if tuned is None:
        result['error'] = f'Не найдено проверенной плоскости для {x_col} и {y_col}.'
        return result
    model = tuned['model']
    mask = model.inlier_mask_
    table = data.copy()
    table['Predicted_sorption'] = model.predict(X_plane)
    table['Residual'] = z - table['Predicted_sorption']
    inlier_table = table.loc[mask].copy().reset_index(drop=True)
    inlier_table.insert(0, 'Plot_Number', np.arange(1, len(inlier_table) + 1))
    outlier_table = table.loc[~mask].copy().sort_values(z_col, kind='mergesort').reset_index(drop=True)
    outlier_table.insert(0, 'Outlier_Number', np.arange(1, len(outlier_table) + 1))
    result.update({'tuned': tuned, 'inliers': inlier_table, 'outliers': outlier_table,
                   'coef': model.estimator_.coef_.tolist(),
                   'intercept': float(model.estimator_.intercept_)})
    return result


def line_confidence_band(x_in, y_in, x_line, level=0.95):
    """НОВОЕ: доверительная полоса средней прямой МНК по точкам прямой
    (аналитическая формула; seaborn.regplot оценивает то же самое бутстрепом)."""
    x_in = np.asarray(x_in, dtype=float).ravel()
    y_in = np.asarray(y_in, dtype=float).ravel()
    n = len(x_in)
    if n < 3:
        return None
    slope, intercept = np.polyfit(x_in, y_in, 1)
    resid = y_in - (slope * x_in + intercept)
    s = np.sqrt(np.sum(resid ** 2) / (n - 2))
    sxx = np.sum((x_in - x_in.mean()) ** 2)
    if sxx <= 0:
        return None
    t = stats.t.ppf(0.5 + level / 2, n - 2)
    x_line = np.asarray(x_line, dtype=float).ravel()
    half = t * s * np.sqrt(1 / n + (x_line - x_in.mean()) ** 2 / sxx)
    centre = slope * x_line + intercept
    return centre - half, centre + half


# ---------------------------------------------------------------------------
# Корреляции (НОВОЕ)
# ---------------------------------------------------------------------------
def correlation_table(frame, descriptors, ycol=SORPTION_MMOL):
    rows = []
    for d in descriptors:
        data, missing = descriptor_data(frame, d, ycol)
        row = {'Descriptor': d, 'n': len(data), 'Pearson_r': np.nan, 'Pearson_p': np.nan,
               'Spearman_rho': np.nan, 'Spearman_p': np.nan}
        if len(data) >= 3 and data[d].nunique() > 1 and data[ycol].nunique() > 1:
            r, p = stats.pearsonr(data[d], data[ycol])
            rho, ps = stats.spearmanr(data[d], data[ycol])
            row.update({'Pearson_r': r, 'Pearson_p': p, 'Spearman_rho': rho, 'Spearman_p': ps})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Коллинеарность (ячейка 80)
# ---------------------------------------------------------------------------
def collinearity(frame, predictors, ycol=SORPTION_MMOL):
    data = frame[[NAME, ycol] + list(predictors)].copy()
    for column in list(predictors) + [ycol]:
        data[column] = pd.to_numeric(data[column].replace('—', np.nan), errors='coerce')
    missing = data.loc[data.isna().any(axis=1), NAME].astype(str).tolist()
    data = data.dropna().reset_index(drop=True)
    out = {'data': data, 'missing': missing, 'error': None}
    if len(predictors) < 2 or len(data) < len(predictors) + 2:
        out['error'] = (f'Для коллинеарности нужно хотя бы 2 дескриптора и не меньше '
                        f'{len(predictors) + 2} веществ без пропусков, сейчас {len(data)}.')
        return out
    Xs = StandardScaler().fit_transform(data[list(predictors)].to_numpy(float))
    corr = data[list(predictors)].corr()
    vif_rows = []
    for i, d in enumerate(predictors):
        target = Xs[:, i]
        others = np.delete(Xs, i, axis=1)
        r2 = LinearRegression().fit(others, target).score(others, target)
        vif_rows.append({'Descriptor': d, 'VIF': np.inf if 1 - r2 <= 1e-12 else 1 / (1 - r2)})
    vif = pd.DataFrame(vif_rows).sort_values('VIF', ascending=False).reset_index(drop=True)
    pca = PCA().fit(Xs)
    comps = np.arange(1, len(predictors) + 1)
    loadings = pd.DataFrame(pca.components_.T, index=list(predictors), columns=[f'PC{i}' for i in comps])
    out.update({'corr': corr, 'vif': vif, 'evr': pca.explained_variance_ratio_, 'loadings': loadings})
    return out


# ---------------------------------------------------------------------------
# Перестановочные тесты (ячейка 82 + НОВОЕ: полный тест)
# ---------------------------------------------------------------------------
def fixed_sequential_score(X, y, fitted_models, min_inliers, seed, max_trials=300,
                           min_r2=0.60, min_cv_r2=0.50):
    """Ячейка 82 без изменений (пороги R² и Q² вынесены в аргументы, по умолчанию как в ноутбуке)."""
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    remaining = np.arange(len(y))
    total = len(y)
    score = 0.0
    for line, fitted in enumerate(fitted_models, start=1):
        if len(remaining) < max(min_inliers, fitted['min_samples'] + 1):
            break
        try:
            model = RANSACRegressor(
                estimator=LinearRegression(), min_samples=fitted['min_samples'],
                residual_threshold=fitted['threshold'], max_trials=max_trials,
                stop_probability=0.999, loss='absolute_error', random_state=seed + line
            ).fit(X[remaining], y[remaining])
        except ValueError:
            break
        model = stabilize_ransac_model(model, X[remaining], y[remaining], fitted['threshold'])
        if model is None:
            break
        mask = model.inlier_mask_
        if mask.sum() < min_inliers:
            break
        r2 = model.estimator_.score(X[remaining][mask], y[remaining][mask])
        q2 = loo_q2(X[remaining][mask], y[remaining][mask])
        if not np.isfinite(q2) or r2 < min_r2 or q2 < min_cv_r2:
            break
        score += q2 * mask.sum() / total
        remaining = remaining[~mask]
    return float(score)


def _full_sequential_score(X, y, settings, seed):
    """НОВОЕ: на перемешанной сорбции заново подбираются τ и min_samples (вся процедура)."""
    models, _ = sequential_ransac_fit(
        X, y, n_iterations=settings['n_iterations'], min_r2=settings['min_r2'],
        max_trials=settings['max_trials_null'], min_cv_r2=settings['min_cv_r2'],
        min_inliers=settings['min_inliers'], require_positive=settings['require_positive'],
        random_state=int(seed) % (2 ** 31), tau_grid=settings['tau_grid'], threshold=settings['threshold'])
    return float(sum(m['q2'] * len(m['inliers_idx']) / len(y) for m in models))


def permutation_data(frame, descriptor, ycol=SORPTION_MMOL):
    """Ячейка 82: dropna в исходном порядке строк (без сортировки)."""
    data = frame[[descriptor, ycol]].copy()
    data[descriptor] = pd.to_numeric(data[descriptor].replace('—', np.nan), errors='coerce')
    data = data.dropna().reset_index(drop=True)
    return data[descriptor].to_numpy(float), data[ycol].to_numpy(float)


def _run_parallel(func, seeds, progress: Callable | None, n_jobs=-1):
    total = len(seeds)
    results = np.empty(total)
    gen = Parallel(n_jobs=n_jobs, return_as='generator')(delayed(func)(int(s)) for s in seeds)
    for i, value in enumerate(gen):
        results[i] = value
        if progress is not None and (i % 10 == 0 or i == total - 1):
            progress((i + 1) / total)
    return results


def permutation_test(frame, descriptor, fitted_models, *, ycol=SORPTION_MMOL, n_permutations=1000,
                     seed=42, mode='conditional', settings=None, progress=None, n_jobs=-1,
                     max_trials_null=300, min_fraction=0.20, min_floor=4):
    """Ячейка 82 (mode='conditional') и НОВОЕ полное повторение подбора (mode='full')."""
    X_d, y_s = permutation_data(frame, descriptor, ycol)
    n = len(y_s)
    observed = sum(m['q2'] * len(m['inliers_idx']) / n for m in fitted_models)
    min_inliers = default_min_inliers(n, min_fraction, min_floor)
    seeds = np.random.default_rng(seed).integers(0, 2 ** 32 - 1, size=n_permutations)
    if mode == 'conditional':
        def one(s):
            return fixed_sequential_score(X_d, np.random.default_rng(s).permutation(y_s),
                                          fitted_models, min_inliers, s, max_trials=max_trials_null)
    else:
        st = dict(settings or {})
        st.setdefault('min_inliers', min_inliers)
        st['max_trials_null'] = max_trials_null

        def one(s):
            return _full_sequential_score(X_d, np.random.default_rng(s).permutation(y_s), st, s)
    null = _run_parallel(one, seeds, progress, n_jobs)
    p_value = (1 + np.sum(null >= observed)) / (n_permutations + 1)
    return {'observed': float(observed), 'null': null, 'p': float(p_value), 'n': n}


def benjamini_hochberg(pvalues):
    """НОВОЕ: поправка Бенджамини–Хохберга на множественные сравнения."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


# ---------------------------------------------------------------------------
# Вложенная LOO-валидация PLS (ячейка 84) + НОВОЕ: y-рандомизация
# ---------------------------------------------------------------------------
def nested_loo_pls(X, y, seed=42, n_jobs=-1):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_pred = X.shape[1]
    predictions = np.empty(len(y))
    selected = np.empty(len(y), dtype=int)
    for fold, (train_index, test_index) in enumerate(LeaveOneOut().split(X)):
        maximum_components = min(n_pred, len(train_index) - 1)
        inner_cv = KFold(n_splits=min(5, len(train_index)), shuffle=True, random_state=seed + fold)
        pipeline = Pipeline([('scale', StandardScaler()),
                             ('pls', PLSRegression(scale=False, max_iter=2000))])
        search = GridSearchCV(pipeline, {'pls__n_components': list(range(1, maximum_components + 1))},
                              scoring='neg_mean_squared_error', cv=inner_cv, n_jobs=n_jobs)
        search.fit(X[train_index], y[train_index])
        predictions[test_index[0]] = np.ravel(search.predict(X[test_index]))[0]
        selected[test_index[0]] = search.best_params_['pls__n_components']
    residuals = y - predictions
    q2 = 1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)
    return {'predictions': predictions, 'components': selected, 'residuals': residuals,
            'q2': float(q2), 'rmsecv': float(np.sqrt(np.mean(residuals ** 2))),
            'maecv': float(np.mean(np.abs(residuals)))}


def pls_validation(frame, predictors, ycol=SORPTION_MMOL, seed=42):
    col = collinearity(frame, predictors, ycol)
    data = col['data']
    if len(predictors) < 1 or len(data) < max(6, len(predictors) + 2):
        return {'error': (f'Для PLS нужно не меньше {max(6, len(predictors) + 2)} веществ без '
                          f'пропусков в выбранных дескрипторах, сейчас {len(data)}.'),
                'data': data, 'missing': col['missing']}
    res = nested_loo_pls(data[list(predictors)].to_numpy(float), data[ycol].to_numpy(float), seed)
    table = data[[NAME, ycol] + list(predictors)].copy()
    table['Predicted_sorption'] = res['predictions']
    table['Residual'] = res['residuals']
    table['Selected_PLS_components'] = res['components']
    res.update({'data': data, 'table': table, 'missing': col['missing'], 'error': None,
                'y': data[ycol].to_numpy(float)})
    return res


def y_randomization(frame, predictors, ycol=SORPTION_MMOL, n_permutations=100, seed=42,
                    progress=None, n_jobs=-1):
    """НОВОЕ: перемешиваем сорбцию и повторяем вложенную LOO-валидацию PLS целиком."""
    col = collinearity(frame, predictors, ycol)
    data = col['data']
    X = data[list(predictors)].to_numpy(float)
    y = data[ycol].to_numpy(float)
    seeds = np.random.default_rng(seed).integers(0, 2 ** 32 - 1, size=n_permutations)

    def one(s):
        return nested_loo_pls(X, np.random.default_rng(s).permutation(y), seed=seed, n_jobs=1)['q2']
    return _run_parallel(one, seeds, progress, n_jobs)
