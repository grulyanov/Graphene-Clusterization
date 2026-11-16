"""Сверка analysis.py с эталоном ноутбука (dev/reference.json).

Запуск из корня проекта:  python dev/verify.py [число_перестановок]
Печатает таблицу «ноутбук vs сайт» и сохраняет её в dev/verify_table.md.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A  # noqa: E402

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
ref = json.load(open(ROOT / 'dev' / 'reference.json'))
rows = []


def add(what, nb, site, tol=1e-9):
    if isinstance(nb, (list, tuple)) or isinstance(site, (list, tuple)):
        ok = list(nb) == list(site)
        nb_s, site_s = str(nb), str(site)
    elif isinstance(nb, str):
        ok = nb == site
        nb_s, site_s = nb, str(site)
    else:
        ok = abs(float(nb) - float(site)) <= tol
        nb_s, site_s = f'{float(nb):.6g}', f'{float(site):.6g}'
    rows.append((what, nb_s if len(nb_s) < 70 else nb_s[:67] + '…',
                 site_s if len(site_s) < 70 else site_s[:67] + '…', '✓' if ok else '✗'))


df = pd.read_excel(ROOT / 'Graphene.xlsx', sheet_name='Epsilon')
df.columns = df.columns.astype(str).str.strip()
df = df.loc[~A.notebook_excluded_mask(df['Name'])].reset_index(drop=True)
add('Веществ после исключения', ref['n_rows'], len(df))

Y = A.SORPTION_MMOL
X = df[[Y, 'ε']].to_numpy(float)
Xs = A.scale(X)

_, _, k_km = max(A.kmeans_candidates(Xs))
y_km, _ = A.fit_kmeans(Xs, k_km)
add('K-Means: k', ref['kmeans']['k'], k_km)
add('K-Means: метки', ref['kmeans']['labels'], y_km.tolist())

_, _, k_hc, link = max(A.hierarchical_candidates(Xs))
y_hc = A.fit_hierarchical(Xs, k_hc, link)
add('Иерархическая: k', ref['hc']['k'], k_hc)
add('Иерархическая: linkage', ref['hc']['linkage'], link)
add('Иерархическая: метки', ref['hc']['labels'], y_hc.tolist())

_, _, k_sp, gamma = max(A.spectral_candidates(Xs))
y_sp = A.fit_spectral(Xs, k_sp, gamma)
add('Спектральная: k', ref['spectral']['k'], k_sp)
add('Спектральная: γ', ref['spectral']['gamma'], gamma)
add('Спектральная: метки', ref['spectral']['labels'], y_sp.tolist())
Aff, _ = A.adjacency_matrix(Xs)
_, _, egk = A.laplacian_eigenvalues(Aff)
add('Eigengap: k', ref['spectral']['eigengap_k'], egk)

_, _, _, eps, ms = max(A.dbscan_candidates(Xs))
y_db = A.fit_dbscan(Xs, eps, ms)
add('DBSCAN: eps', ref['dbscan']['eps'], eps)
add('DBSCAN: min_samples', ref['dbscan']['min_samples'], ms)
add('DBSCAN: метки', ref['dbscan']['labels'], y_db.tolist())

site_metrics = [
    A.clustering_metrics(Xs, 'K-Means', y_km, ''),
    A.clustering_metrics(Xs, 'Hierarchical', y_hc, ''),
    A.clustering_metrics(Xs, 'Spectral', y_sp, ''),
    A.clustering_metrics(Xs, 'DBSCAN', y_db, '', noise=True),
]
for r_nb, r_site in zip(ref['metrics'], site_metrics):
    for key in ('Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Noise_fraction'):
        add(f"Метрика {r_nb['Method']}: {key}", r_nb[key], r_site[key])

# Последовательный RANSAC по дескрипторам (require_positive как в ноутбуке)
positive = {'α', 'β', 'π*', 'DN'}
t0 = time.time()
site_models = {}
for desc, ref_models in ref['ransac'].items():
    res = A.analyze_descriptor(df, desc, Y, require_positive=desc in positive)
    site_models[desc] = res['models']
    add(f'RANSAC {desc}: число прямых', len(ref_models), len(res['models']))
    for i, (m_nb, m_site) in enumerate(zip(ref_models, res['models']), start=1):
        for key in ('slope', 'intercept', 'r2', 'q2', 'threshold', 'min_samples'):
            add(f'RANSAC {desc}, прямая {i}: {key}', m_nb[key], m_site[key])
        add(f'RANSAC {desc}, прямая {i}: inliers', m_nb['inliers_idx'], m_site['inliers_idx'].tolist())
        add(f'RANSAC {desc}, прямая {i}: номера точек', m_nb['ranks'], m_site['ranks'])
print(f'RANSAC по дескрипторам: {time.time() - t0:.1f} c')

# RANSAC внутри кластеров
for method, labels in [('K-Means', y_km), ('Hierarchical', y_hc), ('Spectral', y_sp), ('DBSCAN', y_db)]:
    table = A.cluster_df(df['Name'], df['ε'], df[Y], labels, 'ε (Dielectric constant)', 'Sorption (mmol/g)')
    table = table.sort_values('Cluster').reset_index(drop=True)
    groups = A.analyze_cluster_groups(table, 'ε (Dielectric constant)', 'Sorption (mmol/g)')
    site = pd.concat([g['inliers'] for g in groups if not g['inliers'].empty], ignore_index=True)
    nb = pd.DataFrame(ref['cluster_ransac'][method])
    add(f'Кластерный RANSAC {method}: строк inliers', len(nb), len(site))
    if len(nb) == len(site):
        cols = ['Cluster', 'Line', 'Plot_Number', 'Name', 'Slope', 'Intercept', 'R2', 'Q2_LOO',
                'Residual_threshold', 'Min_samples']
        same = (nb[cols].astype(str).values == site[cols].astype(str).values).all()
        add(f'Кластерный RANSAC {method}: таблица совпадает', 'True', str(bool(same)))

# Плоскости (полные случаи по первым 7 столбцам = Name…DN)
df_sorted = df.iloc[:, :7].dropna(axis=0)
for key, nb in ref['planes'].items():
    a, b = key.split('|')
    res = A.ransac_plane(df_sorted, a, b, Y)
    add(f'Плоскость {key}: n inliers', nb['n_in'], int(res['tuned']['model'].inlier_mask_.sum()))
    add(f'Плоскость {key}: коэф. 1', nb['coef'][0], res['coef'][0])
    add(f'Плоскость {key}: коэф. 2', nb['coef'][1], res['coef'][1])
    add(f'Плоскость {key}: свободный член', nb['intercept'], res['intercept'])

KT = ['ε', 'α', 'β', 'π*', 'DN']
col = A.collinearity(df, KT, Y)
for r in ref['vif']:
    site = col['vif'].set_index('Descriptor').loc[r['Descriptor'], 'VIF']
    add(f"VIF {r['Descriptor']}", r['VIF'], site)
for i, v in enumerate(ref['pca_evr'], start=1):
    add(f'PCA: доля дисперсии PC{i}', v, col['evr'][i - 1])

pls = A.pls_validation(df, KT, Y)
add('PLS: Q² (вложенная LOO)', ref['pls']['q2'], pls['q2'])
add('PLS: RMSECV', ref['pls']['rmsecv'], pls['rmsecv'])
add('PLS: MAECV', ref['pls']['maecv'], pls['maecv'])
add('PLS: n', ref['pls']['n'], len(pls['y']))
add('PLS: выбранные компоненты', ref['pls']['components'], pls['components'].tolist())

model_map = {'α': site_models['α'], 'β': site_models['β'], 'π*': site_models['π*'], 'DN': site_models['DN']}
t0 = time.time()
for r in ref['permutation']:
    res = A.permutation_test(df, r['Descriptor'], model_map[r['Descriptor']], ycol=Y,
                             n_permutations=N_PERM, seed=42)
    tol = 1e-9 if N_PERM == 1000 else 1.0
    add(f"Перестановки {r['Descriptor']}: наблюдаемое S_corr", r['Observed_S_corr'], res['observed'])
    add(f"Перестановки {r['Descriptor']}: p ({N_PERM})", r['Permutation_p'], res['p'], tol)
    add(f"Перестановки {r['Descriptor']}: среднее нулевого", r['Null_mean'], res['null'].mean(), tol)
print(f'Перестановки: {time.time() - t0:.1f} c')

table = pd.DataFrame(rows, columns=['Что сравниваем', 'Ноутбук', 'Сайт', 'Совпало'])
md = '| ' + ' | '.join(table.columns) + ' |\n|' + '---|' * len(table.columns) + '\n' + '\n'.join('| ' + ' | '.join(map(str, r)) + ' |' for r in table.values)
(ROOT / 'dev' / 'verify_table.md').write_text(md, encoding='utf-8')
bad = table[table['Совпало'] != '✓']
print(f'Всего проверок: {len(table)}, не совпало: {len(bad)}')
if len(bad):
    print(bad.to_string())
