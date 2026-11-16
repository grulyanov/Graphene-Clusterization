"""Рисует графики сайта на наборе «Как в ноутбуке filtered» и кладёт их в check_style/
рядом с графиками ноутбука: nb_cellNN_1.png (ноутбук) ↔ site_cellNN.png (сайт).

Запуск из корня проекта:  python dev/style_check.py [число_перестановок]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A  # noqa: E402
import plots as P  # noqa: E402

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = ROOT / 'check_style'
OUT.mkdir(exist_ok=True)


def save(fig, cell):
    fig.savefig(OUT / f'site_cell{cell:02d}.png', dpi=150, bbox_inches='tight')


df = pd.read_excel(ROOT / 'Graphene.xlsx', sheet_name='Epsilon')
df.columns = df.columns.astype(str).str.strip()
df = df.loc[~A.notebook_excluded_mask(df['Name'])].reset_index(drop=True)
Y = A.SORPTION_MMOL
X = df[[Y, 'ε']].to_numpy(float)
Xs = A.scale(X)
XL, YL = 'Sorption (mmol/g)', 'ε (Dielectric constant)'

save(P.raw_scatter(df[Y], df['ε'], YL, 'Dielectric constant', 'Sorption, mmol/g'), 8)
save(P.elbow(*A.elbow_wcss(Xs)), 11)
_, _, k_hc, link = max(A.hierarchical_candidates(Xs))
Z, leaves = A.dendrogram_linkage(df['Name'], X, link)
save(P.dendrogram(Z, leaves, A.dendrogram_threshold(Z, k_hc, len(leaves))), 12)
_, _, k_km = max(A.kmeans_candidates(Xs))
y_km, centers = A.fit_kmeans(Xs, k_km)
save(P.cluster_scatter(X, y_km, 'K-Means', XL, YL,
                       centers=StandardScaler().fit(X).inverse_transform(centers)), 15)
y_hc = A.fit_hierarchical(Xs, k_hc, link)
save(P.cluster_scatter(X, y_hc, 'Hierarchical', XL, YL), 20)
Aff, _ = A.adjacency_matrix(Xs)
save(P.adjacency_hist(Aff), 26)
ev, _, _ = A.laplacian_eigenvalues(Aff)
save(P.laplacian(ev), 27)
_, _, k_sp, gamma = max(A.spectral_candidates(Xs))
y_sp = A.fit_spectral(Xs, k_sp, gamma)
save(P.cluster_scatter(X, y_sp, 'Spectral Clustering', XL, YL), 30)
_, _, _, eps, ms = max(A.dbscan_candidates(Xs))
y_db = A.fit_dbscan(Xs, eps, ms)
save(P.cluster_scatter(X, y_db, 'DBSCAN', XL, YL), 34)

for cell, (method, labels) in zip((44, 45, 46, 47), [('K-Means', y_km), ('Hierarchical', y_hc),
                                                     ('Spectral', y_sp), ('DBSCAN', y_db)]):
    table = A.cluster_df(df['Name'], df['ε'], df[Y], labels, YL, XL).sort_values('Cluster').reset_index(drop=True)
    groups = A.analyze_cluster_groups(table, YL, XL)
    save(P.cluster_ransac(groups, method, YL, XL), cell)

save(P.adjacency_hist(A.adjacency_matrix(StandardScaler().fit_transform(X))[0],
                      'Adjacency Matrix Value Distribution tuned by median heuristic (Standart Scaler)',
                      (15, 5)), 50)
save(P.adjacency_hist(A.adjacency_matrix(RobustScaler().fit_transform(X))[0],
                      'Adjacency Matrix Value Distribution tuned by median heuristic (Robust Scaler)',
                      (15, 5)), 52)

res = {d: A.analyze_descriptor(df, d, Y, require_positive=True) for d in ('β', 'π*')}
save(P.descriptor_pair(res, ['β', 'π*'], XL, 'Dependence of sorption on Kamlet-Taft parameters'), 57)
labels = {'α': 'Kamlet-Taft α', 'DN': 'Donor Number (DN)', 'PA, kJ/mol': 'Proton Affinity (kJ/mol)'}
models = {}
for cell, d, pos in [(59, 'α', True), (60, 'DN', True), (70, 'PA, kJ/mol', False)]:
    r = A.analyze_descriptor(df, d, Y, require_positive=pos)
    models[d] = r['models']
    save(P.descriptor_ransac(r['data'], d, labels[d], r['models'], XL), cell)
models.update({d: res[d]['models'] for d in res})
extra = [(72, 'TopoPSA 2D (RDKit, N/O), Å²', 'TopoPSA 2D (Å²)'),
         (73, 'TPSA 3D (Mordred CPSA), Å²', 'TPSA 3D (Mordred CPSA, Å²)'),
         (75, 'σEB, Å²', 'Molecular cross-section σEB (Å²)'),
         (76, 'σW, Å²', 'Molecular cross-section σW (Å²)'),
         (78, 'Polar SASA 3D (Shrake–Rupley, N/O + H), Å²', 'Polar SASA 3D (Å²)')]
for cell, d, lab in extra:
    r = A.analyze_descriptor(df, d, Y)
    save(P.descriptor_ransac(r['data'], d, lab, r['models'], XL), cell)

df_sorted = df.iloc[:, :7].dropna(axis=0)
for cell, (a, b) in zip((63, 64, 65), [('β', 'π*'), ('β', 'DN'), ('π*', 'DN')]):
    save(P.ransac_plane(A.ransac_plane(df_sorted, a, b, Y), a, b, Y), cell)

col = A.collinearity(df, ['ε', 'α', 'β', 'π*', 'DN'], Y)
save(P.collinearity(col['corr'], col['evr']), 80)
perm = {d: A.permutation_test(df, d, models[d], ycol=Y, n_permutations=N_PERM)
        for d in ('α', 'β', 'π*', 'DN')}
save(P.permutation_hists(perm), 82)
pls = A.pls_validation(df, ['ε', 'α', 'β', 'π*', 'DN'], Y)
save(P.pls_observed_predicted(pls['y'], pls['predictions'], pls['q2'], pls['rmsecv']), 84)
print('saved', len(list(OUT.glob('site_cell*.png'))), 'site figures')
