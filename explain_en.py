"""Short explanations for the English site (same keys as explain.py)."""

T = {
    # general
    'site': 'Reproduces the analysis of `Graphene_unified_analysis_filtered.ipynb` on the current dataset. '
            'Any edit on **Data** or any setting recomputes every plot and table. Permutation tests run on demand.',
    'seed': 'Fixes all random steps (k-means starts, RANSAC subsets, permutations). Same seed → identical results. '
            'Notebook: 42.',
    'units': 'mmol/g as in Excel. g/g = mmol/g × M / 1000 (M from SMILES). q·Vm = mmol/g × M / ρ / 1000, cm³/g '
             '(needs density for all included substances). For g/g and q·Vm the τ grid is scaled by 0.1.',
    'session': 'Changes live only in this browser tab. Use **Download dataset** to keep them and **Load dataset** '
               'to restore.',
    'chem_class': 'Your grouping of substances, used to include/exclude groups in one click. It does not enter '
                  'any calculation.',
    'ir_class': 'IR response: 1 — new bands (proton transfer / reaction); 2 — band shifts only (H-bonding); '
                '3 — no response. Not in the Excel file; fill it in to enable the IR contingency tables.',
    'positive': 'If yes, RANSAC accepts only lines with a positive slope (notebook: α, β, π*, DN).',
    'missing': 'An empty cell only removes that substance from analyses of that descriptor (as `dropna`). '
               'Each plot lists the substances used and those dropped.',

    # clustering
    'clustering': 'Groups substances by sorption and ε (as in the notebook), both standardised.',
    'kmeans': 'Assigns each substance to the nearest of k centres (k-means++, 50 restarts). Best for compact, '
              'similar-sized groups.',
    'hierarchical': 'Merges the most similar substances step by step into a tree; cutting it gives k clusters. '
                    'A long vertical branch marks a natural split.',
    'linkage': 'Distance between groups. ward — minimal within-cluster variance; complete — farthest points; '
               'average — mean pairwise distance.',
    'spectral': 'Partitions a similarity graph. Handles irregular shapes but is sensitive to γ; unsatisfactory '
                'in the notebook.',
    'gamma': 'Similarity = exp(−γ·distance²). Large γ: only close points are similar.',
    'eigen': 'A large gap between consecutive Laplacian eigenvalues suggests k. Shown only; not used to pick k.',
    'adjacency': 'Distribution of pairwise similarities (σ = median distance). Values piled at 0 or 1 mean the '
                 'kernel width is poorly chosen.',
    'dbscan': 'Clusters dense regions and labels isolated points as noise. k is not set directly.',
    'eps': 'Neighbourhood radius (standardised units). Larger eps → fewer noise points, bigger clusters.',
    'min_samples': 'Neighbours within eps needed for a core point. Larger → stricter density, more noise.',
    'auto_k': 'Notebook rule: try k = 2…8, take the highest silhouette (ties → lowest Davies–Bouldin). '
              'Switch off to set parameters manually.',
    'elbow': 'Within-cluster sum of squares vs k. The bend ("elbow") is a reasonable k.',
    'silhouette': 'How much closer a point is to its own cluster than to the next one. −1…1, higher is better.',
    'ch': 'Between- to within-cluster dispersion. Higher is better; compare on the same data only.',
    'db': 'Average similarity of each cluster to its closest neighbour. Lower is better; 0 is ideal.',
    'ari': 'Agreement between two partitions: 1 — identical, ≈0 — chance level.',
    'kruskal': 'Rank test for sorption differences between clusters. Caution: clusters are built on sorption, '
               'so a small p is expected and is not independent evidence.',
    'chi2': 'Cluster × IR class counts with χ² and Fisher tests. With small counts trust Fisher.',
    'cluster_ransac': 'Sequential RANSAC (ε → sorption, positive slope) inside each cluster (notebook cells 44–47). '
                      'DBSCAN noise is skipped.',

    # correlations
    'pearson': 'Pearson r (linear) and Spearman ρ (monotonic, rank-based) over all substances, without RANSAC.',
    'ransac': 'RANSAC fits a line through the largest consistent subset and ignores outliers; sequential RANSAC '
              'repeats on the rest to find several lines. Point numbers match between plots and tables '
              '(ordered by sorption).',
    'tau': 'τ — maximum vertical distance from the line for a point to count as an inlier. '
           'Auto: chosen from the grid by max Q²_LOO × inlier fraction (as in the notebook).',
    'tau_grid': 'Candidate τ values. Default: the notebook grid (×0.1 for g/g and q·Vm).',
    'min_inliers': 'Minimum points per line: this fraction of substances, but not fewer than the floor '
                   '(default 20 %, ≥4).',
    'min_r2': 'R²_in — fit quality on the line’s own points. Lines below the threshold are rejected.',
    'min_q2': 'Q²_LOO — leave-one-out predictive quality. 1 perfect, 0 no better than the mean, <0 worse.',
    'max_lines': 'Maximum number of sequential lines.',
    'max_trials': 'Random subsets tried per fit; stops earlier at 99.9 % confidence.',
    'ci': 'Confidence band of the fitted line, computed from its inliers (as seaborn regplot).',
    'planes': 'Sorption as S = a·X + b·Y + c. Blue points lie within τ of the plane. By default only substances '
              'with all base-set descriptors are used (as in the notebook).',
    'fisher_ir': 'Fisher exact test: is the IR class related to being on a RANSAC line?',
    'q_corr': 'Q_corr = R²_in × inlier fraction (notebook score); C_corr = 1 − Q_corr.',

    # validation
    'permutation': 'Sorption is shuffled and the analysis repeated. p = share of shuffles that score at least '
                   'as well as the real data.',
    's_corr': 'S_corr = Σ Q²_LOO,i × N_in,i / N over all lines — the test statistic.',
    'perm_conditional': 'Conditional (notebook cell 82): τ and min_samples fixed to the fitted values. '
                        'Fast, slightly optimistic.',
    'perm_full': 'Full (new): the whole fit, including τ and min_samples selection, is repeated for each '
                 'shuffle. Slower, stricter.',
    'bh': 'Benjamini–Hochberg adjustment for testing many descriptors; controls the false discovery rate at 5 %.',
    'collinearity': 'Strongly correlated descriptors carry overlapping information, so their individual '
                    'contributions cannot be separated.',
    'vif': 'Variance inflation factor: ≈1 independent, >5 notable, >10 strong collinearity.',
    'pca': 'How many independent directions the descriptor set really contains.',
    'pls': 'PLS regression with nested leave-one-out validation (components chosen inside each training fold). '
           'Q² < 0 means worse than predicting the mean.',
    'y_random': 'y-randomisation (new): shuffle sorption and repeat the nested PLS validation.',
    'base_set': 'Descriptor set for collinearity, PLS and plane case selection. Notebook: ε, α, β, π*, DN.',
}
