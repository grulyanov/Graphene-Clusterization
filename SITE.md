# Graphene oxide probe sorption explorer

Interactive Streamlit application for the sorption of molecular probes on graphene oxide.
It reproduces `Graphene_unified_analysis_filtered.ipynb` on an editable dataset: every plot,
table, cluster, RANSAC line and metric is recomputed after any change.

| | Russian | English |
|---|---|---|
| Entry point | `app.py` | `app_en.py` |
| Live site | https://graphene-clusterization.streamlit.app/ | deploy `app_en.py` (see below) |

## Features

- **Data** — edit any cell (comma or dot decimals), include/exclude substances individually,
  by chemical class or by preset, add substances (descriptors from SMILES via RDKit) and
  descriptors, paste columns from Excel, undo, change log, comparison with the original Excel,
  download/upload of the working dataset (`.xlsx`, lossless).
- **Clustering** — k-means, hierarchical (dendrogram), spectral, DBSCAN. Parameters are picked
  by the notebook rule or set manually; the site flags whether the chosen k is optimal and shows
  silhouette, Calinski–Harabasz and Davies–Bouldin for every k. Sequential RANSAC within each
  cluster, ARI between methods, Kruskal–Wallis, cluster × IR class tests.
- **Correlations** — Pearson/Spearman; sequential RANSAC per descriptor with confidence bands;
  line-coloured inlier tables; RANSAC planes for any descriptor pair.
- **Validation** — correlation matrix, VIF, PCA; nested leave-one-out PLS with y-randomisation;
  conditional and full permutation tests with Benjamini–Hochberg correction.
- Units: mmol/g, g/g (M from SMILES) or q·Vm (needs density). Every table copies and downloads
  (CSV/Excel), every figure downloads (PNG 600 dpi/SVG).

## Run locally

```bash
python3 -m venv .venv-site
.venv-site/bin/pip install -r requirements.txt
.venv-site/bin/streamlit run app_en.py      # English
.venv-site/bin/streamlit run app.py         # Russian
```

## Deploy (Streamlit Community Cloud)

share.streamlit.io → **Create app** → repository `grulyanov/Graphene-Clusterization`,
branch `streamlit-site`, main file `app_en.py` (or `app.py`), Python 3.12 → **Deploy**.
Both apps share the repository; the English app applies its own dark-blue theme at start-up.

## Reproducibility

- `analysis.py` keeps the notebook functions unchanged in logic; new methods are marked.
  The RANSAC engine (`ransac_fit`, `Ols`, `loo_q2`) is a vectorised exact copy of the
  scikit-learn loop (same random subsets): 60× faster, identical results.
- `python dev/verify.py 1000` — 222/222 values match the executed notebook
  (`dev/verify_table.md`).
- `python dev/fast_equivalence.py` — fast vs scikit-learn RANSAC on 1500 random datasets.
- `python dev/reactivity.py` — edit scenarios and the save → reset → load cycle.
- `python dev/style_check.py` — notebook vs site figure pairs in `check_style/`.

## English interface

`app_en.py` is generated from `app.py`:

```bash
python dev/make_en.py
```

Translations live in `dev/translations_en.py`, short explanations in `explain_en.py`, data
labels and messages in `data.py`/`analysis.py` via `i18n.t()`. The generator stops and lists
any new untranslated string.

## Data notes

- `Molar Masses, g/mol` in the Excel file is corrupted (102.089 for most rows); the site uses
  M from SMILES. For acetonitrile the Excel `Sorption, g/g` (0.540) disagrees with
  mmol/g × M / 1000 (≈ 0.160); the recomputed value is used.
- IR response classes and densities are not in the Excel file; the corresponding tables and
  the q·Vm unit appear once they are filled in.

## Licence and citation

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA.md`). Citation: `CITATION.cff`.
