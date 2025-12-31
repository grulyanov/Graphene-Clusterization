# MACHINE LEARNING FOR STUDYING GRAPHENE OXIDES CHEMISTRY

This repository contains a project that represents the processing of experimental data. The sorbiton-dielectric constant pair was measured in the experiment. This project is part of the work on the analysis of acid-base properties of graphene oxide surfaces by ATR IR spectroscopy. One of the main goals is to confirm the hypothesis that solvent sorption does not depend on dielectric constant and is determined by the chemistry of the process rather than physics. In the future, the data of physico-chemical parameters and multidimensional clustering will be analyzed.

## Structure

- `K-Means & Hierarchical clustering` &mdash; a baseline models which represents the best metrics
- `Spectral Clustering`  &mdash; used for more detailed analysis of data with a complex structure and non-standard distribution
- `DBSCAN`  &mdash; to search for clusters of any shape by selecting hyperparameters
- `RANSAC Regression` &mdash; robust regression method for determining strict linear correlations within the constructed clusters
- `Sequential RANSAC` &mdash; a method for searching for multiple linear correlations within the clusters
- `Spectral clustering tuning` &mdash; an attempt to optimize the method to improve the clustering quality
- `Plotly dashboard` &mdash; additional script for visualisation by `plotly`

## Conclusions 

- Four methods were used to solve the clustering problem, of which three showed satisfactory results. Spectral clustering proved unsatisfactory for our data. 
- Based on the table:


- | Method        | Silhouette | Calinski–Harabasz | Davies–Bouldin |
  |---------------|------------|--------------------|----------------|
  | K-Means       | 0.61       | 328.85             | 0.50           |
  | Hierarchical  | **0.70**   | 285.95             | **0.43**       |
  | Spectral      | 0.06       | 7.30               | 2.70           |
  | DBSCAN        | 0.60       | 185.67             | 0.76           |




- **Hierarchical clustering** performs the best overall.  
It has the highest Silhouette score and the lowest Davies–Bouldin index, which means its clusters are compact and well separated.K-Means is also strong but slightly worse.

- A pipeline based on RANSAC and Sequential RANSAC has been built to search for linear correlations within each cluster, which will allow us to make a scientific interpretation of the interactions of solvents with graphene oxide.

- As an additional search for linear correlations, plotly dashboard was used, which returned regression equations with a minimum amount of code.
