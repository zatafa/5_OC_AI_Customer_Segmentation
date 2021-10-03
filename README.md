# :dart: Customer Segmentation
Provide customer segmentation that can be used in a daily basis for marketing campaign.

**Expected output**: A proposition of segments with actionable description.

# :card_index_dividers: Dataset
[Brazilian e-commerce dataset by OLIST](https://www.kaggle.com/olistbr/brazilian-ecommerce)

## Data Schema
<img src=".\pictures\data_schema.png">

# :scroll: Tasks
- :heavy_check_mark: Preprocessing, incl. scaling, normalization
- :heavy_check_mark: Assessment of segments' stability over time

## Optimal "k" for KMeans
Silhouette score, Elbow Method, Davies-Bouldin indice, Calinski-Harabasz indice

<img src=".\pictures\optimal_k_search.png">

## Dataviz: silhouette analysis, PCA, UMAP, T-SNE
<img src=".\pictures\silhouette_score_pca_umap_tsne.png">

## Segment analysis: feature importance, % of population, revenue contribution
<img src=".\pictures\segment_analysis.png">

## Clusters' differentiation: polar plot
<img src=".\pictures\clusters_polar_plot.png">

## Segments' stability: Sankey diagram
<img src=".\pictures\segment_sankey_diagram.png">

# :computer: Dependencies
Pandas, Numpy, Scipy, Matplotlib, Plotly Graph Objects, scikit-learn

# :pushpin: References 
- xx