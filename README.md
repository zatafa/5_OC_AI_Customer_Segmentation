# :dart: Customer Segmentation
Provide customer segmentation that can be used in a daily basis for marketing campaign.

**Expected output**: A proposition of segments with actionable description.

# :card_index_dividers: Dataset
[Brazilian e-commerce dataset by OLIST](https://www.kaggle.com/olistbr/brazilian-ecommerce)

## Data Schema
<img src=".\pictures\data_schema.png">

# :scroll: Tasks
- :heavy_check_mark: Preprocessing, incl. scaling, normalization
- :heavy_check_mark: Search of optimal "k"
- :heavy_check_mark: Assess stability at initialization
- :heavy_check_mark: Perform silhouette analysis with dimensionality reduction
- :heavy_check_mark: Perform segment analysis
- :heavy_check_mark: Assess segments' stability over time for maintenance

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
- [Clustering with scikit-learn](https://scikit-learn.org/stable/modules/clustering.html)
- [KMeans clustering algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- [Optimal "k"](https://gdcoder.com/silhouette-analysis-vs-elbow-method-vs-davies-bouldin-index-selecting-the-optimal-number-of-clusters-for-kmeans-clustering/)
- Dataviz : [Silhouette analysis with scikit-learn](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html), [UMAP](https://umap-learn.readthedocs.io/en/latest/basic_usage.html), [T-SNE](https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b)
- [Adjusted Rand Score with scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#adjusted-rand-score)
- Plotly [Polar Plot](https://plotly.com/python/polar-chart/), [Radar chart](https://plotly.com/python/radar-chart/), [Sankey Diagram](https://www.python-graph-gallery.com/sankey-diagram-with-python-and-plotly)