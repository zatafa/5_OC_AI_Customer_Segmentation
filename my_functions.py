#!/usr/bin/env python
# coding: utf-8

"""
@author: joanaleonard
Last update:
"""

import numpy as np
import pandas as pd
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle
from pandas.plotting import parallel_coordinates
import seaborn as sns
from math import radians
import scipy.stats as st
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score

# ----------------------------------------------------------------------------------------------------------

def merge_check(left_df, left_key, right_df, right_key):
    """Check the result of merging 2 datasets: 
    inner, left, right, outer values"""
    
    # Create a set of unique elements
    left_set = set(left_df[left_key].unique())
    right_set = set(right_df[right_key].unique())

    # Create indicators
    common_values = (left_set & right_set)
    not_in_right = (left_set - right_set)
    not_in_left = (right_set - left_set)
    left_rows_missing = left_df[left_key].isin(not_in_right)

    # Display joining elements
    print("Number of common values: '{}'".format(len(common_values)))
    print("Unique values '{}' not in RIGHT : {}".format(left_key,
                                                        len(not_in_right)))
    print("Unique values '{}' not in LEFT : {}".format(right_key,
                                                       len(not_in_left)))
    print("Exclusive rows of LEFT on {} : {}".format(left_key, len(left_df[left_rows_missing])))
    
# ----------------------------------------------------------------------------------------------------------

def bar_and_pie_plot(df, feature, figsize=(18, 4), categorical=False):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot histogram
    var = df[feature].value_counts()
    labels = var.index

    sns.barplot(x=labels, y=var.values, ax=ax1)
    ax1.set_xlabel('Values', fontsize=14)
    ax1.set_ylabel('Frequencies', fontsize=14)
    if categorical:
        ax1.set_xticklabels(list(labels), rotation=45)
    else:
        ax1.set_xticklabels(list(labels))

    # Plot pie
    ax2.pie(var,
        autopct='%.1f%%')
    ax2.set_xlabel('Relative frequencies',
               fontsize=14)
    ax2.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')

    plt.suptitle('{' + feature + '} distribution',
             fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------------------------------

def compute_haversine(row, latA, lngA, latB, lngB):
    """Find kilometer(km) distance between 2 pairs of
    latitude and longitude"""
    
    lat1 = row[latA]
    lng1 = row[lngA]
    lat2 = row[latB]
    lng2 = row[lngB]
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlng= lng2 - lng1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    km = round(6371 * c, 0)
    return km

# ----------------------------------------------------------------------------------------------------------

def pearson_corr_heatmap(df, mask_upper=True,
                         figsize=(10,10), font=13,
                         cmap='coolwarm'):
    
    """Plot Pearson correlation with or without upper triangle
    Set triange value to True to mask the upper triangle,
    to False to keep the upper triangle"""
    
    # Compute the correlation matrix
    corr_matrix = df.corr()
    
    # Generate a mask for the upper triangle
    # for lower triangle, transpose the mask with .T
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    
    # Delete empty cell of top and bottom
    mask = mask[1:, :-1]
    corr = corr_matrix.iloc[1:, :-1]
    
    plt.figure(figsize=figsize)
    
    if mask_upper:
        with sns.axes_style('white'):
            sns.heatmap(corr, mask=mask,
                        annot=True, annot_kws={'size': font},
                        fmt='.2f', cmap=cmap)
    else:
        sns.heatmap(corr_matrix, annot=True, fmt='.2f',
                    cmap=cmap)
    
    plt.tick_params(labelsize=font)
    plt.title('Pearson correlation Matrix',
              fontsize=15, fontweight='bold',
              x=0.5, y=1.05)
    plt.show()
    
# ----------------------------------------------------------------------------------------------------------

def encode_label(df, categorical_features):
    
    # Instantiate the transformer
    lbe = LabelEncoder()
    
    # Initiate the count of encoded variables
    lbe_count = 0

    for col in categorical_features.columns:
        # Train and transform
        df[col] = lbe.fit_transform(df[col])
        lbe_count += 1

    print("%d columns(s) are encoded with LabelEncoder." % lbe_count)
    
# ----------------------------------------------------------------------------------------------------------

def plot_histograms(df, cols, bins=30, figsize=(9, 4), color = 'lightgrey',
                    skip_outliers=False, z_thresh=3, layout=(3,3)):
    
    """Plot histogram subplots with multi-parameters:
    ------------------------------------------------------------
    ARGUMENTS :
    df = Dataframe
    cols = only numeric columns as list
    color = lightgrey, lightblue, etc.
    skip_outliers = True or False (plot with/without outliers)
    # outliers are computed with scipy.stats z-score
    z_thresh = z-score threshold, default=3
    layout : nb_rows, nb_columns
    ------------------------------------------------------------
    """

    fig = plt.figure(figsize=figsize)

    for i, c in enumerate(cols,1):
        ax = fig.add_subplot(*layout,i)

        # Choice to skip outliers or not
        if skip_outliers:
            features = df[c][np.abs(st.zscore(df[c]))< z_thresh]
        else:
            features = df[c]
        ax.hist(features,  bins=bins, color=color)
        ax.set_title(c)
        ax.vlines(df[c].mean(), *ax.get_ylim(),
                  color='red', ls='-', lw=1.5)
        ax.vlines(df[c].median(), *ax.get_ylim(),
                  color='green', ls='-.', lw=1.5)
        ax.vlines(df[c].mode()[0], *ax.get_ylim(),
                  color='goldenrod', ls='--', lw=1.5)
        ax.legend(['mean', 'median', 'mode'])
        ax.title.set_fontweight('bold')
        # xmin, xmax = ax.get_xlim()
        # ax.set(xlim=(0, xmax/5))

    # Title linked to skip_outliers
    if skip_outliers:
        plt.suptitle('Distribution excluding outliers',
                     fontsize=18, fontweight='bold',
                     y=1.05)
    else:
        plt.suptitle('Distribution including outliers',
                     fontsize=18, fontweight='bold',
                     y=1.05) 
    
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------------------------------

def compared_boxplot(df, xfeat, yfeat,
                     figsize=(18, 4), zscore_thresh=3):

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)

    labels = df[xfeat].value_counts().index

    # Plot WITHOUT outliers
    y1 = df[np.abs(st.zscore(df[yfeat]))< zscore_thresh]
    sns.boxplot(x=y1[xfeat], y=y1[yfeat], ax=ax1)
    ax1.set_xticklabels(list(labels), rotation=45, ha='right')
    ax1.set_title(
        '{} WITHOUT OUTLIERS | average {:.2f}'.format(str(yfeat),
                                                      y1[yfeat].mean()),
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    # Plot OUTLIERS
    y2 = df[np.abs(st.zscore(df[yfeat]))> zscore_thresh]
    sns.boxplot(x=y2[xfeat], y=y2[yfeat], ax=ax2)
    ax2.set_xticklabels(list(labels), rotation=45, ha='right')
    ax2.set_title(
        '{} OUTLIERS | average {:.2f}'.format(str(yfeat),
                                              y2[yfeat].mean()),
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    no_out_pct = (len(y1)/ len(df)) * 100
    out_pct = (len(y2)/ len(df)) * 100
    plt.suptitle(
        '{}: {:.1f}% not outliers, {:.1f}% outliers'.format(
            str(yfeat), no_out_pct, out_pct),
            fontsize=14, fontweight='bold', y=1.02)
    plt.show()
    
    
# ----------------------------------------------------------------------------------------------------------

def log_standardization(df):
    
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.preprocessing import StandardScaler
    
    # Filter on continuous variables
    temp = df[['recency', 'ttl_revenue']]

    # Log continuous variables
    # avoid '-inf' value with log1p instead of log
    transformer = FunctionTransformer(np.log1p)
    temp_log = transformer.transform(temp)

    # Update df with new values
    X_log = df.copy()
    X_log.update(temp_log, join='left')

    # Apply StandardScaler
    # mean=0, std=1
    std = StandardScaler()
    X_std = std.fit_transform(X_log)

    # Convert to dataframe and save to csv file
    scaled_data = pd.DataFrame(X_std,
                               index=df.index,
                               columns=df.columns)
    
    return scaled_data


# ----------------------------------------------------------------------------------------------------------

def pca_scree_plot(X, random_state=42,
                   n_comp=None, threshold=None,
                   text_pos=3):
    
    from sklearn.decomposition import PCA
    
    if n_comp:
        pca = PCA(n_components=n_comp,
                  random_state=random_state)
    else:
        pca = PCA(random_state=random_state)
    
    # Compute PCA
    pca.fit(X)

    # Get explained variance ratio (EVR) for each component
    evr_pct = pca.explained_variance_ratio_ * 100 # percentages
    cum_evr = np.cumsum(evr_pct) # cumulative percentages
    
    # Plot explained variance
    plt.figure(figsize=(9, 4))
    plt.bar(np.arange(len(evr_pct))+1, evr_pct,
            align="center", color='g')
    plt.plot(np.arange(len(evr_pct))+1, cum_evr,
             c='r', marker='o')
    
    # Set a cumulative ratio
    cum_evr_threshold = 90
    # Filter with the expected variance ratio
    temp = cum_evr <= cum_evr_threshold
    selected_dimensions = len(evr_pct[temp])
    plt.text(text_pos, 40,
             str('{}% variance with {} dimensions'.format(cum_evr_threshold,
                                                              selected_dimensions)),
             bbox = dict(boxstyle='square, pad=0.5', fc = 'lightblue'))
    
    plt.ylabel('Variance percentage')
    plt.xticks(range(1,len(evr_pct)+1))
    plt.title('Cumulative explained variance ratio')

    print(cum_evr)

    plt.show()

# ----------------------------------------------------------------------------------------------------------

def pca_correlation_circle(X, dim_a=0, dim_b=1):
    """Plot the PCA correlation circle related to provided dimension,
    a=0, b=1 for 1st two dimensions"""
    
    from sklearn.decomposition import PCA
    
    # Instantiate and compute PCA
    pca = PCA(random_state=42)
    pca.fit(X)

    # Get explained variance ratio (EVR) for each component
    evr_pct = pca.explained_variance_ratio_ * 100 # percentages
    cum_evr = np.cumsum(evr_pct) # cumulative percentages

    
    # Get original features names
    features_names = X.columns
    
    # Format figsize, xy labels, title
    plt.figure(figsize=(6, 6))
    plt.xlabel('Dim{} ({:.2f}%],'.format(dim_a+1, evr_pct[dim_a]))
    plt.ylabel('Dim{} ({:.2f}%],'.format(dim_b+1, evr_pct[dim_b]))
    plt.title('Correlation Circle - PC%d & ' % (dim_a+1) + 'PC%d' % (dim_b+1))
    
    # Plot the circle
    ax = plt.gca()
    ax.add_patch(Circle([0,0], radius=1,
                        color='k', linestyle='-',
                        fill=False, clip_on=False))
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    
    # Plot radius line for each axis
    plt.plot([-1,1], [0,0], color='grey',
             linestyle='dotted', alpha=0.5)
    plt.plot([0,0], [-1,1], color='grey',
             linestyle='dotted', alpha=0.5)
    
    # Plot explained variance in feature space
    x_pca = pca.components_[dim_a]
    y_pca = pca.components_[dim_b]

    sns.scatterplot(x=x_pca, y=y_pca,
                    color='blue', alpha=0.1)

    for x, y, col in zip(x_pca, y_pca, features_names):
        plt.annotate(col, (x,y),
                     textcoords='offset points',
                     xytext=(0, 5+np.random.randint(-20,20)),
                     ha='center')
        ax.arrow(0, 0, x, y,
                 head_width=0.05, head_length=0.02,
                 fc='grey', ec='blue', alpha=0.5)
    plt.grid(False)
    plt.show()

# ----------------------------------------------------------------------------------------------------------

def pca_correlation_matrix(X, n_comp=None, random_state=42,
                           figsize=(15,9)):
    
    from sklearn.decomposition import PCA
    
    if n_comp:
        pca = PCA(n_components=n_comp,
                  random_state=random_state)
    else:
        pca = PCA(random_state=random_state)
    
    # Compute PCA
    pca.fit(X)

    # Get explained variance ratio (EVR) pourcentage
    evr_pct = pca.explained_variance_ratio_ * 100 # percentages
    
    # Plot correlation matrix (heatmap)
    # display all components of selected dimensions
    all_components = [np.abs(pca.components_)[i] for i in range(len(evr_pct))]
    
    # Get original features names
    features_names = X.columns
    
    # Set dimension names
    dimension_names = {'DIM{}'.format(i+1): all_components[i] for i in range(len(evr_pct))}
    
    # Create the correlation matrix dataframe
    correlation_matrix = pd.DataFrame(all_components,
                                      columns=features_names,
                                      index=dimension_names)

    # Plot correlation matrix heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(correlation_matrix.T, annot=True, fmt='.2f', cmap='YlGnBu')
    ax.xaxis.set_ticks_position('top')
    plt.show()
    

# ----------------------------------------------------------------------------------------------------------

def pca_dim_reduction(X, n_comp=10,
                      random_state=42):
    
    from sklearn.decomposition import PCA
    
    # Display initial number of features
    print('Initial number of features :', X.shape[1])
    
    # Compute PCA
    pca = PCA(n_components=n_comp,
          random_state=random_state)
    pca.fit(X)

    # Get explained variance ratio (EVR) for each component
    evr_pct = pca.explained_variance_ratio_ * 100 # percentages
    cum_evr = np.cumsum(evr_pct) # cumulative percentages

    # Print number of dimensions and cumulative evr
    print('Selected number of features :', len(cum_evr))
    print('Cumulative explained variance: {}%'.format(round(cum_evr[-1],1)))

    # Get the projection of the data
    X_proj = pca.transform(X)

    # Convert as DataFrame
    columns = ['DIM{}'.format(i) for i in range(1, X_proj.shape[1]+1)]
    pca_data = pd.DataFrame(X_proj, columns=columns)
    return pca_data

# ----------------------------------------------------------------------------------------------------------

def cluster_scores_plot(df, figsize=(20, 5),
                        n_labels = range(2, 13)):
    """Plot clustering model common scores /
    Evaluate performance of the clustering model
    - Silhouette score, Inertia,
    Davies-Bouldin, Calinski-Harabasz"""
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import davies_bouldin_score
    from sklearn.metrics import calinski_harabasz_score
    
    # Create list for each score
    slh, sse, dbs, chs = {}, {}, {}, {}
    
    # Instantiate k-means score
    for k in n_labels:
        kmeans = KMeans(n_clusters=k,
                        random_state=42).fit(df)
        cluster_labels = kmeans.labels_
        
        # Silhouette score -> maximize
        slh[k] = silhouette_score(df, cluster_labels)
        
        # Sum squared error (sse) / distorsion -> Elbow
        sse[k] = kmeans.inertia_
        
        # Davies-Bouldin score -> minimize
        dbs[k] = davies_bouldin_score(df, cluster_labels)
        
        # Calinski-Harabasz score -> maximize
        chs[k] = calinski_harabasz_score(df, cluster_labels)

    # Create dataframe to save the scores
    cluster_k = {'Clust-{}'.format(i): cluster_labels[i] for i in n_labels}
    
    kClusters_score = pd.DataFrame({'silhouette':slh.values(),
                                    'inertia':sse.values(),
                                    'davis_bouldin':dbs.values(),
                                    'calinski_harabasz':chs.values()},
                                   index=cluster_k)

    # Plot figures
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, 
                                          figsize=figsize,
                                          sharex=True)
    
    # plt.style.use("fivethirtyeight")

    # Silhouette score -> maximize
    ax1.plot(n_labels, list(kClusters_score.silhouette.unique()),
             marker='o', c='r')
    ax1.set_xlabel('Number of cluster')
    ax1.set_xticks(n_labels)
    ax1.set_ylabel('')
    ax1.set_title('Silhouette score')

    # Sum squared error (sse) -> Elbow
    ax2.plot(n_labels, list(kClusters_score.inertia.unique()),
             marker='o', c='b')
    ax2.set_xlabel('Number of cluster')
    ax2.set_xticks(n_labels)
    ax2.set_ylabel('')
    ax2.set_title('Sum of squared error (SSE)')

    # Davies-Bouldin score -> minimize
    ax3.plot(n_labels, list(kClusters_score.davis_bouldin.unique()),
             marker='o', c='g')
    ax3.set_xlabel('Number of cluster')
    ax3.set_xticks(n_labels)
    ax3.set_ylabel('')
    ax3.set_title('Davies-Bouldin score')

    # Calinski-Harabasz score -> maximize
    ax4.plot(n_labels, list(kClusters_score.calinski_harabasz.unique()),
             marker='o', c='purple')
    ax4.set_xlabel('Number of cluster')
    ax4.set_xticks(n_labels)
    ax4.set_ylabel('')
    ax4.set_title('Calinski-Harabasz score')

    plt.suptitle('Clustering indicators for optimal number of clusters',
             fontsize=15, fontweight='bold')
    plt.show()


# ----------------------------------------------------------------------------------------------------------

def optimalK_elbow(X, cluster_range=range(2,13)):
    '''Compute and plot KMeans inertia to find the elbow point
    for optimal k number of cluster'''
    
    sse = {}
    for k in cluster_range:
        kmeans=KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse[k] = kmeans.inertia_
    #Plot
    plt.figure(figsize=(8, 4))
        
    sns.lineplot(data=sse, color='darkslateblue',
                 marker='o', linewidth=2)
    plt.xticks(cluster_range)
    plt.title('Elbow criterion for optimal k clusters',
              fontsize=14, fontweight='bold')
    
    plt.show()

# ----------------------------------------------------------------------------------------------------------

def stability_init(X, model, nb_iter=10):
    """Compute the ARI scores per model iterations"""
    
    # Create randomnly initialized partitions for comparison
    partitions = []

    # Iteration
    for i in range(nb_iter):

        # Fit the model
        model.fit(X)

        # Get the labels
        partitions.append(model.labels_)
    
    # Compute ARI scores for each partition pair
    # Create ARI scores list
    ARI_scores = []

    # For each partition (except last one)
    for i in range(nb_iter-1):

        # Compute the ARI score against other partitions
        for j in range(i+1, nb_iter):
            pairs_list = adjusted_rand_score(partitions[i],
                                           partitions[j])
            ARI_scores.append(pairs_list)
    
    # Compute the mean and standard deviation of ARI scores
    ARI_mean_pct = np.mean(ARI_scores) * 100
    ARI_std_pct = np.std(ARI_scores) * 100

    # Print results
    print('ARI: mean {:.1f}%, std {:.1f}%'.format(ARI_mean_pct,
                                                  ARI_std_pct))
    
    return ARI_scores

# ----------------------------------------------------------------------------------------------------------

def plot_stability_init(X, cluster_list=[3,4,5,6], nb_iter=10):
    
    # Create a dataframe to store the results
    stab_init_ARI = pd.DataFrame()
    
    for k in cluster_list:
        model_name = (f'km_{str(k)}k')
        init_name = (f'stab_init_{str(k)}k')
        stab_init = stability_init(X,
                                   KMeans(n_clusters=k),
                                   nb_iter=nb_iter)
        stab_init_df = pd.DataFrame(stab_init,
                                    columns=[model_name])
        stab_init_ARI = pd.concat([stab_init_ARI,
                                   stab_init_df],
                                  axis=1)
    
    # Plot the result
    plt.figure(figsize=(8,4))
    
    stab_init_ARI.boxplot(color='red', vert=False)
    plt.title(
        'KMeans: Stability at initialization for {} iterations'.format(nb_iter),
        fontweight='bold')
    plt.ylabel('Number of clusters')
    
    plt.show()

# ----------------------------------------------------------------------------------------------------------

def kmeans_segment_analysis(X, initial, k=3, random_state=42):
    '''Fit KMeans with k cluster, assign segment label to initial dataset,
    display relative importance of each feature for each segment,
    display proportion of population per segment,
    display contribution to revenue per segment'''
    
    # Compute Kmeans with k clusters
    kmeans=KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(X)
    
    # Assign segment to each customer in initial dataset
    kmeans_data = initial.assign(segment=kmeans.labels_)
    
    # Compute average for each feature by segment
    kmeans_averages = kmeans_data.groupby(['segment']).mean().round(2)
    
    # Compute the mean and standard deviation of each feature
    kmeans_averages.loc['mean_pop'] = initial.mean().round(2)
    # kmeans_averages.loc['std_pop'] = initial.std().round(2)
    
    # Display kmeans table
    display(kmeans_averages)
    print('\n')
    
       
    # Compute relative importance for each feature by segment
    temp = kmeans_averages[kmeans_averages.index.isin(range(0,k))]
    relative_imp = (temp / initial.mean() - 1)
    relative_imp.round(2)
    
    # Prepare figures configuration
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,4))
    
    # Plot features importance for each segment  
    sns.heatmap(data=relative_imp, annot=True, fmt='.2f',
                cmap='RdYlBu', ax=ax1)
    ax1.set_title(f'Relative importance of features for {k} clusters',
                  fontweight='bold', pad=30)
    ax1.set_ylabel(ylabel='Segment')
    
    # Compute clusters proportion
    pop_perc = (
        kmeans_data['segment'].value_counts() / len(kmeans_data) * 100)
    pop_perc.sort_index(inplace=True)

    # Plot clusters proportion
    _,_, autotexts = ax2.pie(pop_perc, autopct='%1.0f%%',
                             textprops=dict(color="w"))
    plt.setp(autotexts, size=12, weight='bold')
    c_circle=plt.Circle((0,0), 0.40, color='white')
    ax2.add_patch(c_circle)
    ax2.set_title('% of population per segment',
                  fontweight='bold', pad=30)


    # Compute revenue for each segment
    segment_value = kmeans_data.groupby('segment')['ttl_revenue'].sum()

    # Plot contribution to revenue
    _,_, autotexts = ax3.pie(segment_value, autopct='%1.0f%%',
                             textprops=dict(color="w"))
    plt.setp(autotexts, size=12, weight='bold')
    c_circle=plt.Circle((0,0), 0.40, color='white')
    ax3.add_patch(c_circle)
    ax3.legend(labels=segment_value.index, loc='center left',
               bbox_to_anchor=(-0.4, 0.5))
    ax3.set_title('Revenue contribution per segment',
                  fontweight='bold', pad=30)
    
    plt.show()
    
    
# ----------------------------------------------------------------------------------------------------------

def groupby_unique_customer(df):
    '''Iterate groupby per customer_unique_id'''
    
    # Round all floats to 2 decimal places
    pd.options.display.float_format = '{:.2f}'.format
    
    # Initialize customers grouping
    gp_df = df.groupby('customer_unique_id')
    
    # Create the final df
    main_df = gp_df.purchase_date.max().rename('max_purchase_date')
    
    # Create specific features to be aggregated to final dataframe
    specific_features = []
    specific_features.append(gp_df.order_id.nunique().rename('nb_orders'))
    specific_features.append(gp_df.payment_value.sum().rename('ttl_revenue'))
    specific_features.append(gp_df.review_score.mean().rename('avg_review'))
    specific_features.append(gp_df.payment_installments.mean().rename('avg_installments'))
    
    # Merge
    for feature_series in specific_features:
        main_df = pd.merge(main_df, feature_series,
                           on='customer_unique_id')
        
    # df max date against customers' max date of purchase
    main_df['recency'] = ((
        (main_df['max_purchase_date'].max() -
         main_df['max_purchase_date']).dt.days)/30).apply(np.floor)
    
    
    return main_df

# ----------------------------------------------------------------------------------------------------------


def silhouette_pca_umap_tsne(X, X_proj, U, T, k_range=[4,5,6]):
    
    """Plot silhouette per k cluster and visualize with pca, umap and tsne"""
    
    import matplotlib.cm as cm
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import silhouette_samples
    
    for k in k_range:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        fig.set_size_inches(20, 5)
        
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        
        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            cluster_silhouette_values.sort()

            # Compute the (new) y_upper
            size_cluster = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster

            # Assign a color per cluster i in range k
            color = cm.nipy_spectral(float(i) / k)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, cluster_silhouette_values,
                              facecolor=color, edgecolor=color,
                              alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        ax1.set_title('Avg silhouette: {:.3f}'.format(silhouette_avg),
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('Silhouette coefficient values')
        ax1.set_ylabel('Cluster label')

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1
        # For this one, we go from -0.2 to 0.6
        ax1.set_xlim([-0.2, 0.6])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (k + 1) * 10])

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color='red', linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])

        for data, axs, visu in zip([X_proj, U, T],
                                   [ax2, ax3, ax4],
                                   ['PCA', 'UMAP', 'T-SNE']):
            # Visualisation
            sns.scatterplot(x=data[:,0],
                            y=data[:,1],
                            hue=cluster_labels,
                            legend='full',
                            palette=sns.color_palette('hls', k),
                            alpha=0.8,
                            ax=axs)
            # Set layout
            axs.set_title('Projection: '+visu,
                          fontsize=12, fontweight='bold')
            axs.set_xlabel('DIM 1')
            axs.set_ylabel('DIM 2')

        # Labeling the clusters for PCA projection
        centers = kmeans.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
    
        # Add suptitle for each cluster plot
        plt.suptitle('Silhouette analysis for KMeans with {} clusters'.format(k),
                     fontsize=16, fontweight='bold')

    plt.show()

    
# ----------------------------------------------------------------------------------------------------------

# https://github.com/OpenClassrooms-Student-Center/Multivariate-Exploratory-Analysis/blob/master/functions.py
# https://openclassrooms.com/fr/courses/5869986-perform-an-exploratory-data-analysis/6177861-analyze-the-results-of-a-k-means-clustering

def display_parallel_coordinates_centroids(scaled_data,  filter_data,
                                           n_clusters=6):
    '''Display a parallel coordinates plot for the centroids in df'''
    
    from pandas.plotting import parallel_coordinates
    
    # Compute Kmeans with k clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    
    # Create a data frame containing our centroids
    centroids = pd.DataFrame(kmeans.cluster_centers_,
                             columns=filter_data.columns)
    centroids['segments'] = centroids.index
    
    # Prepare the color of the plot
    palette = sns.color_palette('bright', n_clusters)

    # Create the plot
    fig = plt.figure(figsize=(15, 6))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids",
                         fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(centroids, 'segments', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(30)
    
    plt.show()

    
# ----------------------------------------------------------------------------------------------------------

def kmeans_segment_polar(scaled_data, filter_data,
                         n_clusters=6, k_range=range(0, 6)):
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler
    
    # Compute Kmeans with k clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)

    # Assign segment to each customer in initial dataset
    kmeans_data = filter_data.assign(segment=kmeans.labels_)

    # Compute average for each feature by segment
    kmeans_averages = kmeans_data.groupby(['segment']).mean().round(2)
    
    # Instantiate MinMax Scaling on the data
    # Prepare the data for polar plot
    # default range (0,1)
    sc = MinMaxScaler()
    sc_kmeans = sc.fit_transform(kmeans_averages)

    # Convert to dataframe
    kmeans_df = pd.DataFrame(data=sc_kmeans,
                             columns=scaled_data.columns,
                             index=pd.Series(k_range,
                                             name='segment'))
    
    plotly_6means_scatterpolar(kmeans_data, kmeans_df, 
                               k_range=range(0,6), rows=2, cols=3) 
    
# ----------------------------------------------------------------------------------------------------------

def plotly_6means_scatterpolar(kmeans_data, kmeans_df, 
                               k_range=range(0,6), rows=2, cols=3):
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    pio.renderers.default='notebook'
    
    # Prepare the name and the weight of each cluster
    var = (kmeans_data['segment'].value_counts(normalize=True)*100).round(0)
    cls_weight = var.sort_index().values
    cls_name = ['clust{}: {}%'.format(i+1,cls_weight[i]) for i in k_range]
    
    # Set scatterpolar variables
    cls = {}
    cls_rlist = {}
    feat_to_plot = kmeans_df.columns
    df = kmeans_df.reset_index()

    # Plot
    fig = make_subplots(rows=rows, cols=cols,
                        specs=[[{'type': 'polar'}]*cols]*rows,
                        subplot_titles=cls_name)

    for i in k_range:
        # Create scatterpolar data
        cls[i] = df.loc[df['segment'] == i]
        cls_rlist[i]  = cls[i][feat_to_plot].to_numpy()
        # Add trace for each cluster
        if i < cols:
            row = 1
            col = i+1
        else:
            row = 2
            col = i-2
        fig.add_trace(go.Scatterpolar(name=cls_name[i],
                                      r=cls_rlist[i].tolist()[0],
                                      theta=feat_to_plot,
                                      fill='toself',
                                      hoverinfo='name+r'),
                      row=row, col=col)
        # Note : fixed polar size with radialaxis 
        # and min / max values of data
        # run fig.layout to check the values
        fig.update_layout(title={'text':'Clusters comparison',
                                 'y':0.95, 'x':0.5,
                                 'xanchor':'center',
                                 'yanchor':'top'},
                          height=600, width=1000,
                          polar=dict(radialaxis=dict(range=[0,1],
                                                     showticklabels=False,
                                                     ticks='')),
                          polar2=dict(radialaxis=dict(range=[0,1],
                                                      showticklabels=False,
                                                      ticks='')),
                          polar3=dict(radialaxis=dict(range=[0,1],
                                                      showticklabels=False,
                                                      ticks='')),
                          polar4=dict(radialaxis=dict(range=[0,1],
                                                      showticklabels=False,
                                                      ticks='')),
                          polar5=dict(radialaxis=dict(range=[0,1],
                                                      showticklabels=False,
                                                      ticks='')),
                          polar6=dict(radialaxis=dict(range=[0,1],
                                                      showticklabels=False,
                                                      ticks='')),
                          showlegend=False)
        
        if i < cols:
            fig.layout.annotations[i].update(y=1.03)
        else:
            fig.layout.annotations[i].update(y=0.4)
        
    fig.show('notebook')


# ----------------------------------------------------------------------------------------------------------

def dbscan_segment_analysis(filter_data, dbs_labels,
                            random_state=42):
    
    # Assign segment to each customer in initial dataset
    dbs_data = filter_data.assign(segment=dbs_labels)

    # Compute average for each feature by segment
    dbs_averages = dbs_data.groupby(['segment']).mean().round(2)
    dbs_averages 
    
    # Display dbscan table
    display(dbs_averages)
    print('\n')  
       
    # Compute relative importance for each feature by segment
    relative_imp = (dbs_averages / filter_data.mean() - 1)
    relative_imp.round(2)
    
    # Prepare figures configuration
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,4))
    
    # Plot features importance for each segment  
    sns.heatmap(data=relative_imp, annot=True, fmt='.2f',
                cmap='RdYlBu', ax=ax1)
    ax1.set_title(f'Relative importance of features: 4 clusters and noise',
                  fontweight='bold', pad=30)
    ax1.set_ylabel(ylabel='Segment')
    
    # Compute clusters proportion
    pop_perc = (
        dbs_data['segment'].value_counts() / len(dbs_data) * 100)
    pop_perc.sort_index(inplace=True)

    # Plot clusters proportion
    _,_, autotexts = ax2.pie(pop_perc, autopct='%1.0f%%',
                             textprops=dict(color="w"))
    plt.setp(autotexts, size=12, weight='bold')
    c_circle=plt.Circle((0,0), 0.40, color='white')
    ax2.add_patch(c_circle)
    ax2.set_title('% of population per segment',
                  fontweight='bold', pad=30)


    # Compute revenue for each segment
    segment_value = dbs_data.groupby('segment')['ttl_revenue'].sum()

    # Plot contribution to revenue
    _,_, autotexts = ax3.pie(segment_value, autopct='%1.0f%%',
                             textprops=dict(color="w"))
    plt.setp(autotexts, size=12, weight='bold')
    c_circle=plt.Circle((0,0), 0.40, color='white')
    ax3.add_patch(c_circle)
    ax3.legend(labels=segment_value.index, loc='center left',
               bbox_to_anchor=(-0.4, 0.5))
    ax3.set_title('Revenue contribution per segment',
                  fontweight='bold', pad=30)
    
    plt.show()
    