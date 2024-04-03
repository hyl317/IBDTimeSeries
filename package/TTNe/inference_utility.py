import pandas as pd
import numpy as np
import itertools

def get_ibdHistogram_by_sampleCluster(df_ibd, sampleCluster2id, bins, ch_ids):
    """
    Produce a dictionary of IBD histograms for each pair of sample clusters
    """
    histograms_by_sampleClusterPairs = {}
    dfs = []
    # so that IBD segments residing on chromosomes that appear more than once in ch_ids will also be copied more than once
    # this is desired for bootstraping purpose
    for ch_id in ch_ids:
        df = df_ibd[df_ibd['ch'] == ch_id] # subset to chromosomes of interest
        dfs.append(df)
    df_ibd_chromosome_resampled = pd.concat(dfs) 
    for (clusterID1, clusterID2) in itertools.combinations_with_replacement(sampleCluster2id.keys(), 2):
        if clusterID1 != clusterID2:
            df_ibd_subset1 = df_ibd_chromosome_resampled[(df_ibd_chromosome_resampled['iid1'].isin(sampleCluster2id[clusterID1])) \
                               & (df_ibd_chromosome_resampled['iid2'].isin(sampleCluster2id[clusterID2]))]
            df_ibd_subset2 = df_ibd_chromosome_resampled[(df_ibd_chromosome_resampled['iid1'].isin(sampleCluster2id[clusterID2])) \
                                 & (df_ibd_chromosome_resampled['iid2'].isin(sampleCluster2id[clusterID1]))]
            df_ibd_subset = pd.concat([df_ibd_subset1, df_ibd_subset2])
        else:
            df_ibd_subset = df_ibd_chromosome_resampled[(df_ibd_chromosome_resampled['iid1'].isin(sampleCluster2id[clusterID1])) \
                                 & (df_ibd_chromosome_resampled['iid2'].isin(sampleCluster2id[clusterID1]))]
        histogram, _ = np.histogram(100*df_ibd_subset['lengthM'], bins=bins)
        histograms_by_sampleClusterPairs[(clusterID1, clusterID2)] = histogram
    return histograms_by_sampleClusterPairs