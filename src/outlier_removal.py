
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore

def remove_residents_and_initial_cleaning(df):
    """
    Removes residents based on postcode and basic frequency threshold (> 11).
    This is the BASE cleaning applied before any specific outlier detection method.
    """
    postcodes_to_drop = ['18411', '18412', '18413']
    df['postcode'] = df['postcode'].astype(str)
    df = df[~df['postcode'].isin(postcodes_to_drop)].reset_index(drop=True)
    return df

def filter_by_iqr_and_thresholds(cluster_data):
    """
    Applies Standard IQR filtering and specific frequency threshold (<= 11).
    Used as the 'IQR' method in the paper.
    """
    Q1 = cluster_data[['monetary', 'frequency', 'recency']].quantile(0.25)
    Q3 = cluster_data[['monetary', 'frequency', 'recency']].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = cluster_data[
        (cluster_data['frequency'] >= lower_bound['frequency']) &
        (cluster_data['frequency'] <= 11) & 
        (cluster_data['monetary'] >= lower_bound['monetary']) &
        (cluster_data['monetary'] <= upper_bound['monetary']) &
        (cluster_data['recency'] >= lower_bound['recency']) &
        (cluster_data['recency'] <= upper_bound['recency'])
    ].reset_index(drop=True)
    
    return filtered_data

# --- Advanced Outlier Removal Methods ---

def remove_outliers_if(df, contamination=0.05):
    """Isolation Forest"""
    model = IsolationForest(contamination=contamination, random_state=42)
    cols = ['frequency', 'recency', 'monetary']
    # Handle NaN if any
    data = df[cols].dropna()
    preds = model.fit_predict(data)
    # preds: 1 for inliers, -1 for outliers
    return df.loc[data.index][preds == 1]

def remove_outliers_zscore(df, threshold=3):
    """Z-Score"""
    cols = ['frequency', 'recency', 'monetary']
    data = df[cols].dropna()
    z_scores = np.abs(zscore(data))
    # Keep rows where all z-scores are < threshold
    mask = (z_scores < threshold).all(axis=1)
    return df.loc[data.index][mask]

def remove_outliers_ldis_proxy(df, contamination=0.05):
    """LOF as proxy for LDIS (Local Density)"""
    cols = ['frequency', 'recency', 'monetary']
    data = df[cols].dropna()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    preds = lof.fit_predict(data)
    return df.loc[data.index][preds == 1]

def remove_outliers_kmedoids_proxy(df, percent_remove=0.05):
    """
    Proxy for k-medoids based removal. 
    Removes points furthest from the center of their group.
    """
    cols = ['frequency', 'recency', 'monetary']
    
    cleaned_dfs = []
    
    # Group by label if exists, else treat as global
    if 'rfm_label' in df.columns:
        groups = df.groupby('rfm_label')
    else:
        # Just use global mean if no labels yet (unlikely in this pipeline but safe)
        groups = [("all", df)]
        
    for name, group in groups:
        if len(group) < 2:
            cleaned_dfs.append(group)
            continue
            
        # Check for NaNs
        g_data = group[cols].dropna()
        if g_data.empty:
            continue
            
        centroid = g_data.mean().values
        distances = np.linalg.norm(g_data.values - centroid, axis=1)
        
        # Determine threshold distance
        threshold = np.percentile(distances, 100 * (1 - percent_remove))
        mask = distances <= threshold
        
        cleaned_dfs.append(group.loc[g_data.index][mask])
        
    return pd.concat(cleaned_dfs)

def remove_outliers_zscore_percent(df, percent_remove=0.05):
    """
    Removes top `percent_remove` based on Z-score magnitude.
    Score = max(|z_score|) across columns.
    """
    cols = ['frequency', 'recency', 'monetary']
    data = df[cols].dropna()
    
    # Calculate Abs Z-scores
    z_scores = np.abs(zscore(data))
    # Take the max z-score across the 3 dimensions as the "anomaly score"
    max_z = z_scores.max(axis=1)
    
    # Threshold is the value at (1-percent) quantile
    threshold = np.percentile(max_z, 100 * (1 - percent_remove))
    
    # Keep rows where score <= threshold
    mask = max_z <= threshold
    return df.loc[data.index][mask]

def remove_outliers_iqr_percent(df, percent_remove=0.05):
    """
    Removes top `percent_remove` based on deviation from IQR bounds.
    Score = max deviation factor beyond Q1/Q3.
    """
    cols = ['frequency', 'recency', 'monetary']
    data = df[cols].dropna()
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Avoid division by zero
    IQR = IQR.replace(0, 1e-9)
    
    # Calculate score: for each value x, score = max((Q1-x)/IQR, (x-Q3)/IQR)
    # If x is within [Q1, Q3], score is negative. If outlier, score > 0 (measures units of IQR away)
    # Actually, standard outlier is > 1.5 * IQR.
    # We want a continuous score to rank.
    
    # Distance from nearest quarterly
    # If x < Q1: dist = Q1 - x
    # If x > Q3: dist = x - Q3
    # Else: dist = 0 (or negative) - let's treat inliers as low score.
    # More aggressively: simply use distance from Median normalized by IQR?
    # No, stick to IQR bound logic. 
    
    lower_dist = (Q1 - data) / IQR
    upper_dist = (data - Q3) / IQR
    
    # Element-wise max of these two gives the "number of IQRs beyond bounds"
    # values < 0 mean inside box. 
    deviation_scores = np.maximum(lower_dist, upper_dist).max(axis=1)
    
    # Threshold
    threshold = np.percentile(deviation_scores, 100 * (1 - percent_remove))
    
    mask = deviation_scores <= threshold
    return df.loc[data.index][mask]
