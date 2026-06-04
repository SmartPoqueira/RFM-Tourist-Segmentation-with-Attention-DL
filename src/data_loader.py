import pandas as pd
import os

def load_and_clean_data(base_path="../original"):
    """
    Loads visitor and monetary data, performs cleaning and feature engineering.
    """
    visits_path = os.path.join(base_path, "BD_visitas_DEFINITIVO_POQ.csv")
    monetary_path = os.path.join(base_path, "Monetary_INE.csv")

    if not os.path.exists(visits_path) or not os.path.exists(monetary_path):
        raise FileNotFoundError(f"Data files not found in {base_path}. Please check file paths.")

    print("Loading data...")
    df = pd.read_csv(visits_path, sep=',', low_memory=False)
    
    # Select relevant columns
    cols_to_keep = ['num_plate', 'num_plate_ID', 'nights', 'entry_time', 'entry_date', 
                    'exit_time', 'exit_date', 'entry_cam', 'exit_cam', 'visit_time', 
                    'postcode', 'autonomous_community', 'province', 'max_num_seats']
    df = df[cols_to_keep]

    # Date conversion
    df['entry_datetime'] = pd.to_datetime(df['entry_date'] + ' ' + df['entry_time'])
    df['exit_datetime'] = pd.to_datetime(df['exit_date'] + ' ' + df['exit_time'])
    df = df.drop(['entry_time', 'entry_date', 'exit_date', 'exit_time'], axis=1)

    # Initial Cleaning
    df = df.dropna(subset=['province']).reset_index(drop=True)
    df = df[df['max_num_seats'] > 0].reset_index(drop=True)

    # Postcode cleaning
    df['postcode'] = df['postcode'].astype(str).str.split('.', expand=True)[0].str.zfill(5)

    # Period creation
    df['Periodo'] = df['entry_datetime'].dt.year.astype(str) + 'T' + (df['entry_datetime'].dt.month // 4 + 1).astype(str)

    # Visit time processing
    df['visit_time'] = pd.to_timedelta(df['visit_time'])
    df['visit_time_timedelta'] = df['visit_time']
    df['visit_time'] = df['visit_time'].dt.total_seconds() / (24 * 60 * 60) # Convert to days

    # Monetary Data Merge
    df_ine = pd.read_csv(monetary_path, sep=',', low_memory=False)
    df_ine.rename(columns={'Total': 'monetary'}, inplace=True)
    df_ine['monetary'] = df_ine['monetary'].str.replace(',', '.').astype(float)

    dataset = pd.merge(df, df_ine[['Periodo', 'autonomous_community', 'monetary']], 
                       on=['Periodo', 'autonomous_community'], how='left')

    # Monetary total calculation (Occupancy factor 0.52 from surveys)
    dataset['monetary_total'] = dataset['monetary'] * dataset['visit_time']
    dataset['monetary_total'] = dataset['monetary_total'] * dataset['max_num_seats'] * 0.52

    # Final filter
    dataset = dataset[dataset['visit_time'] > 0].reset_index(drop=True)
    
    print(f"Data loaded successfully. Shape: {dataset.shape}")
    return dataset

def aggregate_rfm_data(dataset, reference_date='2023-08-01'):
    """
    Aggregates dataset to RFM level by number plate.
    """
    grouped_df = dataset.groupby('num_plate').agg({
        'visit_time': 'sum',
        'nights': 'sum',
        'monetary_total': 'sum',
        'num_plate_ID': 'first',
        'postcode': 'first',
        'province': 'first',
        'exit_datetime': 'max',
        'autonomous_community': 'first',
        'num_plate': 'count'
    })

    grouped_df.rename(columns={'num_plate': 'total_frequency'}, inplace=True)
    grouped_df.reset_index(inplace=True)

    reference_datetime = pd.to_datetime(reference_date)
    grouped_df['recency'] = (reference_datetime - grouped_df['exit_datetime']).dt.days
    
    # Rename for standard RFM
    cluster_data = grouped_df[['num_plate', 'monetary_total', 'total_frequency', 'recency', 'postcode']].copy()
    cluster_data.rename(columns={'monetary_total': 'monetary', 'total_frequency': 'frequency'}, inplace=True)
    
    # Filter monetary > 0
    cluster_data = cluster_data[cluster_data["monetary"] > 0].reset_index(drop=True)
    
    return cluster_data
