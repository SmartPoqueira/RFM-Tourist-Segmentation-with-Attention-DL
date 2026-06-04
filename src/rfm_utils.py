import pandas as pd
import numpy as np

def calculate_rfm_scores(df_rfm):
    """
    Compute R, F, M scores and concatenate them into a 3-digit RFM_SCORE string.

    Scoring rules:
      R (Recency):   quintile cut, labels [5,4,3,2,1] — lower recency (more recent) → score 5.
      F (Frequency): fixed bins (−∞,1,2,3,4,∞], labels [1,2,3,4,5] — higher frequency → score 5.
      M (Monetary):  quintile cut, labels [1,2,3,4,5] — higher spend → score 5.
    """
    df_rfm["R"] = pd.qcut(df_rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

    df_rfm["F"] = pd.cut(df_rfm['frequency'],
                         bins=[-float('inf'), 1, 2, 3, 4, float('inf')],
                         labels=[1, 2, 3, 4, 5])

    df_rfm["M"] = pd.qcut(df_rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    df_rfm["RFM_SCORE"] = (df_rfm['R'].astype(str) +
                           df_rfm['F'].astype(str) +
                           df_rfm['M'].astype(str))

    return df_rfm

SEG_MAP = {
    'champions': ['555', '554', '544', '545', '454', '455', '445'],
    'loyal_customers': ['543', '444', '435', '355', '354', '345', '344', '335'],
    'potential_loyalists': ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451', '442', '441', '431', '453', '433', '432', '423', '353', '352', '351', '342', '341', '333', '323'],
    'new_customers': ['512', '511', '422', '421', '412', '411', '311'],
    'promising': ['525', '524', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313'],
    'need_attention': ['535', '534', '443', '434', '343', '334', '325', '324'],
    'about_to_sleep': ['331', '321', '312', '221', '213', '231', '241', '251'],
    'cant_lose_them': ['155', '154', '144', '214', '215', '115', '114', '113'],
    'at_risk': ['255', '254', '245', '244', '253', '252', '243', '242', '235', '234', '225', '224', '153', '152', '145', '143', '142', '135', '134', '133', '125', '124'],
    'hibernating': ['332', '322', '233', '232', '223', '222', '132', '123', '122', '212', '211'],
    'lost': ['111', '112', '121', '131', '141', '151'],
}

def map_segment(rfm_score):
    for segment, scores in SEG_MAP.items():
        if rfm_score in scores:
            return segment
    return 'unknown'

def assign_segments(df_rfm):
    df_rfm = calculate_rfm_scores(df_rfm)
    df_rfm['rfm_label'] = df_rfm['RFM_SCORE'].apply(map_segment)
    return df_rfm

LABEL_MAPPING = {
    'champions': 0,
    'loyal_customers': 1,
    'potential_loyalists': 2,
    'new_customers': 3,
    'promising': 4,
    'need_attention': 5,
    'about_to_sleep': 6,
    'cant_lose_them': 7,
    'at_risk': 8,
    'hibernating': 9,
    'lost': 10
}

IDX2CLASS = {v: k for k, v in LABEL_MAPPING.items()}
