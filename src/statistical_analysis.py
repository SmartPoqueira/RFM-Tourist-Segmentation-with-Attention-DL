
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy import stats
import time
import warnings

# Local imports
from data_loader import load_and_clean_data, aggregate_rfm_data
from outlier_removal import (
    remove_residents_and_initial_cleaning, 
    filter_by_iqr_and_thresholds, 
    remove_outliers_if, 
    remove_outliers_zscore,
    remove_outliers_zscore_percent, # New
    remove_outliers_iqr_percent,    # New
    remove_outliers_ldis_proxy,
    remove_outliers_kmedoids_proxy
)
from rfm_utils import assign_segments, LABEL_MAPPING

# Suppress warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
FAST_MODE = True  # Set to False for rigorous paper results (Takes hours)
EPOCHS = 5 if FAST_MODE else 70
BATCH_SIZE = 1048
LEARNING_RATE = 6e-5
N_SPLITS = 10 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Model Definition (Inline to ensure self-contained script) ---
class MulticlassClassificationWithAttentionHead(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassificationWithAttentionHead, self).__init__()
        self.layer_1 = nn.Linear(num_feature, 1024)
        self.layer_2 = nn.Linear(1024, 512)
        self.layer_3 = nn.Linear(512, 256)
        self.layer_5 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.multihead_attention_before = nn.MultiheadAttention(embed_dim=1024, num_heads=2)
        self.multihead_attention_after = nn.MultiheadAttention(embed_dim=64, num_heads=2)
        self.layer_out = nn.Linear(64, num_class)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.batchnorm5 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = x.unsqueeze(0)
        x, _ = self.multihead_attention_before(x, x, x)
        x = x.squeeze(0)
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.layer_5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = x.unsqueeze(0)
        x, _ = self.multihead_attention_after(x, x, x)
        x = x.squeeze(0)
        x = self.layer_out(x)
        return x

class SimpleDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__ (self):
        return len(self.X_data)

def train_and_evaluate_fold(X_train, y_train, X_test, y_test, num_classes):
    """
    Trains model and returns dictionaries of metrics (F1, Prec, Recall)
    Both Macro and Per-Class.
    """
    # 1. Prepare DataLoaders with WeightedSampler
    train_dataset = SimpleDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset = SimpleDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    
    target_list = torch.tensor(y_train)
    class_vals, counts = np.unique(y_train, return_counts=True)
    
    # Weight calculation
    weight_map = {c: 1.0/cnt for c, cnt in zip(class_vals, counts)}
    # Weights for sampler (per sample)
    weights_all = [weight_map[t.item()] for t in target_list]
    
    weighted_sampler = WeightedRandomSampler(
        weights=weights_all,
        num_samples=len(weights_all),
        replacement=True
    )
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Weights for Loss (per class)
    # Ensure tensor covers all classes 0..num_classes-1
    loss_weights = torch.zeros(num_classes)
    for c, w in weight_map.items():
        loss_weights[c] = w
        
    model = MulticlassClassificationWithAttentionHead(num_feature=X_train.shape[1], num_class=num_classes)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training
    model.train()
    for e in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list.extend(y_pred_tags.cpu().numpy())
            y_true_list.extend(y_batch.numpy())
            
    # Metrics
    # Macro
    f1_macro = f1_score(y_true_list, y_pred_list, average='macro')
    prec_macro = precision_score(y_true_list, y_pred_list, average='macro')
    rec_macro = recall_score(y_true_list, y_pred_list, average='macro')
    
    # Per Class (returns array)
    f1_class = f1_score(y_true_list, y_pred_list, average=None, labels=list(range(num_classes)))
    prec_class = precision_score(y_true_list, y_pred_list, average=None, labels=list(range(num_classes)))
    rec_class = recall_score(y_true_list, y_pred_list, average=None, labels=list(range(num_classes)))
    
    return {
        'f1_macro': f1_macro, 'prec_macro': prec_macro, 'rec_macro': rec_macro,
        'f1_class': f1_class, 'prec_class': prec_class, 'rec_class': rec_class
    }

def evaluate_config(method_name, outlier_func, rfm_df):
    """
    Runs 10-fold CV for a specific configuration.
    Returns list of result dicts (one per fold).
    """
    print(f"Evaluating: {method_name}...")
    
    # 1. Outlier Removal
    if outlier_func:
        data = outlier_func(rfm_df.copy())
    else:
        data = rfm_df.copy()
        
    # 2. Prep
    # ALWAYS re-assign to ensure quantiles adapt to the cleaned distribution
    # Drop existing label/score columns so assign_segments recalculates fresh
    cols_to_drop = ['rfm_label', 'RFM_SCORE', 'R', 'F', 'M', 'rfm_sum']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns], errors='ignore')
    
    data = assign_segments(data)
    
    data['cluster'] = data['rfm_label'].map(LABEL_MAPPING)
    data = data.dropna(subset=['cluster'])
    
    X = data[['frequency', 'recency', 'monetary']].values
    y = data['cluster'].values
    num_classes = len(LABEL_MAPPING)
    
    # 3. CV
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_results = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        res = train_and_evaluate_fold(X_train, y_train, X_test, y_test, num_classes)
        fold_results.append(res)
        
    return fold_results

def print_table_row(name, metrics, metric_key, class_metric_key):
    """
    Formats row like: MacroMean +/- Std | Class1Mean +/- Std | ...
    """
    # Macro
    vals = [r[metric_key] for r in metrics]
    mean = np.mean(vals)
    std = np.std(vals)
    
    row_str = f"{name:<20} | {mean:.2f} \u00B1 {std:.2f}"
    
    # Per Class
    # We aggregate per class across folds
    num_classes = len(LABEL_MAPPING)
    class_means = []
    class_stds = []
    
    for c in range(num_classes):
        c_vals = [r[class_metric_key][c] for r in metrics]
        c_mean = np.mean(c_vals)
        c_std = np.std(c_vals)
        row_str += f" | {c_mean:.2f} \u00B1 {c_std:.2f}"
        
    print(row_str)

def main():
    print("Loading Base Data...")
    raw_df = load_and_clean_data(base_path="../original")
    clean_df = remove_residents_and_initial_cleaning(raw_df)
    rfm_df = aggregate_rfm_data(clean_df)
    rfm_df = assign_segments(rfm_df) # Ensure labels present
    
    # Define Configurations (Method x Percent)
    # 0%
    configs = [("None (0%)", None)]
    
    # Methods to test for 2, 5, 10
    # Note: IQR is fixed, but let's assume it represents one of the configs or just exclude if not flexible.
    # The paper mentions IQR for 2,5,10. My `filter_by_iqr_and_thresholds` is fixed standard IQR.
    # I will skip IQR variations and just use standard IQR as one entry if requested, OR assume IQR doesn't have percent param.
    # For Z-score, IF, LDIS, K-Medoids, we have params.
    
    # Updated configurations using new percent-based functions from outlier_removal.py
    # Ensure these are imported: 
    # from outlier_removal import remove_outliers_zscore_percent, remove_outliers_iqr_percent

    percents = [0.02, 0.05, 0.10]
    
    methods_factories = {
        "Isolation Forest": lambda p: lambda d: remove_outliers_if(d, contamination=p),
        "Z-score": lambda p: lambda d: remove_outliers_zscore_percent(d, percent_remove=p),
        "IQR": lambda p: lambda d: remove_outliers_iqr_percent(d, percent_remove=p),
        "LDIS": lambda p: lambda d: remove_outliers_ldis_proxy(d, contamination=p),
        "k-medoids": lambda p: lambda d: remove_outliers_kmedoids_proxy(d, percent_remove=p)
    }
    
    # Build list
    for p in percents:
        pct_str = int(p*100)
        for m_name, factory in methods_factories.items():
            configs.append((f"{m_name} ({pct_str}%)", factory(p)))
            
    # Add Standard (Fixed) versions if needed for reference, but table focuses on %
    # configs.append(("IQR (Standard)", filter_by_iqr_and_thresholds))
    
    # Run Evaluations
    all_results = {}
    
    for name, func in configs:
        all_results[name] = evaluate_config(name, func, rfm_df)
        
    # SAVE RESULTS
    save_results_to_csv(all_results)
        
    print("\n" + "="*50)
    print("RESULTS GENERATION (Mean \u00B1 Std)")
    print("="*50)
    
    # Table 5: F1 Scores
    print("\nTable 5: F1-Scores")
    print_table_row("Method", [], "", "") # Header placeholder logic needed?
    # Simpler print:
    for name, res in all_results.items():
        print_table_row(name, res, 'f1_macro', 'f1_class')
        
    # Table 8: Precision
    print("\nTable 8: Precision")
    for name, res in all_results.items():
        print_table_row(name, res, 'prec_macro', 'prec_class')

    # Table 9: Recall
    print("\nTable 9: Recall")
    for name, res in all_results.items():
        print_table_row(name, res, 'rec_macro', 'rec_class')
        
    # Hypothesis Testing (Friedman on F1 Macro)
    print("\n" + "="*50)
    print("HYPOTHESIS TESTING (Table 5 Results)")
    print("="*50)
    
    # Gather Macro F1 lists
    f1_arrays = []
    names = []
    for name, res in all_results.items():
        f1_vals = [r['f1_macro'] for r in res]
        f1_arrays.append(f1_vals)
        names.append(name)
        
    
    stat, p_value = stats.friedmanchisquare(*f1_arrays)
    print(f"Friedman Test (Macro F1): Chi2={stat:.4f}, p={p_value:.6e}")
    
    if p_value < 0.05:
        print("Significant difference found. Post-hoc Wilcoxon vs Best:")
        # Find best mean
        means = [np.mean(arr) for arr in f1_arrays]
        best_idx = np.argmax(means)
        best_name = names[best_idx]
        best_scores = f1_arrays[best_idx]
        print(f"Best Method: {best_name} (Mean F1={means[best_idx]:.4f})")
        
        for i, name in enumerate(names):
            if i == best_idx: continue
            s, p = stats.wilcoxon(best_scores, f1_arrays[i])
            print(f"  vs {name:<20}: p={p:.6e}")

def save_results_to_csv(all_results, filename="model_evaluation_results.csv"):
    """
    Flattens the all_results dict into a dataframe and saves to CSV.
    Structure: Method | Fold | Metric | Value ...
    """
    print(f"\nSaving detailed results to {filename}...")
    records = []
    
    # Get class names for column headers
    class_names = [label for label in sorted(LABEL_MAPPING.keys(), key=lambda k: LABEL_MAPPING[k])]
    
    for method_name, fold_data_list in all_results.items():
        for fold_idx, res in enumerate(fold_data_list):
            # Base record
            rec = {
                "Method": method_name,
                "Fold": fold_idx,
                "F1_Macro": res['f1_macro'],
                "Precision_Macro": res['prec_macro'],
                "Recall_Macro": res['rec_macro']
            }
            
            # Add Per-Class Metrics
            # F1
            for i, c_name in enumerate(class_names):
                rec[f"F1_{c_name}"] = res['f1_class'][i]
                
            # Precision
            for i, c_name in enumerate(class_names):
                rec[f"Prec_{c_name}"] = res['prec_class'][i]
                
            # Recall
            for i, c_name in enumerate(class_names):
                rec[f"Rec_{c_name}"] = res['rec_class'][i]
                
            records.append(rec)
            
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows to {filename}.")

if __name__ == "__main__":
    main()
