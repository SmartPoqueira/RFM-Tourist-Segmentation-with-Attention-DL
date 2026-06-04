
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys

# Load results
try:
    df = pd.read_csv("model_evaluation_results.csv")
except FileNotFoundError:
    print("Error: model_evaluation_results.csv not found.")
    sys.exit(1)

# Helper to format Mean +/- Std
def format_mean_std(mean, std):
    return f"{mean:.2f} $\\pm$ {std:.2f}"

def get_clean_method_name_and_pct(full_name):
    # Parses "Isolation Forest (2%)" -> ("Isolation Forest", 2)
    # Parses "None (0%)" -> ("None", 0)
    # Parses "Z-score (Threshold=3)" -> ("Z-score", "Fixed")
    if "(" not in full_name:
        return full_name, "?"
    
    parts = full_name.split("(")
    name = parts[0].strip()
    param = parts[1].replace(")", "").strip()
    
    if "%" in param:
        try:
            val = int(param.replace("%", ""))
            return name, val
        except:
            return name, param
    elif "Threshold" in param:
        # Map fixed Z-score to something? Or just exclude if not in target list
        return name, "Fixed"
    elif "Standard" in param:
        return name, "Fixed"
    
    # Handle baseline
    if "0%" in param:
        return name, 0
    
    return name, param

# Add Pct and BaseName columns
df['BaseName'], df['Pct'] = zip(*df['Method'].apply(get_clean_method_name_and_pct))

# Define target percentages order: 0, 2, 5, 10
target_pcts = [0, 2, 5, 10]

# Metrics groups
metric_types = {
    "F1": "F1",
    "Precision": "Prec",
    "Recall": "Rec"
}

# --- 1. GENERATE LATEX TABLES ---

for metric_name, prefix in metric_types.items():
    print(f"\n\n% --- TABLE FOR {metric_name.upper()} ---")
    print("\\begin{sidewaystable}")
    print("\\centering")
    print("\\caption{\\\\ Results for the " + metric_name + "-score metric (Mean $\\pm$ Std)}")
    # Adjust column count: % | Method | Macro | 11 classes... | Time
    # 1 + 1 + 1 + 11 + 0 (Time missing) = 14 columns
    print("\\resizebox{19cm}{!}{%")
    print("\\begin{tabular}{cccccccccccccc}")
    print("\\hline")
    # Header 1
    print("\\textbf{\\begin{tabular}[c]{@{}c@{}}\\% outliers\\\\ removed\\end{tabular}} & "
          "\\textbf{Method} & "
          "\\textbf{\\begin{tabular}[c]{@{}c@{}}Macro-averaged\\\\ " + metric_name + "-score\\end{tabular}} & "
          "\\multicolumn{11}{c}{\\textbf{\\begin{tabular}[c]{@{}c@{}}" + metric_name + "-score \\\\ class (support)\\end{tabular}}} & "
           # Time removed as we don't have it
          "\\\\ \\hline")
    
    # Header 2 (Classes) - Hardcode or extract?
    # We can extract from columns starting with prefix_
    cols = [c for c in df.columns if c.startswith(prefix + "_") and not c.endswith("Macro")]
    # Sort them to match paper order if possible? 
    # Paper order: champions, loyal, potential, new, promising, need att, about to sleep, can't lose, at risk, hibernating, lost
    # My labels might be sorted alphabetically or by ID. 
    # Let's try to map my column names to paper order.
    # Assuming my csv cols are F1_0, F1_1 or F1_champions if I managed to map them.
    # Check csv content logic: "rec[f'F1_{c_name}'] = ..."
    # I trust the class ordering in the list.
    
    class_order_paper = [
        "champions", "loyal customers", "potential loyalist", "new customers", "promising", 
        "need attention", "about to sleep", "can't lose them", "at risk", "hibernating", "lost"
    ]
    # Try to find matching columns
    sorted_cols = []
    for paper_cat in class_order_paper:
        # Find col that contains this string (ignoring case/spaces)
        found = None
        for c in cols:
            # c is like F1_06_promising or F1_promising
            if paper_cat.replace(" ", "_") in c or paper_cat in c: 
                found = c
                break
            # Flexible matching
            if paper_cat.split()[0] in c and paper_cat.split()[-1] in c:
                found = c
                break
        
        if found:
            sorted_cols.append(found)
        else:
            # Fallback: just take remaining? Or print error?
            pass

    if len(sorted_cols) != 11:
        # Fallback to just all cols sorted
        sorted_cols = sorted(cols)

    # Subheader row
    header_row = "& & & " + " & ".join([f"\\textbf{{{c.replace(prefix+'_', '').replace('_', ' ')}}}" for c in sorted_cols]) + " \\\\"
    print(header_row)
    
    # Rows
    current_pct = -1
    
    # Filter for target percentages
    subset = df[df['Pct'].isin(target_pcts)]
    
    # Aggregate Mean +/- Std
    # Group by BaseName, Pct
    # structure: Pct -> List of methods
    
    for pct in target_pcts:
        print("\\hline")
        # Get methods for this pct
        pct_data = subset[subset['Pct'] == pct]
        if pct_data.empty: continue
        
        # Sort methods: None, then IF, Z, IQR, LDIS, Medoids
        method_order = ["None", "Isolation Forest", "Z-score", "IQR", "LDIS", "k-medoids"]
        
        # Group to get stats
        grouped = pct_data.groupby('BaseName')
        
        # Iterate in specific order
        first_row = True
        
        for m_name in method_order:
            if m_name not in grouped.groups:
                continue
                
            stats_df = grouped.get_group(m_name)
            
            # Means and Stds
            macro_mean = stats_df[f"{metric_name}_Macro"].mean()
            macro_std = stats_df[f"{metric_name}_Macro"].std()
            
            # Class stats
            class_cells = []
            for c in sorted_cols:
                m = stats_df[c].mean()
                s = stats_df[c].std()
                class_cells.append(format_mean_std(m, s))
                
            # Row Start
            if first_row:
                pct_cell = f"\\multirow{{{len(grouped)}}}{{*}}{{{pct}}}"
                first_row = False
            else:
                pct_cell = ""
                
            row_str = f"{pct_cell} & {m_name} & {format_mean_std(macro_mean, macro_std)} & " + " & ".join(class_cells) + " \\\\"
            print(row_str)

    print("\\hline")
    print("\\end{tabular}%")
    print("}")
    print("\\label{tab:" + metric_name.lower() + "-table}")
    print("\\end{sidewaystable}")


# --- 2. STATISTICAL TESTS (Friedman + Wilcoxon) ---

print("\n\n% --- STATISTICAL TEST RESULTS ---")
print("% Methodology: Friedman Test on F1-Macro (Fold-level) followed by Wilcoxon vs Best")

# Prepare data for Friedman: Dataframe where Index=Fold, Columns=MethodName(Full)
# Pivot
f1_df = df.pivot(index='Fold', columns='Method', values='F1_Macro')

# Run Friedman
# args: list of arrays (one per method)
# We need to ensure columns are consistent
methods = f1_df.columns.tolist()
data_arrays = [f1_df[m].values for m in methods]

stat, p_value = stats.friedmanchisquare(*data_arrays)

print(f"\nDetailed Stats Report:")
print(f"Friedman Test (N={len(methods)} algorithms, k={len(f1_df)} folds):")
print(f"Chi-squared = {stat:.4f}")
print(f"p-value = {p_value:.6e}")

text_block = ""

if p_value < 0.05:
    text_block += f"The Friedman test revealed significant differences in F1-Macro performance among the outlier removal methods ($\\chi^2 = {stat:.2f}$, $p = {p_value:.2e}$), rejecting the null hypothesis of equal performance.\n\n"
    
    # Post-hoc: Find best mean
    means = f1_df.mean().sort_values(ascending=False)
    best_method = means.index[0]
    best_score = means.iloc[0]
    
    text_block += f"To identify which differences were significant, we conducted post-hoc Wilcoxon Signed-Rank tests comparing the top-performing method, \\textbf{{{best_method}}} (Mean F1 = {best_score:.4f}), against all others. "
    text_block += "Results indicate that:\n\\begin{itemize}\n"
    
    comparisons = []
    
    best_data = f1_df[best_method]
    
    for m in means.index[1:]: # Compare against rest
        stat_w, p_w = stats.wilcoxon(best_data, f1_df[m])
        # Bonferroni-Holm correction could be applied here manually or just report raw p
        # For simplicity in this script output I report raw p, but mention significance level
        signif = "significantly different" if p_w < 0.05 else "not significantly different"
        emoji = "(*)" if p_w < 0.05 else "(ns)"
        
        comparisons.append((m, p_w))
        text_block += f"  \\item vs {m}: $p = {p_w:.2e}$ {emoji} - {signif}.\n"

    text_block += "\\end{itemize}"
    
else:
    text_block += f"The Friedman test did not find statistically significant differences in F1-Macro performance among the outlier removal methods ($\\chi^2 = {stat:.2f}$, $p = {p_value:.3f}$). This suggests that the model is robust to the choice of outlier removal technique, or that the differences in outliers removed (0-10\\%) were not sufficient to drastically alter the classification boundaries statistically."

print("\n\n" + "="*40)
print("SUGGESTED LATEX TEXT FOR SECTION 4.4")
print("="*40)
print(text_block)
