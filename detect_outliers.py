import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys

# === CONFIG ===
CSV_PATH = "humor_dataset.csv"   # change to your filename
GROUP_COL_CANDIDATES = ['class', 'label', 'type', 'category']  # possible names for first column
# === END CONFIG ===

# 1) Load CSV robustly (handles header or no header)
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    # try without header (assume 4 columns)
    df = pd.read_csv(CSV_PATH, header=None)
    print("Read file without header; assigned default column names")

# If columns don't include a message name, assume second column is the message
if 'message' in df.columns:
    msg_col = 'message'
else:
    msg_col = df.columns[1]    # second column (index 1)

# Choose a grouping column (first column)
group_col = None
for c in GROUP_COL_CANDIDATES:
    if c in df.columns:
        group_col = c
        break
if group_col is None:
    # fallback to first column
    group_col = df.columns[0]

# Choose last column as numeric label if needed
label_col = df.columns[-1]

print("Using columns:", "group_col=", group_col, "message_col=", msg_col, "label_col=", label_col)

# 2) Compute numeric features from text for boxplotting/outlier detection
def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

df[msg_col] = df[msg_col].apply(safe_str)

df['char_count'] = df[msg_col].str.len()
df['word_count'] = df[msg_col].str.split().apply(len)
df['unique_word_count'] = df[msg_col].str.split().apply(lambda tokens: len(set(tokens)) if tokens else 0)
df['punct_count'] = df[msg_col].str.count(r'[^\w\s]')    # punctuation and symbols
# uppercase ratio: fraction of characters that are uppercase (avoid divide by zero)
df['uppercase_ratio'] = df[msg_col].apply(lambda s: sum(1 for ch in s if ch.isupper()) / len(s) if len(s) > 0 else 0)

# 3) Outlier detection functions
def detect_outliers_iqr(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)

def detect_outliers_zscore(series, thresh=3.0):
    # fill NaN with median to avoid nan zscores
    s = series.fillna(series.median())
    z = np.abs(stats.zscore(s.astype(float)))
    return z > thresh

# 4) Apply detection for main metrics (you can add others)
metrics = ['char_count', 'word_count', 'punct_count', 'uppercase_ratio']
for m in metrics:
    df[f'outlier_{m}_iqr'] = detect_outliers_iqr(df[m])
    df[f'outlier_{m}_zscore'] = detect_outliers_zscore(df[m])

# 5) Save dataframe with outlier flags to CSV
df.to_csv("humor_dataset_with_outliers.csv", index=False)
print("Wrote: humor_dataset_with_outliers.csv")

# 6) Produce boxplots and highlight IQR outliers
sns.set_theme(style="whitegrid")
for metric in ['char_count', 'word_count']:
    plt.figure(figsize=(8,6))
    # boxplot grouped by first column (e.g., ham/spam)
    ax = sns.boxplot(x=group_col, y=metric, data=df, whis=1.5, showfliers=False)
    # draw all points (jitter) to see distribution
    sns.stripplot(x=group_col, y=metric, data=df, color='lightgray', jitter=True, size=4, alpha=0.6)
    # overlay detected IQR outliers in red
    out = df[df[f'outlier_{metric}_iqr']]
    if not out.empty:
        sns.stripplot(x=group_col, y=metric, data=out, color='red', jitter=False, size=6, marker='D')
    plt.title(f"{metric} by {group_col} (red = IQR outliers)")
    plt.tight_layout()
    fname = f"boxplot_{metric}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print("Saved boxplot:", fname)

# 7) Create a CSV with only the detected outliers (by char_count IQR by default)
outliers_df = df[df['outlier_char_count_iqr']].copy()
outliers_df.to_csv("humor_outliers_charcount_iqr.csv", index=False)
print("Wrote outlier subset (char_count IQR): humor_outliers_charcount_iqr.csv")

# Summary counts
for m in metrics:
    print(f"Metric {m}: IQR outliers={df[f'outlier_{m}_iqr'].sum()}, Zscore outliers={df[f'outlier_{m}_zscore'].sum()}")

print("Done.")