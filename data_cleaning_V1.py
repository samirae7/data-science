# %%
import seaborn as sns

# Import pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# %%
df_1 = pd.read_csv("./CTG.csv", sep=";")

data = df_1

# %%
data.dtypes

# %%
data.describe()

# %%
data.isna().sum(axis=1).sum()
data.isna().any(axis=1).sum()

# %%
fig, ax = plt.subplots(figsize=(14, 5)) #ax = the plotting area where the heatmap will be drawn.
sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
ax.set_title("Missing Values Heatmap\n(yellow = missing, purple = present)", fontsize=13)
plt.tight_layout()
plt.show()

"""
# %%
data = data.dropna(how="all") #drops empty row
data.head()

# %%
data = data.dropna(subset=["CLASS","NSP"]) #A row without a label can’t be used for training or evaluation.
# Listwise/ dropping entire rows
"""

# %%
na_per_row = data.isna().sum(axis=1)
print(na_per_row[na_per_row > 0].value_counts().sort_index())

# %%
tabelle = data[data.isna().sum(axis=1) == 1].copy()
tabelle["fehlendes Feld"] = data.isna().idxmax(axis=1)
tabelle

# %% [markdown]
# 22 rows have exactly one missing value. These are isolated gaps, scattered across the dataset, which is what random missingness looks like.
# 3 rows have 20, 22 and 23 missing values. These are nearly empty rows. With about 37 columns left, more than half of each row is blank. They are damaged or incomplete records, not random gaps, so drop them. There’s nothing to salvage.

"""
# %%
data = data[na_per_row < 10] # dros rows with 20, 22, 23 missing values. 


# %%
data_noNA = data.dropna(axis=0) #drops all rows with missingness
data_noNA.shape

# %%
data_noNA.drop(columns=["FileName", "Date", "SegFile"], inplace=True)
data_noNA.shape

# %% [markdown]
# FileName, Date and SegFile are identifiers, not measurements. Drop them before modelling, or the model may learn from them by accident.

# %%
data_noNA.isna().sum(axis=1).sum()
data_noNA.dtypes

# %%
text_cols = data_noNA.select_dtypes(include="object").columns
print(list(text_cols))

for c in text_cols:
    data_noNA[c] = pd.to_numeric(
        data_noNA[c].astype(str).str.strip().str.replace(",", "."),
        errors="coerce"
    )

print(data_noNA.dtypes.value_counts())
print(data_noNA.isna().sum().sum())
print(data_noNA[data_noNA.isna().any(axis=1)])

# %%
data_noNA = data_noNA.dropna()

# %%
skip = ["CLASS","NSP","A","B","C","D","E","AD","DE","LD","FS","SUSP","Tendency"]
num_cols = [c for c in data_noNA.columns if c not in skip]

rows = []
for c in num_cols:
    q1, q3 = data_noNA[c].quantile([.25, .75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    n = ((data_noNA[c] < lo) | (data_noNA[c] > hi)).sum()
    rows.append((c, data_noNA[c].min(), data_noNA[c].max(), n, round(100*n/len(data_noNA), 1)))
summary = pd.DataFrame(rows, columns=["col","min","max","n_outliers","pct"])
print(summary.sort_values("pct", ascending=False))

# %% [markdown]
# Drop DR
# 
# Its min and max are both 0, so it’s constant and carries no information. Remove it before modelling (DS is nearly constant too, with a max of 1, but I’d keep it for now).
# 
# Min, Max, Mean, Median and Mode describe a heart-rate histogram. As far as I know, a fetal heart rate of 100 to 200 bpm is the normal range, so values in the hundreds of thousands can’t be real. Width (the histogram’s range) can’t be negative. ASTV is a percentage, so it can’t be below 0 or above 100. These look like formatting or data-entry errors, such as a misplaced decimal separator. The huge Mean, Min and Max values may well come from the same few rows, so check them together.
# 
# Probably fine: high percentages that are expected
# 
# Nzeros (23.5%), FM (14.6%), ALTV (14.5%), Variance, DP, AC and DL have lots of IQR outliers, but their values are plausible: a few counts, a percentage up to 91, movements up to 564. These columns are skewed and mostly zeros. The IQR rule flags every non-zero value as unusual when most values are 0. These are real observations, so keep them.
# 
# LB, LBE, b and e have no outliers, and their ranges look sensible (a baseline of 106 to 160 bpm).

# %%
bad = (
    (data_noNA[["Min","Max","Mean","Median","Mode"]] > 300).any(axis=1)
    | (data_noNA["Width"] < 0) | (data_noNA["Width"] > 300)
    | (data_noNA["ASTV"] < 0) | (data_noNA["ASTV"] > 100)
)
print(bad.sum())
print(data_noNA[bad][["Min","Max","Mean","Median","Mode","Width","ASTV"]])

# %%
data_noNA.shape

# %%
data_clean = data_noNA[~bad].drop(columns="DR")
data_clean.shape

# %%
cols_without_time = [c for c in data_clean.columns if c not in ["b", "e"]]
print(data_clean.duplicated().sum(), "duplicates including b and e")
print(data_clean.duplicated(subset=cols_without_time).sum(), "duplicates ignoring b and e")

# %%
data_clean = data_clean.drop_duplicates(subset=cols_without_time)
data_clean.shape

# %% [markdown]
# The output means:
# 
# 8 rows are exact copies of another row, including b and e. These are certainly duplicates, so drop them.
# 2 more rows (10 minus 8) have identical measurements but different b and e. They are probably different recording segments that happen to look the same. That is unusual across about 37 measurement columns, but not impossible, so these two are a judgment call.
# 
# My recommendation: drop the 8 exact copies for sure. For the other 2, the decision doesn’t matter much, since 2 rows out of about 2,000 won’t change anything. I’d choose based on your plan:
# 
# If you will drop b and e before modelling (I’d do that, because they’re time indices, not measurements), drop all 10. Otherwise those 2 rows would be exact duplicates in the data your model sees, with the train/test leakage problem I mentioned.
# If you keep b and e, drop only the 8 and keep the 2.

# %% [markdown]
# Quick Summary of what was dropped:
# 
# Rows
# 
# The original file had 2,127 rows. After cleaning, data_clean has 2052 rows, so 75 were removed (about 3.5%). The removals were:
# 
# 1 completely empty row, which carried no information.
# 18 rows with a missing label (CLASS/NSP), because rows without a label can’t be used for training or evaluation.
# 3 nearly empty rows (20 to 23 values missing out of about 37), which were damaged records.
# 22 rows with a single missing value. The gaps were spread across columns and looked random, and the rows were under 2% of the data, so I dropped them instead of imputing.
# 21 rows with physically impossible values, such as Mean, Median, Mode, Min and Max in the hundreds of thousands, negative Width, and ASTV below 0 or above 100. These looked like formatting or data-entry errors.
# 10 duplications
# 
# Columns
# 
# The original file had 40 columns. 36 are left.
# 
# FileName, Date, SegFile were removed because they are identifiers, not measurements.
# DR was removed because it is constant (always 0), so it carries no information.
# 
# No columns were dropped for missing values, because the missingness was tiny (1 to 9 values per column) and scattered. Dropping rows was the better option.
# 
# 
"""

