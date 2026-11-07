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
"""
# %%
fig, ax = plt.subplots(figsize=(14, 5)) #ax = the plotting area where the heatmap will be drawn.
sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
ax.set_title("Missing Values Heatmap\n(yellow = missing, purple = present)", fontsize=13)
plt.tight_layout()
plt.show()
"""
"""
# %%
data = data.dropna(how="all") #drops empty row
data.head()

# %%
data = data.dropna(subset=["CLASS","NSP"]) #A row without a label can’t be used for training or evaluation.
# Listwise/ dropping entire rows
"""
data_2 = data.dropna(how="all") #drops empty row

dropped = data_2[data_2[["CLASS","NSP"]].isna().any(axis=1)]
print(len(dropped))

data_3 = data_2.dropna(subset=["CLASS","NSP"]) #A row without a label can’t be used for training or evaluation.
# Listwise/ dropping entire rows

print(len(data_2), "rows before")
print(len(data_3), "rows after")
print(len(data_2) - len(data_3), "rows dropped")

# %%
na_per_row = data_3.isna().sum(axis=1)
print(na_per_row[na_per_row > 0].value_counts().sort_index())
# 22 rows have exactly one missing value. These are isolated gaps, scattered across the dataset, which is what random missingness looks like.
# 3 rows have 20, 22 and 23 missing values. These are nearly empty rows. With about 37 columns left, more than half of each row is blank. They are damaged or incomplete records, not random gaps, so drop them. There’s nothing to salvage.


# %%
tabelle = data_3[data_3.isna().sum(axis=1) == 1].copy()
tabelle["fehlendes Feld"] = data_3.isna().idxmax(axis=1)
print(tabelle)
print(len(tabelle))

data_4 = data_3[na_per_row < 10] # dros rows with 20, 22, 23 missing values. 


#Does missingness relate to the label?
from scipy.stats import chi2_contingency
data_4["has_na"] = data_4.isna().any(axis=1)
ct = pd.crosstab(data_4["has_na"], data_4["NSP"])
print(ct)
print("p =", chi2_contingency(ct)[1])
#A large p-value (above 0.05) means no evidence that missingness depends on the class.

#Does missingness relate to the other measurements (the MCAR check)?
from scipy.stats import mannwhitneyu

data_4 = data_4.copy()
data_4["has_na"] = data_4.isna().any(axis=1)

num_cols = [c for c in data_4.columns if c not in ["has_na", "CLASS", "NSP"]]
res = []
for c in num_cols:
    a = data_4.loc[data_4["has_na"], c].dropna()
    b = data_4.loc[~data_4["has_na"], c].dropna()
    if len(a) < 3 or data_4[c].nunique() < 2:
        continue                      # too few values or constant column
    try:
        res.append((c, mannwhitneyu(a, b).pvalue))
    except Exception as e:
        print(c, "->", type(e).__name__, e)

res = pd.DataFrame(res, columns=["col", "p"]).sort_values("p")
print(res)

 

# %%
data_noNA = data_4.dropna(axis=0) #drops all rows with missingness
data_noNA.shape

print("Check for no NAN")
print(data_noNA.isna().any(axis=1).sum())

print("Rows dropped: ")
print(len(data_4), "rows before")
print(len(data_noNA), "rows after")
print(len(data_4) - len(data_noNA), "rows dropped")
# %%

data_5 = data_noNA
data_5.drop(columns=["FileName", "Date", "SegFile"], inplace=True)
data_5.shape

# %% [markdown]
# FileName, Date and SegFile are identifiers, not measurements. Drop them before modelling, or the model may learn from them by accident.

# %%
print("WHat are the data types: ")
print(data_5.dtypes)

# %%
# turns object to numeric values
text_cols = data_5.select_dtypes(include="object").columns #Finds all columns that pandas stored as text (dtype object)
print(list(text_cols))

for c in text_cols:
    data_5[c] = pd.to_numeric(
        data_5[c].astype(str).str.strip().str.replace(",", "."),
        errors="coerce" # If a value still can’t be read as a number (for example "?"), coerce turns it into NaN instead of raising an error.
    )

print("check: ")
print(data_5.dtypes.value_counts())
print(data_5.isna().sum().sum())
print(data_5[data_5.isna().any(axis=1)])


# %%
data_6 = data_5.dropna() #drops rows where it couldnt be converted into numeric values
print(len(data_5), "rows before")
print(len(data_6), "rows after")
print(len(data_5) - len(data_6), "rows dropped")
# %%
print()
print("Check for outliers: ")
skip = ["CLASS","NSP","A","B","C","D","E","AD","DE","LD","FS","SUSP","Tendency"]
num_cols = [c for c in data_6.select_dtypes(include="number").columns if c not in skip]

rows = []
for c in num_cols: # how many values fall outside the usual range (1.5 × IQR beyond the quartiles)
    q1, q3 = data_6[c].quantile([.25, .75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    n = ((data_6[c] < lo) | (data_6[c] > hi)).sum()
    rows.append((c, data_6[c].min(), data_6[c].max(), n, round(100*n/len(data_6), 1)))
summary = pd.DataFrame(rows, columns=["col","min","max","n_outliers","pct"])
print(summary.sort_values("pct", ascending=False)) # table of each column’s min, max, number of outliers and percentage of outliers, sorted from most to least.


# %% [markdown]
# Drop DR: Its min and max are both 0, so it’s constant and carries no information. Remove it before modelling (DS is nearly constant too, with a max of 1, but I’d keep it for now).
# 
# Min, Max, Mean, Median and Mode describe a heart-rate histogram. As far as I know, a fetal heart rate of 100 to 200 bpm is the normal range, so values in the hundreds of thousands can’t be real. 
# 
# Width (the histogram’s range) can’t be negative. 
# 
# ASTV is a percentage, so it can’t be below 0 or above 100. These look like formatting or data-entry errors, such as a misplaced decimal separator. The huge Mean, Min and Max values may well come from the same few rows, so check them together.
# 
# Probably fine: high percentages that are expected
# Nzeros (23.5%), FM (14.6%), ALTV (14.5%), Variance, DP, AC and DL have lots of IQR outliers, but their values are plausible: a few counts, a percentage up to 91, movements up to 564. These columns are skewed and mostly zeros. The IQR rule flags every non-zero value as unusual when most values are 0. These are real observations, so keep them.
# LB, LBE, b and e have no outliers, and their ranges look sensible (a baseline of 106 to 160 bpm).

# %%
bad = ( #Starts building a True/False column called bad, with one entry per row.
    (data_6[["Min","Max","Mean","Median","Mode"]] > 300).any(axis=1) #any(axis=1) collapses each row to a single True if at least one of those five values exceeds 300
    | (data_6["Width"] < 0) | (data_6["Width"] > 300) #The | means “or”, so a row is flagged if any condition is true.
    | (data_6["ASTV"] < 0) | (data_6["ASTV"] > 100)
)
print(bad.sum())
print(data_6[bad][["Min","Max","Mean","Median","Mode","Width","ASTV"]])


# %%
data_7 = data_6[~bad].drop(columns="DR") #Keeps only the rows that are not bad. Removes the column DR from the result
print(len(data_6), "rows before")
print(len(data_7), "rows after")
print(len(data_6) - len(data_7), "rows dropped")


# %%
cols_without_time = [c for c in data_7.columns if c not in ["b", "e"]]
print(data_7.duplicated().sum(), "duplicates including b and e")
print(data_7.duplicated(subset=cols_without_time).sum(), "duplicates ignoring b and e")

# %%
data_8 = data_7.drop_duplicates(subset=cols_without_time)
data_8.shape

print(len(data_7), "rows before")
print(len(data_8), "rows after")
print(len(data_7) - len(data_8), "rows dropped")

print()
print("Total rows dropped: ")
print(len(data), "rows before")
print(len(data_8), "rows after")
print(len(data) - len(data_8), "rows dropped")

print()
print("Final Check: ")
print(data_8.isna().sum(axis=1).sum())
print(data_8.duplicated().any())

output_path = 'CTG_cleaned_V1.csv'
data_8.to_csv(output_path, index=False)

"""
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
# 10 rows with a missing label (CLASS/NSP), because rows without a label can’t be used for training or evaluation.
# 3 nearly empty rows (20 to 23 values missing out of about 37), which were damaged records.
# 22 rows with a single missing value. The gaps were spread across columns and looked random, and the rows were under 2% of the data, so I dropped them instead of imputing.
# 8 rows where the content couldn't be turned into numerical values
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

