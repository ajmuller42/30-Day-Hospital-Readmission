import pandas as pd
import numpy as np

df = pd.read_csv('diabetic_data.csv', keep_default_na=False, na_values='?', low_memory=False)

# ----- Data Cleaning ----- #

# drop if discharged bc of hospice or death. Those cannot be readmitted ever. Do not learn off this data
df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]
# drop if gender is not known
df = df[df['gender'] != 'Unknown/Invalid']

# set positive targe to be only readmissions within 30 days
y = (df['readmitted'] == '<30').astype(int)

#order columns based on missing rate, and print
missing = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
print(missing[missing > 0])
# checking more missing value rates. Missing number analysis not complete!!!!!!!!!!!!
print(df['gender'].value_counts())            # a few 'Unknown/Invalid' rows
print(df['admission_type_id'].value_counts()) # IDs 5, 6, 8 mean null/not available
print(df['admission_source_id'].value_counts()) # IDs 9, 15, 17, 20, 21 are null/unknown
print(df['discharge_disposition_id'].value_counts()) # 18, 25, 26 are unknown
print(df['A1Cresult'].value_counts(dropna=False))
print(df['max_glu_serum'].value_counts(dropna=False))