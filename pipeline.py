# given from the dataset website lol thanks UC Irvice Machine Learning Repository
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import time
  
# ----------- metadata section -------------------------------------------------------------- #
# ----------- Everyone run this section and read output to start. ---------------- #
# ----------- Important information about the data that we should know going in. ------------ #

#diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 

# data (as pandas dataframes) 
#X = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
#y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets 

# metadata 
#print(diabetes_130_us_hospitals_for_years_1999_2008.metadata) 
  
# variable information 
#print(diabetes_130_us_hospitals_for_years_1999_2008.variables) 


df = pd.read_csv('diabetic_data.csv', keep_default_na=False, na_values='?', low_memory=False)

# --- Row filters (all of them, first) ---
df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]
df = df[df['gender'] != 'Unknown/Invalid']

# --- Target and groups, built after filtering so they stay aligned ---
y = (df['readmitted'] == '<30').astype(int)
groups = df['patient_nbr']

# --- Drop columns that shouldn't be features ---
df = df.drop(columns=['weight', 'encounter_id', 'patient_nbr', 'readmitted'])
constant_cols = [c for c in df.columns if df[c].nunique() == 1]
print('Constant columns dropped:', constant_cols)   # expect examide, citoglipton
df = df.drop(columns=constant_cols)

# --- Recode "not tested" ---
for col in ['A1Cresult', 'max_glu_serum']:
    df[col] = df[col].replace('None', 'Not Tested')

# --- Coded-unknown IDs, treated as categories ---
unknown_map = {
    'admission_type_id': [5, 6, 8],
    'admission_source_id': [9, 17, 20],
    'discharge_disposition_id': [18, 25],
}
for col, codes in unknown_map.items():
    df[col] = df[col].replace(codes, -1).astype(str)

# --- Truly missing categoricals ---
for col in ['race', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].fillna('Unknown').astype(str)

# --- Sanity checks ---
assert len(df) == len(y) == len(groups)
print(df.isna().sum().sum(), 'NaNs remaining')   # should be 0