import pandas as pd

# Code to fix test dataset to be the same as the train dataset
df = pd.read_csv("../data/chilla_cleaned_2008_14.csv")
cols_of_interest = ['cohort', 'session', 'id', 'girl', 'date', 'dob', 'entry']
df = df[df.loc[:, cols_of_interest].notnull().all(axis=1)]
df['dob'] = pd.to_datetime(df.loc[:, 'dob'], format='%d%b%Y', errors='coerce')
df['date'] = pd.to_datetime(df.loc[:, 'date'], format='%d%b%Y', errors='coerce')
df['total_ab_sess'] = 0
grouped = df.groupby(['cohort', 'session', 'id'])
for name, group in grouped:
    df.loc[(df['cohort'] == name[0]) & (df['session'] == name[1]) & (
                df['id'] == name[2]), 'total_ab_sess'] = group.entry.sum()
df.to_csv("../data/chilla_2008_14_fixed.csv", index=False)