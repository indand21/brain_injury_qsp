#!/usr/bin/env python3
"""Check trajectory and mortality in simulated cohort."""

import pandas as pd

df = pd.read_csv('simulated_neurocritical_cohort_n2000.csv')

# Check trajectory class distribution
print('Trajectory class value counts:')
print(df['trajectory_class'].value_counts().sort_index())
print()

# Check mortality by trajectory class
print('Mortality by trajectory class:')
for tc in sorted(df['trajectory_class'].unique()):
    subset = df[df['trajectory_class'] == tc]
    mort_rate = subset['mortality_12m'].mean() * 100
    print(f'  Class {tc}: {mort_rate:.1f}% mortality (n={len(subset)})')
print()

# Check GCS distribution
print('GCS distribution:')
print(df['gcs_admission'].describe())
print()

# Check severity categories
mild = (df['gcs_admission'] >= 13).sum()
mod = ((df['gcs_admission'] >= 9) & (df['gcs_admission'] <= 12)).sum()
sev = (df['gcs_admission'] <= 8).sum()
total = len(df)
print(f'Mild (13-15): {mild} ({mild/total*100:.1f}%)')
print(f'Moderate (9-12): {mod} ({mod/total*100:.1f}%)')
print(f'Severe (3-8): {sev} ({sev/total*100:.1f}%)')
print()

# Overall mortality
print(f'Overall 12-month mortality: {df["mortality_12m"].mean()*100:.1f}%')
