# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:34:12 2025

@author: User
"""
import pandas as pd
df = pd.read_csv('results/ARIMA/all_models_metrics.csv')

arima_rolling = df[df['split/model'] == 'ARIMA Rolling - Test']

arima_rolling.describe()

#%%

import matplotlib.pyplot as plt

arima_rolling['year'] = arima_rolling['dataset'].str.extract(r'(\d{4})').astype(int)
arima_rolling['symbol'] = arima_rolling['dataset'].str.extract(r'(\w+usdt_\d+d)')

plt.figure(figsize=(10, 5))
for symbol in arima_rolling['symbol'].unique():
    subset = arima_rolling[arima_rolling['symbol'] == symbol]
    plt.plot(subset['year'], subset['MAPE'], marker='o', label=symbol.upper())
plt.ylabel('MAPE (%)')
plt.xlabel('Year')
plt.title('ARIMA Rolling - MAPE per Year')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%%
plt.figure(figsize=(10, 5))
for symbol in arima_rolling['symbol'].unique():
    subset = arima_rolling[arima_rolling['symbol'] == symbol]
    plt.plot(subset['year'], subset['R²'], marker='s', label=symbol.upper())
plt.ylabel('R²')
plt.xlabel('Year')
plt.title('ARIMA Rolling - R² per Year')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
import seaborn as sns
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=arima_rolling, 
    x='MAPE', y='R²', hue='symbol', style='symbol', s=100
)
plt.xlabel('MAPE (%)')
plt.ylabel('R²')
plt.title('ARIMA Rolling: R² vs MAPE')
plt.grid(True)
plt.tight_layout()
plt.show()
