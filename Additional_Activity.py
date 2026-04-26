import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv('spotify_top_1000_tracks.csv')

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['year'] = df['release_date'].dt.year

df_filtered = df[df['year'] >= 2023].copy()

print(f"✅ Data ready! Analyzing {len(df_filtered)} songs from 2023 to present.")

plt.figure(figsize=(10, 8))
numeric_cols = df_filtered.select_dtypes(include=['float64', 'int64'])
correlation = numeric_cols.corr()

sns.heatmap(correlation, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)

plt.title(f'Feature Correlation (2023-Present)\nStudent: FERNANDEZ', fontsize=12)
plt.savefig('additional_heatmap.png')
print("✅ Heatmap saved as 'additional_heatmap.png'")
plt.show()

plt.figure(figsize=(10, 6))

sns.violinplot(x='year', y='energy', data=df_filtered, hue='year', palette='YlGnBu', inner='quartile')

plt.title('Energy Distribution (2023 onwards) - Quartile View', fontsize=12)
plt.savefig('additional_violin_plot.png')
print("✅ Violin Plot saved as 'additional_violin_plot.png'")
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x='energy', y='loudness', data=df_filtered, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})

plt.title('Relationship: Energy vs. Loudness (2023-Present)', fontsize=12)
plt.savefig('additional_scatter.png')
print("✅ Scatter Plot saved as 'additional_scatter.png'")
plt.show()