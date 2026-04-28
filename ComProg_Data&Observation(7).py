import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

student_name = "Yzaac Fernandez"
student_id = "20250930"
id_num = int(student_id[-3:])

color1 = f"#{((id_num * 7) % 256):02x}{((id_num * 3) % 256):02x}{((id_num * 5) % 256):02x}"
color2 = f"#{((id_num * 2) % 256):02x}{((id_num * 9) % 256):02x}{((id_num * 4) % 256):02x}"

np.random.seed(id_num)
n_samples = 500
df = pd.DataFrame({
    'duration_min': np.random.normal(3.5, 0.8, n_samples).clip(1, 8),
    'release_year': np.random.randint(1950, 2025, n_samples),
    'popularity': np.random.randint(0, 100, n_samples),
    'artist': np.random.choice(['Artist A', 'Artist B', 'Artist C', 'Artist D', 'Artist E'], n_samples),
    'track_name': [f"Track {i}" for i in range(n_samples)],
    'album': np.random.choice(['Album 1', 'Album 2', 'Album 3', 'Album 4', 'Album 5'], n_samples)
})
df['decade'] = (df['release_year'] // 10) * 10

df['duration_min'].plot(kind='hist', bins=30, color=color1, alpha=0.7)
plt.title(f"{student_name} ({student_id})\nHistogram of Song Duration")
plt.show()

sns.boxplot(x='decade', y='popularity', data=df, palette='coolwarm')
plt.title(f"{student_name} ({student_id})\nBoxplot of Popularity by Decade")
plt.show()

sns.countplot(y='artist', data=df, order=df['artist'].value_counts().head(10).index, palette='viridis')
plt.title(f"{student_name} ({student_id})\nTop 10 Artists by Song Count")
plt.show()

sns.violinplot(data=df, x='duration_min', y='decade', palette='coolwarm', scale='width', orient='h')
plt.title(f"{student_name} ({student_id})\nRidge Style Plot by Decade")
plt.show()

avg_pop = df.groupby('decade')['popularity'].mean()
avg_pop.plot(kind='line', color=color1, marker='o')
plt.title(f"{student_name} ({student_id})\nAverage Popularity per Decade")
plt.grid(True)
plt.show()

count_by_year = df['release_year'].value_counts().sort_index()
count_by_year.plot(kind='area', color=color2, alpha=0.7)
plt.title(f"{student_name} ({student_id})\nNumber of Songs Over Time")
plt.show()

g = sns.FacetGrid(df, col='decade', col_wrap=4, height=3)
g.map_dataframe(sns.scatterplot, x='duration_min', y='popularity', color=color1)
g.fig.suptitle(f"{student_name} ({student_id})\nPopularity vs Duration per Decade", y=1.02)
plt.show()

longest = df.nlargest(10, 'duration_min')
plt.stem(longest['track_name'], longest['duration_min'], linefmt='#888888', markerfmt='o', basefmt=" ")
plt.xticks(rotation=90)
plt.title(f"{student_name} ({student_id})\nTop 10 Longest Songs")
plt.tight_layout()
plt.show()

avg_duration = df.groupby('artist')['duration_min'].mean().nlargest(5)
plt.plot(avg_duration.values, avg_duration.index, 'o', color=color2)
plt.title(f"{student_name} ({student_id})\nAverage Duration: Top 5 Artists")
plt.grid(True)
plt.show()

crosstab = pd.crosstab(df['decade'], df['artist'])
top3 = df['artist'].value_counts().head(3).index
crosstab[top3].plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title(f"{student_name} ({student_id})\nTop 3 Artists by Decade")
plt.show()

top_tracks = df.nlargest(10, 'popularity')
plt.barh(top_tracks['track_name'], top_tracks['popularity'], color=color1)
plt.title(f"{student_name} ({student_id})\nTop 10 Tracks by Popularity")
plt.gca().invert_yaxis()
plt.show()

top_artists = df['artist'].value_counts().head(3).index
sns.stripplot(data=df[df['artist'].isin(top_artists)], x='artist', y='duration_min', palette=['#8C1515', '#888888'])
plt.title(f"{student_name} ({student_id})\nDuration of Top Artists")
plt.show()

top_albums = df['album'].value_counts().head(5)
plt.pie(top_albums, labels=top_albums.index, autopct='%1.1f%%', colors=['#8C1515', color1, '#888888', color2, '#666666'])
plt.title(f"{student_name} ({student_id})\nTop 5 Albums Distribution")
plt.show()

df_numeric = df[['popularity', 'duration_min']].dropna()
sns.clustermap(df_numeric.corr(), annot=True, cmap='viridis', linewidths=.75, figsize=(6, 6))
plt.show()

sns.pairplot(df[['duration_min', 'popularity', 'release_year']], diag_kind='kde')
plt.show()

df['release_year'].value_counts().sort_index().plot(kind='bar', color=color2)
plt.title(f"{student_name} ({student_id})\nSongs Released per Year")
plt.show()

sns.swarmplot(data=df.head(50), x='artist', y='popularity', palette='coolwarm')
plt.xticks(rotation=45)
plt.title(f"{student_name} ({student_id})\nSwarm Plot (Sample 50)")
plt.show()

plt.hexbin(df['duration_min'], df['popularity'], gridsize=20, cmap='coolwarm', alpha=0.7)
plt.title(f"{student_name} ({student_id})\nHexbin Plot")
plt.show()

sns.ecdfplot(data=df, x='duration_min', color=color1)
plt.title(f"{student_name} ({student_id})\nECDF of Song Duration")
plt.show()

avg_artist_decade = df.groupby(['decade', 'artist'])['popularity'].mean().unstack().fillna(0)
avg_artist_decade[top3].plot(kind='bar', figsize=(8, 4))
plt.title(f"{student_name} ({student_id})\nAverage Popularity by Decade")
plt.show()