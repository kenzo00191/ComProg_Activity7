import pandas as pd
import os

dataset_path = 'spotify_top_1000_tracks.csv'


df = pd.read_csv(dataset_path, encoding="utf-8")

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['year'] = df['release_date'].dt.year

print("✅ Dataset loaded and basic preprocessing complete!")

print(df.head(3))

import numpy as np

df['track_name'] = df['track_name'].str.strip()
df['artist'] = df['artist'].str.strip()
df['album'] = df['album'].str.strip()

df['year'] = df['year'].fillna(0).astype(int)

cols_to_drop = ['spotify_url', 'id', 'release_date']

for col in ['time_signature', 'key', 'mode']:
    if col in df.columns:
        cols_to_drop.append(col)

df = df.drop(columns=cols_to_drop, errors='ignore')

tempo_bins = [0, 100, 140, np.inf]
tempo_labels = ['Slow', 'Medium', 'Fast']

if 'tempo' in df.columns:
    df['tempo_category'] = pd.cut(
        df['tempo'], bins=tempo_bins, 
        labels=tempo_labels, right=False
    )
    print("Feature 'tempo_category' created.")
else:
    print("Warning: 'tempo' column not found; skipping 'tempo_category' creation.")

df = df.drop_duplicates(subset=['track_name', 'artist'], keep='first')

print(f"✅ Data cleaning and feature engineering complete.")
print(f"Final Row Count after deduplication: {len(df)}")

import matplotlib.pyplot as plt

column_to_plot = 'duration_min' 

plt.figure(figsize=(10, 6))
plt.hist(df[column_to_plot], bins=80, color='green', edgecolor='blue')

plt.title(f'Distribution of {column_to_plot.replace("_", " ").title()}', fontsize=14)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

print(f"Generating histogram for {column_to_plot}...")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

selected_palette = 'plasma' 

sns.boxplot(x='year', y='popularity', data=df, palette=selected_palette)

plt.title(f'Popularity Distribution by Year (Palette: {selected_palette})', fontsize=14)
plt.xlabel('Release Year')
plt.ylabel('Popularity Score')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)

print(f"Generating Boxplot using the '{selected_palette}' color scheme...")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

sns.scatterplot(x='duration_min', y='popularity', data=df, color='pink', alpha=0.6)

sns.regplot(x='duration_min', y='popularity', data=df, scatter=False, color='black')

plt.title('Song Duration vs. Popularity', fontsize=14)
plt.xlabel('Duration (minutes)')
plt.ylabel('Popularity Score (0-100)')
plt.grid(True, linestyle='--', alpha=0.7)

print("Generating scatter plot with regression line...")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

key_features = ['duration_min', 'popularity']

sns.pairplot(
    df[key_features], 
    diag_kind='kde', 
    corner=True, 
    plot_kws={'alpha': 0.6, 'color': "#25A913"}
)

plt.suptitle('Pair Plot of Duration and Popularity', y=1.02, fontsize=16)

print("Generating Pair Plot... (Close all previous windows to see this one)")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

sns.jointplot(
    x='duration_min', 
    y='popularity', 
    data=df, 
    kind='reg',           
    height=8, 
    marginal_kws=dict(bins=25, color='gray', edgecolor='black'), 
    color='darkred',      # Set the color here instead of inside joint_kws
    scatter_kws={'alpha': 0.6} # Use scatter_kws for transparency in reg plots
)

plt.suptitle('Joint Distribution of Duration and Popularity', y=1.02, fontsize=14)

print("✅ Final Joint Plot window opening...")
plt.show()

import os
import webbrowser
from matplotlib.animation import FuncAnimation, PillowWriter

yearly_pop = df.groupby('year')['popularity'].mean().reset_index()
yearly_pop = yearly_pop.sort_values('year')

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(yearly_pop['year'].min(), yearly_pop['year'].max())
ax.set_ylim(0, yearly_pop['popularity'].max() * 1.1)
line, = ax.plot([], [], color='royalblue', linewidth=2.5, label='Average Popularity')

ax.set_title("Evolution of Track Popularity Over Time", fontsize=14, color='navy')
ax.set_xlabel("Year of Release")
ax.set_ylabel("Average Popularity")
ax.legend(loc="upper left")
plt.tight_layout()

def animate(i):
    x = yearly_pop['year'][:i]
    y = yearly_pop['popularity'][:i]
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, animate, frames=len(yearly_pop), interval=60, repeat=False)

gif_path = os.path.abspath("yearly_popularity_trend.gif")
ani.save(gif_path, writer=PillowWriter(fps=10))

print(f"✅ GIF saved successfully at: {gif_path}")

webbrowser.open(f"file://{gif_path}")
plt.close(fig)