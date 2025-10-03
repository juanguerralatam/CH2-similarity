import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as mticker

# Filter data from 2019-11-30 onwards

from matplotlib.dates import DateFormatter
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
import os

# Load and preprocess data
csv_path = 'videos_statistics.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")
df = pd.read_csv(csv_path)
df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
df = df[df['publishedAt'] >= '2019-11-30']
df = df.dropna(subset=['videoId','title','publishedAt', 'channelId','thumbnails','duration','viewCount', 'likeCount'])
df = df.sort_values('publishedAt')

# Select only 'Google Cloud Tech', 'AWS Developers', 'Microsoft Azure', and 'IBM Technology' channels
top_channels = ['Google Cloud Tech', 'AWS Developers', 'Microsoft Azure', 'IBM Technology']
df_top = df[df['channelTitle'].isin(top_channels)]

# Remove outliers based on z-score of log(viewCount) within each channel
df_top['logViewCount'] = np.log1p(df_top['viewCount'])
def remove_outliers(group):
    z = zscore(group['logViewCount'])
    return group[(np.abs(z) < 2)]
df_no_outliers = df_top.groupby('channelTitle', group_keys=False).apply(remove_outliers)

# Show statistics for each channel

# Pretty print channel statistics and export to CSV
channel_stats = df_no_outliers.groupby('channelTitle')['viewCount'].agg(['mean', 'std', 'min', 'max', 'count'])
print("\nChannel Statistics (No Outliers):")
print(channel_stats.round(2).to_string())
channel_stats.to_csv('output_csv/channel_statistics.csv')
print('Exported channel statistics to output_csv/channel_statistics.csv')

# Plot independent graphs for each channel (log scale, no outliers)
line_styles = ['-', '--', '-.', ':']
for idx, channel in enumerate(top_channels):
    channel_data = df_no_outliers[df_no_outliers['channelTitle'] == channel]
    plt.figure(figsize=(12, 6))
    sns.set(style='whitegrid')
    sns.lineplot(x=channel_data['publishedAt'], y=channel_data['logViewCount'],
                 linestyle=line_styles[idx % len(line_styles)], alpha=0.8,
                 color=sns.color_palette('tab10')[idx % 10])
    plt.xlabel('Published Date')
    plt.ylabel('Log(View Count + 1)')
    plt.title(f'Log Video Views Over Time (No Outliers) - {channel}')
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'picture/log_views_{channel.replace(" ", "_")}.png')
    plt.close()
print('Exported individual channel plots to PNG files.')

# Plot all channels together with Gaussian smoothing
plt.figure(figsize=(14, 7))
sns.set(style='whitegrid')
for idx, channel in enumerate(top_channels):
    channel_data = df_no_outliers[df_no_outliers['channelTitle'] == channel]
    x = channel_data['publishedAt'].sort_values()
    y = channel_data.loc[x.index, 'logViewCount']
    y_smooth = gaussian_filter1d(y, sigma=2)
    plt.plot(x, y_smooth, label=channel, linestyle=line_styles[idx % len(line_styles)], alpha=0.9,
             color=sns.color_palette('tab10')[idx % 10])
plt.xlabel('Published Date')
plt.ylabel('Smoothed Log(View Count + 1)')
plt.title('Smoothed Log Video Views Over Time by Channel (No Outliers)')
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.legend(title='Channel')
plt.tight_layout()
plt.savefig('picture/smoothed_log_views_all_channels.png')
plt.close()
print('Exported smoothed log views plot to PNG file.')


# Calculate top and bottom quartiles for each channel in df_no_outliers, generate statistics, plot, and export to CSV with quartile identification
quartile_df_list = []
for channel in top_channels:
    data = df_no_outliers[df_no_outliers['channelTitle'] == channel].copy()
    q1 = data['viewCount'].quantile(0.25)
    q3 = data['viewCount'].quantile(0.75)
    top_q = data[data['viewCount'] >= q3].copy()
    top_q['performance'] = 'Top Quartile'
    top_q['logViewCount'] = np.log1p(top_q['viewCount'])
    bottom_q = data[data['viewCount'] <= q1].copy()
    bottom_q['performance'] = 'Bottom Quartile'
    bottom_q['logViewCount'] = np.log1p(bottom_q['viewCount'])
    quartile_df_list.append(top_q)
    quartile_df_list.append(bottom_q)

# Combine all quartile data
quartile_df = pd.concat(quartile_df_list, ignore_index=True)

# Median view count per quartile for each channel (table)


# Add video count, average duration, and total duration per quartile
def duration_to_seconds(duration_str):
    # Supports HH:MM:SS or MM:SS or SS format
    if not isinstance(duration_str, str) or duration_str.strip() == '':
        return 0
    parts = duration_str.strip().split(':')
    try:
        if len(parts) == 3:
            h, m, s = [int(p) for p in parts]
        elif len(parts) == 2:
            h = 0
            m, s = [int(p) for p in parts]
        elif len(parts) == 1:
            h = 0
            m = 0
            s = int(parts[0])
        else:
            return 0
        return h * 3600 + m * 60 + s
    except Exception:
        return 0

quartile_df['duration_seconds'] = quartile_df['duration'].apply(duration_to_seconds)

agg_table = quartile_df.groupby(['channelTitle', 'performance']).agg(
    Median_View_Count=('viewCount', 'median'),
    Video_Count=('videoId', 'count'),
    Avg_Duration_Minutes=('duration_seconds', lambda x: round(np.mean(x)/60, 2)),
    Total_Duration_Hours=('duration_seconds', lambda x: round(np.sum(x)/3600, 2))
).unstack()

print("\nQuartile Statistics Table:")
print(agg_table.round(2).to_string())
agg_table.to_csv('output_csv/median_views_per_quartile.csv')
print('Exported quartile statistics to output_csv/median_views_per_quartile.csv')

# Export quartile data to CSV
quartile_df.to_csv('output_csv/all_channels_quartile_videos.csv', index=False)
print('Exported quartile data to output_csv/all_channels_quartile_videos.csv')

# Plot videos in log scale
plt.figure(figsize=(14, 7))
sns.set(style='whitegrid')
sns.scatterplot(
    data=quartile_df,
    x='publishedAt',
    y='logViewCount',
    hue='channelTitle',
    style='performance',
    alpha=0.7,
    palette='tab10'
    )
plt.xlabel('Published Date')
plt.ylabel('Log(View Count + 1)')
plt.title('Top and Bottom Quartile Video Performance by Channel (Log Scale)')
plt.legend(title='Channel / Performance', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('picture/quartile_video_performance_scatter.png')
plt.close()
print('Exported quartile video performance scatter plot to PNG file.')

