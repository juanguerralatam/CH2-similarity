"""
Econometric Analysis: Panel Data Regression of YouTube Engagement on Similarity

This script performs a panel data regression with fixed effects to analyze the relationship
between YouTube engagement and similarity measures.

Y_it: YouTube engagement = total likes + views per firm (channel) per month
Controls: number of videos per firm per month
Fixed Effects: firm (channel) fixed effects, month fixed effects

Data Sources:
- Video data: output_csv/all_channels_quartile_videos.csv
- Similarity data: output_csv/monthly_channel_pair_similarity.csv

Output:
- Regression summary
- Panel data CSV: output_csv/panel_data_monthly.csv
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

def load_and_prepare_data():
    """Load video and similarity data, prepare for merging."""
    videos = pd.read_csv('output_csv/all_channels_quartile_videos.csv')
    similarity = pd.read_csv('output_csv/monthly_channel_pair_similarity.csv')

    videos['publishedAt'] = pd.to_datetime(videos['publishedAt'])
    videos['month'] = videos['publishedAt'].dt.to_period('M').dt.to_timestamp()
    similarity['month'] = pd.to_datetime(similarity['month']).dt.to_period('M').dt.to_timestamp()

    return videos, similarity

def aggregate_to_panel(videos_df, similarity_df):
    """Aggregate data to channel-month level."""
    # Aggregate videos to channel-month
    panel = videos_df.groupby(['channelTitle', 'month']).agg(
        total_engagement=('viewCount', 'sum'),  # Y: sum of views
        total_likes=('likeCount', 'sum'),      # sum of likes
        num_videos=('videoId', 'count')
    ).reset_index()
    panel['total_engagement'] = panel['total_engagement'] + panel['total_likes']

    # Compute average Top_vs_Top similarity per month
    monthly_sim = similarity_df.groupby('month')['Top_vs_Top'].mean().reset_index()
    monthly_sim.rename(columns={'Top_vs_Top': 'avg_top_similarity'}, inplace=True)

    # Merge
    panel = pd.merge(panel, monthly_sim, on='month', how='left')
    panel = panel.dropna()

    return panel

def export_panel_data(panel_df):
    """Export panel data to CSV."""
    panel_df.to_csv('output_csv/panel_data_monthly.csv', index=False)
    print("Panel data exported to output_csv/panel_data_monthly.csv")

def run_panel_regression(panel_df):
    """Run panel OLS with fixed effects."""
    # Set multi-index
    panel_df = panel_df.set_index(['channelTitle', 'month'])

    # Dependent variable
    y = panel_df['total_engagement']

    # Independent variables
    X = panel_df[['avg_top_similarity', 'num_videos']]

    # Run PanelOLS with entity and time effects
    model = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    results = model.fit()

    return results

def main():
    """Main function."""
    videos, similarity = load_and_prepare_data()
    panel = aggregate_to_panel(videos, similarity)
    export_panel_data(panel)
    results = run_panel_regression(panel)
    print(results.summary)

if __name__ == "__main__":
    main()