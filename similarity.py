from itertools import combinations
def compute_channel_pair_similarity(df):
    """
    For each month, compute average cosine similarity between all channel pairs for:
    - Top vs Top
    - Low vs Low
    - Low vs Top
    Output: DataFrame with columns: month, channel_1, channel_2, Top_vs_Top, Low_vs_Low, Low_vs_Top
    """
    df['month'] = df['publishedAt'].dt.to_period('M').astype(str)
    results = []
    for month, group in df.groupby('month'):
        top = group[group['performance'] == 'Top Quartile']
        low = group[group['performance'] == 'Bottom Quartile']
        channels = sorted(set(top['channelTitle'].unique()).union(set(low['channelTitle'].unique())))
        for ch1, ch2 in combinations(channels, 2):
            # Top vs Top
            vids1_top = top[top['channelTitle'] == ch1]
            vids2_top = top[top['channelTitle'] == ch2]
            if len(vids1_top) > 0 and len(vids2_top) > 0:
                emb1_top = np.vstack(vids1_top['title_embedding'].values)
                emb2_top = np.vstack(vids2_top['title_embedding'].values)
                sim_top = cosine_similarity(emb1_top, emb2_top)
                avg_top = np.mean(sim_top)
            else:
                avg_top = np.nan
            # Low vs Low
            vids1_low = low[low['channelTitle'] == ch1]
            vids2_low = low[low['channelTitle'] == ch2]
            if len(vids1_low) > 0 and len(vids2_low) > 0:
                emb1_low = np.vstack(vids1_low['title_embedding'].values)
                emb2_low = np.vstack(vids2_low['title_embedding'].values)
                sim_low = cosine_similarity(emb1_low, emb2_low)
                avg_low = np.mean(sim_low)
            else:
                avg_low = np.nan
            # Low vs Top
            if len(vids1_low) > 0 and len(vids2_top) > 0:
                emb1_low = np.vstack(vids1_low['title_embedding'].values)
                emb2_top = np.vstack(vids2_top['title_embedding'].values)
                sim_lowtop = cosine_similarity(emb1_low, emb2_top)
                avg_lowtop = np.mean(sim_lowtop)
            else:
                avg_lowtop = np.nan
            if len(vids2_low) > 0 and len(vids1_top) > 0:
                emb2_low = np.vstack(vids2_low['title_embedding'].values)
                emb1_top = np.vstack(vids1_top['title_embedding'].values)
                sim_lowtop2 = cosine_similarity(emb2_low, emb1_top)
                avg_lowtop2 = np.mean(sim_lowtop2)
            else:
                avg_lowtop2 = np.nan
            # Average both directions for Low vs Top
            if not np.isnan(avg_lowtop) and not np.isnan(avg_lowtop2):
                avg_lowtop_final = (avg_lowtop + avg_lowtop2) / 2
            elif not np.isnan(avg_lowtop):
                avg_lowtop_final = avg_lowtop
            elif not np.isnan(avg_lowtop2):
                avg_lowtop_final = avg_lowtop2
            else:
                avg_lowtop_final = np.nan
            results.append({
                'month': month,
                'channel_1': ch1,
                'channel_2': ch2,
                'Top_vs_Top': avg_top,
                'Low_vs_Low': avg_low,
                'Low_vs_Top': avg_lowtop_final
            })
    return pd.DataFrame(results)


# --- Relevant Data Preprocessing and Similarity Analysis ---
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import csv
from datetime import datetime

def preprocess_data(csv_path):
    """Load and clean video data from CSV."""
    df = pd.read_csv(csv_path)
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df = df.dropna(subset=['publishedAt', 'viewCount', 'channelTitle', 'title', 'videoId', 'channelId', 'performance'])
    df = df.sort_values('publishedAt')
    return df

def compute_embeddings(df, text_column='title', model_name='all-MiniLM-L6-v2'):
    """Compute sentence embeddings for a DataFrame column."""
    model = SentenceTransformer(model_name)
    df['title_embedding'] = model.encode(df[text_column].tolist()).tolist()
    return df

def save_embeddings_to_csv(df, filename='videoid_title_embeddings.csv'):
    """Save videoId, channelId, and title_embedding to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['videoId', 'channelId', 'viewCount', 'title_embedding'])
        for idx, row in df.iterrows():
            writer.writerow([row['videoId'], row['channelId'], row['viewCount'], row['title_embedding']])

def compute_monthly_quartile_similarity(df):
    """
    For each month, compute average similarity for:
    - Top vs Top
    - Low vs Low
    - Low vs Top
    Output: DataFrame with columns: month, Top_vs_Top, Low_vs_Low, Low_vs_Top
    """
    df['month'] = df['publishedAt'].dt.to_period('M').astype(str)
    results = []
    for month, group in df.groupby('month'):
        top = group[group['performance'] == 'Top Quartile'].reset_index(drop=True)
        low = group[group['performance'] == 'Bottom Quartile'].reset_index(drop=True)
        # Top vs Top
        if len(top) > 1:
            top_emb = np.vstack(top['title_embedding'].values)
            sim_top = cosine_similarity(top_emb)
            mask = ~np.eye(len(top), dtype=bool)
            top_vs_top = np.mean(sim_top[mask])
        else:
            top_vs_top = np.nan
        # Low vs Low
        if len(low) > 1:
            low_emb = np.vstack(low['title_embedding'].values)
            sim_low = cosine_similarity(low_emb)
            mask = ~np.eye(len(low), dtype=bool)
            low_vs_low = np.mean(sim_low[mask])
        else:
            low_vs_low = np.nan
        # Low vs Top
        if len(low) > 0 and len(top) > 0:
            sim_cross = cosine_similarity(np.vstack(low['title_embedding'].values), np.vstack(top['title_embedding'].values))
            low_vs_top = np.mean(sim_cross)
        else:
            low_vs_top = np.nan
        # Store month, scores, and counts
        results.append({
            'month': month,
            'Top_vs_Top': top_vs_top,
            'Low_vs_Low': low_vs_low,
            'Low_vs_Top': low_vs_top,
            'Top_Count': len(top),
            'Low_Count': len(low),
            'Top_ViewSum': top['viewCount'].sum(),
            'Low_ViewSum': low['viewCount'].sum()
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load and preprocess quartile data
    df = preprocess_data('output_csv/all_channels_quartile_videos.csv')
    # Compute embeddings
    df = compute_embeddings(df)
    # Save embeddings (optional)
    save_embeddings_to_csv(df, filename='output_csv/videoid_title_embeddings.csv')
    # Compute monthly quartile similarity (overall)
    monthly_similarity = compute_monthly_quartile_similarity(df)
    monthly_similarity.to_csv('output_csv/monthly_quartile_similarity.csv', index=False)
    # Compute channel-pair similarity for top quartile
    channel_pair_similarity = compute_channel_pair_similarity(df)
    print(channel_pair_similarity.head())
    channel_pair_similarity.to_csv('output_csv/monthly_channel_pair_similarity.csv', index=False)




