import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Assuming the similarity data is in a file named 'similarity.csv' with columns: month, Top_vs_Top, Low_vs_Low
df = pd.read_csv('output_csv/monthly_quartile_similarity.csv')

# Convert month to datetime for proper time series plotting
df['month'] = pd.to_datetime(df['month'])

# Sort by month to ensure order
df = df.sort_values('month')

plt.figure(figsize=(10, 6))

# Plot the original lines
plt.plot(df['month'], df['Top_vs_Top'], label='Top_vs_Top', marker='o')
plt.plot(df['month'], df['Low_vs_Low'], label='Low_vs_Low', marker='s')

# Add trend lines to visualize slope
# For Top_vs_Top
x_num = np.arange(len(df))
slope_top, intercept_top = np.polyfit(x_num, df['Top_vs_Top'], 1)
trend_top = slope_top * x_num + intercept_top
plt.plot(df['month'], trend_top, label=f'Top_vs_Top Trend (slope={slope_top:.6f})', linestyle='--')

# For Low_vs_Low
slope_low, intercept_low = np.polyfit(x_num, df['Low_vs_Low'], 1)
trend_low = slope_low * x_num + intercept_low
plt.plot(df['month'], trend_low, label=f'Low_vs_Low Trend (slope={slope_low:.6f})', linestyle='--')

plt.xlabel('Month')
plt.ylabel('Similarity')
plt.title('Similarity Plot with Trends')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()