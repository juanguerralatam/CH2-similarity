# Content Similarity and Engagement Analysis on Social Media Video Platforms

## Abstract

This study explores how firms can differentiate their content on social media video platforms like YouTube to capture audience attention in crowded competitive spaces. Using machine learning to analyze multimodal data from four major technology channels (Google Cloud Tech, AWS Developers, Microsoft Azure, and IBM Technology), we investigate the relationship between content similarity and engagement performance. Our findings reveal that a balance of distinctiveness and professional credibility in content boosts engagement. Contrary to expectations, similar narratives can enhance engagement when aligned with timely and relevant topics. Visual and audio effects don't significantly influence engagement, but strategic content variation helps firms avoid commoditization. The research provides insights for managers on leveraging tools like AI to maintain distinctiveness while avoiding similarity-driven commoditization, contributing to digital marketing and innovation theories.

## Research Methodology

### Data Collection and Processing
- **Dataset**: YouTube video metadata from major cloud technology channels (2019-2025)
- **Channels Analyzed**: Google Cloud Tech, AWS Developers, Microsoft Azure, IBM Technology
- **Sample Size**: 20,000+ videos with comprehensive metadata
- **Time Period**: November 2019 - July 2025

### Analytical Framework
1. **Content Similarity Analysis**: Using sentence transformers (all-MiniLM-L6-v2) to compute semantic similarity between video titles
2. **Performance Classification**: Videos categorized into performance quartiles based on view counts
3. **Panel Data Econometric Analysis**: Fixed effects regression modeling engagement outcomes
4. **Temporal Analysis**: Monthly aggregation to capture time-varying effects

### Key Variables
- **Dependent Variable**: Total engagement (views + likes)
- **Independent Variables**: 
  - Content similarity measures (Top vs Top, Low vs Low, Cross-quartile)
  - Number of videos per channel-month
- **Controls**: Channel fixed effects, time fixed effects

## Repository Structure

```
├── dataAnalisys.py          # Exploratory data analysis and visualization
├── similarity.py            # Content similarity computation using ML embeddings
├── econometric.py           # Panel regression analysis with fixed effects
├── plotData.ipynb          # Data visualization and trend analysis
├── try.ipynb               # Experimental analysis notebook
├── videos_statistics.csv   # Raw YouTube video metadata
└── output_csv/
    ├── channel_statistics.csv              # Channel-level summary statistics
    ├── all_channels_quartile_videos.csv    # Performance-classified video data
    ├── monthly_channel_pair_similarity.csv # Similarity measures by channel pairs
    ├── panel_data_monthly.csv              # Panel dataset for econometric analysis
    └── median_views_per_quartile.csv       # Performance quartile statistics
```

## Key Findings

### Content Similarity Patterns
- **Intra-quartile Similarity**: High-performing content shows moderate similarity within channels
- **Cross-quartile Effects**: Strategic differentiation from low-performing content enhances engagement
- **Temporal Dynamics**: Content similarity effects vary significantly over time

### Engagement Drivers
- **Professional Credibility**: Technical accuracy and expertise signals boost engagement
- **Strategic Variation**: Optimal content distinctiveness prevents commoditization
- **Topic Relevance**: Alignment with trending technologies enhances similar content performance

### Managerial Implications
- Content managers should balance distinctiveness with professional credibility
- AI tools can effectively identify optimal similarity levels for content strategy
- Temporal monitoring of similarity patterns prevents engagement decay

## Technical Implementation

### Machine Learning Pipeline
```python
# Content similarity computation
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(video_titles)
similarity_matrix = cosine_similarity(embeddings)
```

### Econometric Modeling
```python
# Panel regression with fixed effects
from linearmodels.panel import PanelOLS
model = PanelOLS(engagement, similarity_features, 
                entity_effects=True, time_effects=True)
results = model.fit()
```

## Reproducibility

### Requirements
- Python 3.8+
- pandas, numpy, scikit-learn
- sentence-transformers
- linearmodels, statsmodels
- matplotlib, seaborn

### Usage
```bash
# 1. Data preprocessing and quartile classification
python dataAnalisys.py

# 2. Compute content similarity measures
python similarity.py

# 3. Run econometric analysis
python econometric.py

# 4. Generate visualizations
jupyter notebook plotData.ipynb
```

## Academic Contribution

This research contributes to:
- **Digital Marketing Theory**: Quantifying the role of content similarity in social media engagement
- **Innovation Management**: Understanding how firms maintain distinctiveness in crowded digital spaces
- **Information Systems**: Demonstrating AI applications in content strategy optimization
- **Strategic Management**: Providing empirical evidence for differentiation vs. similarity trade-offs

## Citation

[To be updated upon publication]

## Contact

[Research team contact information]

---

*Keywords: Content similarity, social media marketing, machine learning, panel data analysis, engagement optimization, digital marketing strategy*
