"""
OTT Content Analytics Dashboard
A comprehensive Data Science project for analyzing streaming content performance.
Built with: Python, Pandas, Plotly, Scikit-Learn, Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="OTT Content Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #E50914;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #E50914;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 5px solid #E50914;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING FUNCTION (with caching) - ROBUST VERSION
# =============================================================================
@st.cache_data
def load_data():
    """Loads and cleans the Netflix dataset. Works with different column names."""
    df = pd.read_csv('netflix_titles.csv')

    # Show actual columns for debugging (hidden in production)
    # st.write("Dataset columns found:", list(df.columns))

    # -------------------- COLUMN NAME MAPPING --------------------
    # Different Kaggle versions use different column names.
    # We normalize them so the code works with any version.

    col_map = {
        'release_year': ['release_year', 'release date', 'year', 'date_added'],
        'imdb_score': ['imdb_score', 'imdb rating', 'rating', 'score', 'imdb'],
        'tmdb_score': ['tmdb_score', 'tmdb rating', 'tmdb'],
        'runtime': ['runtime', 'duration', 'length', 'Runtime'],
        'imdb_votes': ['imdb_votes', 'votes', 'imdbVotes', 'imdb votes'],
        'tmdb_popularity': ['tmdb_popularity', 'popularity', 'tmdb_pop'],
        'genres': ['genres', 'genre', 'listed_in', 'category', 'categories'],
        'production_countries': ['production_countries', 'country', 'countries', 'production_countries'],
        'age_certification': ['age_certification', 'rating', 'rated', 'content_rating', 'mpaa_rating', 'certification'],
        'description': ['description', 'overview', 'synopsis', 'summary', 'plot'],
        'title': ['title', 'name', 'Title'],
        'type': ['type', 'Type', 'content_type', 'content type', 'show_type'],
    }

    # Find the actual column names in the dataset
    actual_cols = {}
    available_cols = set(df.columns.str.lower())

    for standard_name, possible_names in col_map.items():
        for name in possible_names:
            if name.lower() in available_cols or name in df.columns:
                # Find the actual case-sensitive column name
                for real_col in df.columns:
                    if real_col.lower() == name.lower():
                        actual_cols[standard_name] = real_col
                        break
                break

    # -------------------- DATA CLEANING --------------------
    # Numeric columns - only process if they exist
    if 'release_year' in actual_cols:
        df['release_year'] = pd.to_numeric(df[actual_cols['release_year']], errors='coerce')
    else:
        df['release_year'] = np.nan

    if 'imdb_score' in actual_cols:
        df['imdb_score'] = pd.to_numeric(df[actual_cols['imdb_score']], errors='coerce')
    else:
        df['imdb_score'] = np.nan

    if 'tmdb_score' in actual_cols:
        df['tmdb_score'] = pd.to_numeric(df[actual_cols['tmdb_score']], errors='coerce')
    else:
        df['tmdb_score'] = np.nan

    if 'runtime' in actual_cols:
        df['runtime'] = pd.to_numeric(df[actual_cols['runtime']], errors='coerce')
    else:
        df['runtime'] = np.nan

    if 'imdb_votes' in actual_cols:
        df['imdb_votes'] = pd.to_numeric(df[actual_cols['imdb_votes']], errors='coerce')
    else:
        df['imdb_votes'] = 100  # Default size for scatter plots

    if 'tmdb_popularity' in actual_cols:
        df['tmdb_popularity'] = pd.to_numeric(df[actual_cols['tmdb_popularity']], errors='coerce')
    else:
        df['tmdb_popularity'] = np.nan

    # Text columns
    if 'genres' in actual_cols:
        df['genres'] = df[actual_cols['genres']].fillna('Unknown').astype(str)
    else:
        df['genres'] = 'Unknown'

    if 'production_countries' in actual_cols:
        df['production_countries'] = df[actual_cols['production_countries']].fillna('Unknown').astype(str)
    else:
        df['production_countries'] = 'Unknown'

    if 'age_certification' in actual_cols:
        df['age_certification'] = df[actual_cols['age_certification']].fillna('Not Rated').astype(str)
    else:
        df['age_certification'] = 'Not Rated'

    if 'description' in actual_cols:
        df['description'] = df[actual_cols['description']].fillna('').astype(str)
    else:
        df['description'] = ''

    if 'title' in actual_cols:
        df['title'] = df[actual_cols['title']].fillna('Unknown').astype(str)
    else:
        df['title'] = 'Unknown'

    if 'type' in actual_cols:
        df['type'] = df[actual_cols['type']].fillna('Movie').astype(str)
    else:
        df['type'] = 'Movie'

    # Clean up string representations of lists (remove brackets/quotes)
    df['genres'] = df['genres'].str.replace(r"[\[\]'\"]", '', regex=True)
    df['production_countries'] = df['production_countries'].str.replace(r"[\[\]'\"]", '', regex=True)

    # Extract primary genre (first one in the list)
    df['primary_genre'] = df['genres'].apply(lambda x: x.split(',')[0].strip() if ',' in str(x) else str(x).strip())

    # Extract primary country
    df['primary_country'] = df['production_countries'].apply(lambda x: x.split(',')[0].strip() if ',' in str(x) else str(x).strip())

    # Create combined text for recommendations
    df['content_text'] = df['title'].astype(str) + ' ' + df['description'].astype(str) + ' ' + df['genres'].astype(str)

    # If release_year is missing, try to extract from date_added or other date columns
    if df['release_year'].isna().all() and 'date_added' in actual_cols:
        df['release_year'] = pd.to_datetime(df[actual_cols['date_added']], errors='coerce').dt.year

    return df

# =============================================================================
# ML: CONTENT RECOMMENDATION ENGINE
# =============================================================================
@st.cache_data
def build_recommendation_matrix(df):
    """TF-IDF + Cosine Similarity for content-based recommendations."""
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content_text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# =============================================================================
# ML: K-MEANS CLUSTERING
# =============================================================================
@st.cache_data
def perform_clustering(df, n_clusters=5):
    """K-Means clustering on content features."""
    features = []
    # Only use features that have actual data
    for f in ['imdb_score', 'runtime', 'release_year', 'tmdb_popularity']:
        if df[f].notna().sum() > 50:  # At least 50 non-null values
            features.append(f)

    if len(features) < 2:
        return None, None

    cluster_df = df[features].dropna().copy()

    if len(cluster_df) < n_clusters:
        return None, None

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    cluster_df['cluster'] = cluster_labels

    return cluster_df, kmeans

# =============================================================================
# RECOMMENDATION FUNCTION
# =============================================================================
def get_recommendations(title, df, cosine_sim, top_n=10):
    """Given a title, returns top N most similar items."""
    indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()
    idx = indices.get(title.lower())

    if idx is None:
        return None

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]

    recommendations = df.iloc[movie_indices][['title', 'type', 'primary_genre', 'imdb_score', 'release_year']].copy()
    recommendations['similarity_score'] = [round(score[1], 3) for score in sim_scores]
    return recommendations

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please ensure 'netflix_titles.csv' is in the same folder as app.py")
        st.info("📥 Download from: https://www.kaggle.com/datasets/maso0dahmed/netflix-movies-and-shows")
        return

    cosine_sim = build_recommendation_matrix(df)
    cluster_df, kmeans_model = perform_clustering(df, n_clusters=5)

    # =========================================================================
    # SIDEBAR FILTERS
    # =========================================================================
    st.sidebar.markdown("## 🎛️ Dashboard Controls")
    st.sidebar.markdown("---")

    content_types = st.sidebar.multiselect("Select Content Type", options=df['type'].unique(), default=df['type'].unique())

    all_genres = sorted(df['primary_genre'].unique())
    selected_genres = st.sidebar.multiselect("Select Genres", options=all_genres, default=all_genres[:5] if len(all_genres) >= 5 else all_genres)

    # Year filter
    valid_years = df['release_year'].dropna()
    if len(valid_years) > 0:
        min_year, max_year = int(valid_years.min()), int(valid_years.max())
        year_range = st.sidebar.slider("Release Year Range", min_value=min_year, max_value=max_year, value=(max(2000, min_year), max_year))
    else:
        year_range = (1900, 2100)

    # IMDB filter
    valid_imdb = df['imdb_score'].dropna()
    if len(valid_imdb) > 0:
        min_imdb = st.sidebar.slider("Minimum IMDB Score", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    else:
        min_imdb = 0.0

    age_certs = st.sidebar.multiselect("Age Certification", options=sorted(df['age_certification'].unique()), 
                                       default=sorted(df['age_certification'].unique()))

    # Apply filters
    filtered_df = df[
        (df['type'].isin(content_types)) &
        (df['primary_genre'].isin(selected_genres)) &
        (df['release_year'] >= year_range[0]) &
        (df['release_year'] <= year_range[1]) &
        (df['imdb_score'] >= min_imdb) &
        (df['age_certification'].isin(age_certs))
    ].copy()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**📊 Filtered Results:** {len(filtered_df)} titles")

    # =========================================================================
    # HEADER
    # =========================================================================
    st.markdown('<p class="main-title">🎬 OTT Content Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Data-Driven Insights into Streaming Content Performance & Audience Preferences</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>What is this project?</b> This dashboard analyzes Netflix's content catalog using <b>Data Science</b> techniques 
    including <b>TF-IDF NLP</b> for recommendations, <b>K-Means Clustering</b> for content segmentation, 
    and <b>interactive visualizations</b> for business intelligence. Built entirely in Python with Streamlit.
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # KPI METRICS
    # =========================================================================
    st.markdown('<p class="section-header">📈 Key Performance Indicators</p>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    colors = [
        "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
        "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
        "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
    ]

    avg_imdb = filtered_df['imdb_score'].mean() if filtered_df['imdb_score'].notna().any() else 0
    avg_runtime = filtered_df['runtime'].mean() if filtered_df['runtime'].notna().any() else 0

    metrics = [
        (len(filtered_df), "Total Titles"),
        (f"{avg_imdb:.1f}", "Avg IMDB Score"),
        (f"{avg_runtime:.0f}", "Avg Runtime (min)"),
        (filtered_df['primary_genre'].nunique(), "Unique Genres"),
        (filtered_df['primary_country'].nunique(), "Countries")
    ]

    for i, (col, (value, label)) in enumerate(zip([col1, col2, col3, col4, col5], metrics)):
        with col:
            st.markdown(f'<div class="metric-container" style="background: {colors[i]};"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # ROW 1: CONTENT DISTRIBUTION
    # =========================================================================
    st.markdown('<p class="section-header">📊 Content Distribution & Trends</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        type_counts = filtered_df['type'].value_counts().reset_index()
        type_counts.columns = ['Content Type', 'Count']
        fig = px.pie(type_counts, values='Count', names='Content Type', title='Content Type Distribution',
                     color_discrete_sequence=['#E50914', '#564d4d'], hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        genre_counts = filtered_df['primary_genre'].value_counts().head(10).reset_index()
        genre_counts.columns = ['Genre', 'Count']
        fig = px.bar(genre_counts, x='Count', y='Genre', orientation='h', title='Top 10 Genres by Volume',
                     color='Count', color_continuous_scale='Reds')
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # ROW 2: TEMPORAL ANALYSIS
    # =========================================================================
    st.markdown('<p class="section-header">📅 Temporal Analysis</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        yearly = filtered_df.groupby('release_year').size().reset_index(name='Count')
        yearly = yearly[yearly['release_year'] >= 1980]
        if len(yearly) > 0:
            fig = px.line(yearly, x='release_year', y='Count', title='Content Production Trend Over Time', markers=True)
            fig.update_traces(line_color='#E50914', marker_size=8)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No release year data available for timeline.")

    with c2:
        yq = filtered_df.groupby('release_year')['imdb_score'].mean().reset_index()
        yq = yq[yq['release_year'] >= 1980]
        if len(yq) > 0 and yq['imdb_score'].notna().any():
            fig = px.area(yq, x='release_year', y='imdb_score', title='Average Content Quality Trend (IMDB)',
                          color_discrete_sequence=['#E50914'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No IMDB score data available for quality trend.")

    # =========================================================================
    # ROW 3: GEOGRAPHIC & QUALITY
    # =========================================================================
    st.markdown('<p class="section-header">🌍 Geographic & Quality Analysis</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        cc = filtered_df['primary_country'].value_counts().head(15).reset_index()
        cc.columns = ['Country', 'Count']
        fig = px.bar(cc, x='Count', y='Country', orientation='h', title='Top 15 Production Countries',
                     color='Count', color_continuous_scale='Blues')
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sd = filtered_df.dropna(subset=['imdb_score', 'runtime']).copy()
        if len(sd) > 0:
            # Plotly cannot handle NaN in the 'size' parameter, so fill missing votes
            sd['imdb_votes'] = sd['imdb_votes'].fillna(sd['imdb_votes'].median()).replace(0, 100)
            fig = px.scatter(sd, x='runtime', y='imdb_score', color='primary_genre', size='imdb_votes',
                             hover_data=['title', 'release_year'], title='Content Quality vs Runtime Analysis', opacity=0.7)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for scatter plot (need both IMDB score and runtime).")

    # =========================================================================
    # ROW 4: ML CLUSTERING
    # =========================================================================
    st.markdown('<p class="section-header">🤖 ML: Content Clustering (K-Means)</p>', unsafe_allow_html=True)

    if cluster_df is not None and len(cluster_df) > 0:
        st.markdown("""
        <div class="info-box">
        <b>K-Means Clustering</b> groups movies into natural segments based on available numeric features 
        (IMDB Score, Runtime, Release Year, Popularity). Each color = a distinct content cluster.
        </div>
        """, unsafe_allow_html=True)

        fig = px.scatter_3d(cluster_df, x='imdb_score', y='runtime', z='release_year', color='cluster',
                            title='3D Content Clustering Visualization', opacity=0.7, color_continuous_scale='Viridis')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Cluster Characteristics:**")
        stats = cluster_df.groupby('cluster').mean().round(2)
        st.dataframe(stats, use_container_width=True)
    else:
        st.info("📌 Not enough numeric data available for clustering analysis. This happens with certain dataset versions.")

    # =========================================================================
    # ROW 5: ML RECOMMENDATIONS
    # =========================================================================
    st.markdown('<p class="section-header">🔮 ML: Content Recommendation Engine</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Content-Based Filtering:</b> Type any title. The system uses <b>TF-IDF (NLP)</b> to analyze 
    description, genre, and title text, then finds the most similar content using <b>Cosine Similarity</b>.
    </div>
    """, unsafe_allow_html=True)

    search = st.text_input("Enter a movie/show title to get recommendations:", "The Crown")

    if search:
        recs = get_recommendations(search, df, cosine_sim, top_n=8)
        if recs is not None:
            st.markdown(f"**Because you liked '{search}', you might also enjoy:**")
            cols = st.columns(4)
            for idx, (_, row) in enumerate(recs.iterrows()):
                with cols[idx % 4]:
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"📺 {row['type']} | 🎭 {row['primary_genre']}")
                    imdb_val = f"{row['imdb_score']:.1f}" if pd.notna(row['imdb_score']) else "N/A"
                    year_val = int(row['release_year']) if pd.notna(row['release_year']) else "N/A"
                    st.markdown(f"⭐ {imdb_val} | 📅 {year_val}")
                    st.markdown(f"🎯 Similarity: {row['similarity_score']}")
                    st.markdown("---")
        else:
            st.error(f"Title '{search}' not found. Try another!")

    # =========================================================================
    # ROW 6: DATA EXPLORER
    # =========================================================================
    st.markdown('<p class="section-header">🔍 Data Explorer</p>', unsafe_allow_html=True)

    with st.expander("Click to view/filter raw dataset"):
        st.dataframe(filtered_df, use_container_width=True)
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Data as CSV", csv, 'filtered_ott_data.csv', 'text/csv')

    # =========================================================================
    # FOOTER
    # =========================================================================
    st.markdown("---")
    st.markdown('<p style="text-align:center;color:#666;">Built with Python, Pandas, Plotly, Scikit-Learn & Streamlit | Data Science OTT Analytics Project</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()