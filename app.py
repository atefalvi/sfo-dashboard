import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from pathlib import Path
from scipy.stats import kruskal, ttest_ind, levene
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# --- 1. Page Configuration & Design Tokens ---
st.set_page_config(
    layout="wide",
    page_title="SFO Customer Survey Dashboard",
    page_icon="✈️"
)

# --- Add the main title ---
st.title("Case Study - SFO Customer Survey 2017")

# Color tokens
BRAND_COLOR = "#009ade"
MUTED_GRAY = "#808080"
CUSTOM_HEATMAP_COLORS = ["#D62828", "#F06A6A", "#F39C12", "#F6D65A", "#B9E36A", "#2ECC71"]
SENTIMENT_COLORS = {"Positive": "#2ECC71", "Neutral": "#F39C12", "Negative": "#D62828"}

# Mappings for human-readable labels
Q7_LABELS = {
    "Q7ART": "Art Exhibits", "Q7FOOD": "Food & Beverage (Amenities)", "Q7STORE": "Retail Stores",
    "Q7SIGN": "Wayfinding/Signage", "Q7WALKWAYS": "Walkways", "Q7SCREENS": "Flight Info Screens",
    "Q7INFODOWN": "Info/Download", "Q7INFOUP": "Info/Upload", "Q7WIFI": "Wi-Fi",
    "Q7ROADS": "Roads & Curbsides", "Q7PARK": "Parking", "Q7AIRTRAIN": "AirTrain (Amenities)",
    "Q7LTPARKING": "Long-Term Parking", "Q7RENTAL": "Rental Car Center (Amenities)"
}
Q9_LABELS = {
    "Q9BOARDING": "Boarding Areas", "Q9AIRTRAIN": "AirTrain (Cleanliness)", "Q9RENTAL": "Rental Car Center (Cleanliness)",
    "Q9FOOD": "Food & Beverage (Cleanliness)", "Q9RESTROOM": "Restrooms"
}
TOPIC_REGEX = {
    "Security": r"(tsa|security|screen|queue|precheck)",
    "Restrooms": r"(restroom|toilet|washroom)",
    "Food & Beverage": r"(food|restaurant|cafe|beverage)",
    "Signage": r"(sign|wayfind|direction)",
    "Staff": r"(staff|employee|agent|service)",
    "Charging": r"(charge|plug|outlet)",
    "Wi-Fi": r"(wifi|wi-fi|internet)",
    "Cleanliness": r"(clean|dirty|messy)",
}

PROFILE_FIELDS = {
    'Q2PURP1_LBL': "Travel Purpose",
    'Q19AGE_LBL': "Age",
    'Q20GENDER_LBL': "Gender",
}

# --- 2. Data Loading & Preprocessing (Modular & Cached) ---
@st.cache_data
def load_and_preprocess_data():
    """Loads, merges, and preprocesses all dataframes based on V1 requirements."""
    try:
        df_survey = pd.read_csv("model_outputs/fact_survey.csv")
        df_ratings = pd.read_csv("model_outputs/fact_ratings.csv")
        df_comments = pd.read_csv("model_outputs/fact_comments.csv")
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}. Please ensure data is in the `model_outputs/` directory.")
        st.stop()

    # --- Data Cleaning and Type Conversion ---
    df_survey['inferred_date'] = pd.to_datetime(df_survey['inferred_date'], errors='coerce')
    df_survey['NETPRO'] = pd.to_numeric(df_survey['NETPRO'], errors='coerce')
    df_survey['HOWLONG_bucket'] = pd.cut(
        df_survey['HOWLONG'],
        bins=[0, 60, 120, 180, np.inf],
        labels=['0-1h', '1-2h', '2-3h', '3h+'],
        right=False
    )
    df_survey.dropna(subset=['inferred_date'], inplace=True)
    
    # Drop invalid ratings (0, 6) as per requirements
    df_ratings = df_ratings[~df_ratings['rating_value'].isin([0, 6])]
    df_ratings['question_group'] = df_ratings['question_code'].str.extract(r"^(Q\d+)")

    # --- Respondent-level Aggregations ---
    # Pivot ratings to wide format for easy merging
    df_ratings_wide = df_ratings.pivot_table(
        index='RESPNUM_FK', columns='question_code', values='rating_value', aggfunc='mean'
    ).reset_index()

    # Merge all dataframes
    df_merged = pd.merge(df_survey, df_ratings_wide, left_on='RESPNUM_PK', right_on='RESPNUM_FK', how='left')
    df_merged = pd.merge(df_merged, df_comments, left_on='RESPNUM_PK', right_on='RESPNUM_FK', how='left')
    df_merged.drop_duplicates(subset='RESPNUM_PK', inplace=True)

    # Calculate Q7ALL and Q9ALL after merge to ensure columns exist
    q7_cols = [col for col in df_merged.columns if col.startswith('Q7') and col != 'Q7ALL']
    q9_cols = [col for col in df_merged.columns if col.startswith('Q9') and col != 'Q9ALL']
    
    # Check if there are Q7/Q9 columns before calculating mean
    if q7_cols:
        df_merged['Q7ALL'] = df_merged[q7_cols].mean(axis=1)
    else:
        df_merged['Q7ALL'] = np.nan
        
    if q9_cols:
        df_merged['Q9ALL'] = df_merged[q9_cols].mean(axis=1)
    else:
        df_merged['Q9ALL'] = np.nan

    # Clean comments data
    df_comments_clean = df_comments.copy()
    df_comments_clean['Comment'] = df_comments_clean['Comment'].astype(str).str.strip().replace('nan', "No comment provided.")
    df_comments_clean.drop_duplicates(subset=['RESPNUM_FK', 'Comment'], inplace=True)
    
    return df_merged, df_comments_clean

df_main, df_comments_raw = load_and_preprocess_data()
df_main = df_main.sort_values('inferred_date')

# --- 3. Helper Functions & Precomputed Caches ---
@st.cache_data
def compute_nps(series):
    """Calculates NPS from a series of ratings (0-10 scale)."""
    promoters = (series >= 9).sum()
    detractors = (series <= 6).sum()
    total = len(series)
    if total == 0:
        return np.nan
    nps = ((promoters - detractors) / total) * 100
    return round(nps, 2)

@st.cache_data
def get_daily_metrics(df):
    """Aggregates data to daily level for trends and KPIs."""
    daily_metrics = df.groupby('inferred_date', as_index=False).agg(
        q7_mean=('Q7ALL', 'mean'),
        q9_mean=('Q9ALL', 'mean'),
        nps_val=('NETPRO', compute_nps),
        responses=('RESPNUM_PK', 'count')
    )
    return daily_metrics

@st.cache_data
def get_comment_sentiment_rates(df_comments, df_survey):
    """Calculates daily sentiment rates."""
    df_merged = pd.merge(df_comments, df_survey[['RESPNUM_PK', 'inferred_date']], left_on='RESPNUM_FK', right_on='RESPNUM_PK', how='left')
    df_daily_sentiment = df_merged.groupby(['inferred_date', 'Sentiment']).size().unstack(fill_value=0)
    df_daily_sentiment = df_daily_sentiment.div(df_daily_sentiment.sum(axis=1), axis=0) * 100
    return df_daily_sentiment

# --- 4. Sidebar Filters ---
st.sidebar.image('images/logo.png', use_container_width=True)
st.sidebar.markdown(f"<h1 style='color:{BRAND_COLOR};'>Dashboard Filters</h1>", unsafe_allow_html=True)
st.sidebar.markdown("Use the selections below to refine the data across all tabs.")

# Date Range Filter
st.sidebar.markdown("### Date Range")
min_date = df_main['inferred_date'].min().date()
max_date = df_main['inferred_date'].max().date()
date_range = st.sidebar.slider(
    "Date Range",
    value=(min_date, max_date),
    format="YYYY-MM-DD",
    key='date_slider'
)
selected_start_date = pd.to_datetime(date_range[0])
selected_end_date = pd.to_datetime(date_range[1])

# Boarding Area Filter
st.sidebar.markdown("### Segments")
all_barea = sorted(df_main['BAREA'].dropna().unique())
selected_barea = st.sidebar.multiselect("Boarding Areas", all_barea, default=all_barea)

# Apply global filters
filtered_df = df_main[
    (df_main['inferred_date'] >= selected_start_date) &
    (df_main['inferred_date'] <= selected_end_date) &
    (df_main['BAREA'].isin(selected_barea))
]

# --- 5. Main Dashboard Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Summary", 
    "📈 Detailed Ratings Analysis", 
    "✈️ Passenger Flow & Operations",
    "💬 Customer Feedback",
    "⚙️ Benchmarks & Cohorts",
    "👤 About Me"
])

# --- Tab 1: Executive Summary ---
with tab1:
    st.header("Executive Summary")
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your selections.")
    else:
        # A. KPI Tiles
        overall_metrics = get_daily_metrics(df_main)
        filtered_metrics = get_daily_metrics(filtered_df)
        
        # Get overall and last 7-day average for comparison
        overall_nps, overall_q7, overall_q9 = overall_metrics[['nps_val', 'q7_mean', 'q9_mean']].mean()
        
        # Calculate last period (7-day) change
        last_7_days_df = filtered_metrics.tail(7)
        prev_7_days_df = filtered_metrics.iloc[-14:-7]
        
        last_7_nps = last_7_days_df['nps_val'].mean()
        prev_7_nps = prev_7_days_df['nps_val'].mean()
        nps_delta_wow = last_7_nps - prev_7_nps if not prev_7_days_df.empty else np.nan

        last_7_q7 = last_7_days_df['q7_mean'].mean()
        prev_7_q7 = prev_7_days_df['q7_mean'].mean()
        q7_delta_wow = last_7_q7 - prev_7_q7 if not prev_7_days_df.empty else np.nan

        last_7_q9 = last_7_days_df['q9_mean'].mean()
        prev_7_q9 = prev_7_days_df['q9_mean'].mean()
        q9_delta_wow = last_7_q9 - prev_7_q9 if not prev_7_days_df.empty else np.nan

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label=f"Net Promoter Score (Δ WoW)", 
                value=f"{filtered_metrics['nps_val'].mean():.0f}%" if pd.notna(filtered_metrics['nps_val'].mean()) else "N/A", 
                delta=f"{nps_delta_wow:+.0f}" if pd.notna(nps_delta_wow) else None,
                help="Delta vs. Overall Airport Average. Green is better, Red is worse."
            )
        with col2:
            st.metric(
                label=f"Amenities (Q7) Score (Δ WoW)", 
                value=f"{filtered_metrics['q7_mean'].mean():.2f}" if pd.notna(filtered_metrics['q7_mean'].mean()) else "N/A", 
                delta=f"{q7_delta_wow:+.2f}" if pd.notna(q7_delta_wow) else None
            )
        with col3:
            st.metric(
                label=f"Cleanliness (Q9) Score (Δ WoW)", 
                value=f"{filtered_metrics['q9_mean'].mean():.2f}" if pd.notna(filtered_metrics['q9_mean'].mean()) else "N/A", 
                delta=f"{q9_delta_wow:+.2f}" if pd.notna(q9_delta_wow) else None
            )
        with col4:
            st.metric(
                label="Total Responses", 
                value=f"{len(filtered_df):,}"
            )
        
        st.markdown("---")
        
        # B. Radar Charts with New Local Filters
        st.subheader("Ratings Comparison: Filtered vs. Overall")
        
        # New local filters for the radar charts
        radar_filter_cols = st.columns(3)
        with radar_filter_cols[0]:
            all_purposes = sorted(filtered_df['Q2PURP1_LBL'].dropna().unique())
            selected_purposes = st.multiselect("Travel Purpose", all_purposes, key='radar_purposes')
        with radar_filter_cols[1]:
            all_genders = sorted(filtered_df['Q20GENDER_LBL'].dropna().unique())
            selected_genders = st.multiselect("Gender", all_genders, key='radar_genders')
        with radar_filter_cols[2]:
            all_ages = sorted(filtered_df['Q19AGE_LBL'].dropna().unique())
            selected_ages = st.multiselect("Age", all_ages, key='radar_ages')
            
        # Apply local radar filters
        filtered_for_radar_df = filtered_df.copy()
        if selected_purposes:
            filtered_for_radar_df = filtered_for_radar_df[filtered_for_radar_df['Q2PURP1_LBL'].isin(selected_purposes)]
        if selected_genders:
            filtered_for_radar_df = filtered_for_radar_df[filtered_for_radar_df['Q20GENDER_LBL'].isin(selected_genders)]
        if selected_ages:
            filtered_for_radar_df = filtered_for_radar_df[filtered_for_radar_df['Q19AGE_LBL'].isin(selected_ages)]
            
        radar_cols = st.columns(2)
        with radar_cols[0]:
            q_list_q7, q_labels_q7 = list(Q7_LABELS.keys()), Q7_LABELS
            profile_means_q7 = filtered_for_radar_df[q_list_q7].mean().values.tolist()
            overall_means_q7 = df_main[q_list_q7].mean().values.tolist()
            fig_q7_radar = go.Figure()
            fig_q7_radar.add_trace(go.Scatterpolar(r=profile_means_q7, theta=list(Q7_LABELS.values()), fill='toself', name='Filtered Selection', line_color=BRAND_COLOR))
            fig_q7_radar.add_trace(go.Scatterpolar(r=overall_means_q7, theta=list(Q7_LABELS.values()), name='Overall Average', line_color=MUTED_GRAY))
            fig_q7_radar.update_layout(title_text="Amenities (Q7) Ratings", polar=dict(radialaxis=dict(visible=True, range=[1, 5])))
            st.plotly_chart(fig_q7_radar, use_container_width=True)

        with radar_cols[1]:
            q_list_q9, q_labels_q9 = list(Q9_LABELS.keys()), Q9_LABELS
            profile_means_q9 = filtered_for_radar_df[q_list_q9].mean().values.tolist()
            overall_means_q9 = df_main[q_list_q9].mean().values.tolist()
            fig_q9_radar = go.Figure()
            fig_q9_radar.add_trace(go.Scatterpolar(r=profile_means_q9, theta=list(Q9_LABELS.values()), fill='toself', name='Filtered Selection', line_color=BRAND_COLOR))
            fig_q9_radar.add_trace(go.Scatterpolar(r=overall_means_q9, theta=list(Q9_LABELS.values()), name='Overall Average', line_color=MUTED_GRAY))
            fig_q9_radar.update_layout(title_text="Cleanliness (Q9) Ratings", polar=dict(radialaxis=dict(visible=True, range=[1, 5])))
            st.plotly_chart(fig_q9_radar, use_container_width=True)

        st.markdown("---")

        # C. Trends - Area Score Over Time
        st.subheader("Rolling Trends by Boarding Area")
        area_trends_df = filtered_df.groupby(['inferred_date', 'BAREA'], as_index=False).agg(
            q7_mean=('Q7ALL', 'mean'),
            q9_mean=('Q9ALL', 'mean')
        ).sort_values('inferred_date')
        
        selected_trend_metric = st.selectbox("Select Metric", ["Amenities (Q7ALL)", "Cleanliness (Q9ALL)"])
        selected_metric_col = 'q7_mean' if "Q7" in selected_trend_metric else 'q9_mean'
        
        fig_area_trends = go.Figure()
        for area in area_trends_df['BAREA'].unique():
            df_area = area_trends_df[area_trends_df['BAREA'] == area].copy()
            df_area['rolling_mean'] = df_area[selected_metric_col].rolling(window=7, min_periods=1).mean()
            fig_area_trends.add_trace(go.Scatter(
                x=df_area['inferred_date'],
                y=df_area['rolling_mean'],
                mode='lines',
                name=f"Area {area}"
            ))
        
        fig_area_trends.add_trace(go.Scatter(
            x=filtered_metrics['inferred_date'],
            y=filtered_metrics[selected_metric_col].rolling(window=7, min_periods=1).mean(),
            mode='lines',
            name="Airport Baseline",
            line=dict(color=MUTED_GRAY, dash='dash')
        ))
        
        fig_area_trends.update_layout(title=f"7-Day Rolling {selected_trend_metric} by Boarding Area", hovermode="x unified")
        st.plotly_chart(fig_area_trends, use_container_width=True)

# --- Tab 2: Detailed Ratings Analysis ---
with tab2:
    st.header("Detailed Ratings Explorer")
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your selections.")
    else:
        # D. Heatmap - Question x Area
        st.subheader("Heatmap: Mean Rating by Question and Boarding Area")
        
        # Melt the filtered data to long format for groupby
        q_cols_to_melt = list(Q7_LABELS.keys()) + list(Q9_LABELS.keys())
        heatmap_df = filtered_df.melt(
            id_vars=['BAREA'], 
            value_vars=q_cols_to_melt, 
            var_name='question_code', 
            value_name='rating_value'
        ).dropna(subset=['rating_value'])
        
        # Aggregate the data for the heatmap
        heatmap_agg = heatmap_df.groupby(['question_code', 'BAREA']).agg(
            mean_rating=('rating_value', 'mean'),
            N=('rating_value', 'count')
        ).reset_index()
        
        if not heatmap_agg.empty:
            heatmap_pivot = heatmap_agg.pivot_table(index='question_code', columns='BAREA', values='mean_rating')
            
            # Z-score for better comparison
            heatmap_pivot_zscore = heatmap_pivot.sub(heatmap_pivot.mean(axis=1), axis=0).div(heatmap_pivot.std(axis=1), axis=0)
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_pivot_zscore.values,
                x=heatmap_pivot_zscore.columns,
                y=heatmap_pivot_zscore.index.map({**Q7_LABELS, **Q9_LABELS}),
                colorscale='RdYlGn',
                zmid=0,
                hovertemplate='<b>Area: %{x}</b><br>Question: %{y}<br>Z-Score: %{z:.2f}<extra></extra>'
            ))
            fig_heatmap.update_layout(title="Z-scored Mean Rating by Question and Boarding Area")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No ratings available for the selected filters to generate a heatmap.")

        st.markdown("---")

        # E. Distribution - Box/Violin by Area
        st.subheader("Distribution of Amenity & Cleanliness Scores")
        dist_metric = st.selectbox("Select Metric", ['Amenities (Q7ALL)', 'Cleanliness (Q9ALL)'], key='dist_metric')
        dist_metric_col = 'Q7ALL' if 'Q7' in dist_metric else 'Q9ALL'

        fig_dist = go.Figure()
        for area in filtered_df['BAREA'].unique():
            fig_dist.add_trace(go.Violin(
                y=filtered_df[filtered_df['BAREA'] == area][dist_metric_col].dropna(),
                name=f"Area {area}",
                box_visible=True,
                meanline_visible=True,
            ))
        fig_dist.update_layout(title=f"Distribution of {dist_metric} by Boarding Area")
        st.plotly_chart(fig_dist, use_container_width=True)

# --- Tab 3: Passenger Flow & Operations ---
with tab3:
    st.header("Passenger Flow & Operations")
    
    # H. Sankey - New flow: peak_lbl -> Q2PURP1_LBL -> BAREA
    st.subheader("Passenger Flow: Peak -> Purpose -> Boarding Area")
    df_sankey = filtered_df.copy()
    
    # Get counts for the Sankey flow
    sankey_counts = df_sankey.groupby(['PEAK_LBL', 'Q2PURP1_LBL', 'BAREA']).size().reset_index(name='count')
    
    # Create lists of unique nodes for each level
    sources = sankey_counts['PEAK_LBL'].unique()
    targets_from_sources = sankey_counts['Q2PURP1_LBL'].unique()
    targets_from_targets = sankey_counts['BAREA'].unique()
    
    all_nodes = list(sources) + list(targets_from_sources) + list(targets_from_targets)
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    # Create links
    links_1 = sankey_counts[['PEAK_LBL', 'Q2PURP1_LBL', 'count']]
    links_1['source'] = links_1['PEAK_LBL'].map(node_map)
    links_1['target'] = links_1['Q2PURP1_LBL'].map(node_map)
    
    links_2 = sankey_counts[['Q2PURP1_LBL', 'BAREA', 'count']]
    links_2['source'] = links_2['Q2PURP1_LBL'].map(node_map)
    links_2['target'] = links_2['BAREA'].map(node_map)
    
    all_links = pd.concat([links_1, links_2], ignore_index=True)
    
    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(label=all_nodes),
        link=dict(
            source=all_links['source'],
            target=all_links['target'],
            value=all_links['count']
        )
    )])
    sankey_fig.update_layout(title_text="Passenger Flow (Peak -> Purpose -> Boarding Area)", height=800)
    st.plotly_chart(sankey_fig, use_container_width=True)


# --- Tab 4: Customer Feedback ---
with tab4:
    st.header("Customer Comments and Feedback")
    
    # Filter comments based on the overall filtered_df
    relevant_respnums = filtered_df['RESPNUM_PK'].unique()
    filtered_comments = df_comments_raw[df_comments_raw['RESPNUM_FK'].isin(relevant_respnums)].copy()
    
    # Keyword Counts at the top
    st.subheader("Comment Keywords and Counts")
    df_topics = filtered_comments.copy()
    topic_counts = {topic: 0 for topic in TOPIC_REGEX.keys()}
    for topic, keywords in TOPIC_REGEX.items():
        count = df_topics['Comment'].str.contains(keywords, case=False, na=False).sum()
        topic_counts[topic] = count
    topic_df = pd.DataFrame(list(topic_counts.items()), columns=['Topic', 'Count']).sort_values('Count', ascending=False)
    fig_topics = px.bar(topic_df, x='Topic', y='Count', title="Comment Topics by Count")
    st.plotly_chart(fig_topics, use_container_width=True, key='topic_chart')
    st.markdown("---")

    # Filters on a single row
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        sentiment_options = sorted(filtered_comments['Sentiment'].dropna().unique())
        selected_sentiment = st.multiselect("Filter by Sentiment", sentiment_options, key='comment_sentiment_filter')
    with filter_col2:
        group_options = sorted(filtered_comments['comment_group'].dropna().unique())
        selected_group = st.multiselect("Filter by Comment Group", group_options, key='comment_group_filter')
    with filter_col3:
        search_term = st.text_input("Search for keywords", help="e.g., 'charging', 'food', 'staff'", key='comment_search_filter')
    
    # Apply local filters
    if selected_sentiment:
        filtered_comments = filtered_comments[filtered_comments['Sentiment'].isin(selected_sentiment)]
    if selected_group:
        filtered_comments = filtered_comments[filtered_comments['comment_group'].isin(selected_group)]
    if search_term:
        filtered_comments = filtered_comments[filtered_comments['Comment'].str.contains(search_term, case=False, na=False)]
    
    st.info(f"Displaying {len(filtered_comments)} Comments")
    st.markdown("---")

    # Main content columns
    comments_col, profile_col = st.columns([1, 1])
    
    with comments_col:
        st.subheader("Filtered Comments List")
        # Make the comment selection area scrollable with a fixed height
        with st.container(height=600, border=True):
            if not filtered_comments.empty:
                comment_data = filtered_comments.to_dict('records')
                comment_options = []
                for i, row in enumerate(comment_data):
                    comment_text = str(row.get('Comment', '')).strip()
                    if comment_text == 'nan' or not comment_text:
                        comment_text = "No comment provided."
                    
                    display_text = f"{row.get('Sentiment', 'N/A')} - {row.get('comment_group', 'N/A')} - {comment_text[:60]}..."
                    comment_options.append(display_text)
                
                selected_comment_index = st.radio(
                    "Select a comment:",
                    list(range(len(comment_options))),
                    format_func=lambda x: comment_options[x],
                    key='comment_selection_radio',
                    help="Select a comment to view the passenger's profile and ratings."
                )
                
                # Use the index to get the corresponding row data
                selected_row = comment_data[selected_comment_index]
                st.session_state.selected_respnum = selected_row['RESPNUM_FK']
            else:
                st.session_state.selected_respnum = None
                st.info("No comments match the selected filters or search term.")

    with profile_col:
        st.subheader("Passenger Profile & Ratings")
        
        if st.session_state.selected_respnum:
            # Get the full profile and ratings for the selected respondent
            selected_profile = df_main[df_main['RESPNUM_PK'] == st.session_state.selected_respnum].iloc[0]
            
            with st.container(height=600, border=True):
                st.markdown(f"**Respondent ID:** `{st.session_state.selected_respnum}`")
                
                # Demographics
                for field, label in PROFILE_FIELDS.items():
                    st.markdown(f"**{label}:** {selected_profile.get(field, 'N/A')}")
                
                st.markdown("---")
                st.markdown("**All Passenger Ratings**")
                
                # Filter for all Q7 and Q9 rating columns
                q7_q9_cols = list(Q7_LABELS.keys()) + list(Q9_LABELS.keys())
                all_ratings_series = selected_profile[q7_q9_cols]

                # Reformat and display ratings
                all_ratings_df = all_ratings_series.reset_index()
                all_ratings_df.columns = ['Question', 'Rating']
                all_ratings_df = all_ratings_df.dropna()
                all_ratings_df['Question_Label'] = all_ratings_df['Question'].map({**Q7_LABELS, **Q9_LABELS})
                
                if not all_ratings_df.empty:
                    ratings_chart = px.bar(
                        all_ratings_df, 
                        x='Question_Label', 
                        y='Rating', 
                        color='Rating',
                        color_continuous_scale=CUSTOM_HEATMAP_COLORS,
                        height=300, 
                        labels={'Rating': 'Score (1-5)', 'Question_Label': 'Question'},
                        title="Individual Rating Breakdown"
                    )
                    ratings_chart.update_layout(xaxis={'categoryorder': 'total descending'}, coloraxis_showscale=False)
                    st.plotly_chart(ratings_chart, use_container_width=True)
                else:
                    st.info("No ratings available for this passenger.")

        else:
            st.info("Select a comment from the list to view the passenger's details.")

# --- Tab 5: Benchmarks & Cohorts ---
with tab5:
    st.header("Benchmarks & Cohorts")
    
    # P. Cohort Compare - HOWLONG buckets
    st.subheader("Scores by Dwell Time Cohort")
    cohort_df = filtered_df.groupby('HOWLONG_bucket').agg(
        nps=('NETPRO', compute_nps),
        q7_all=('Q7ALL', 'mean'),
        q9_all=('Q9ALL', 'mean')
    ).reset_index()
    
    fig_cohorts = go.Figure()
    fig_cohorts.add_trace(go.Bar(x=cohort_df['HOWLONG_bucket'], y=cohort_df['q7_all'], name='Amenities (Q7ALL)', marker_color=BRAND_COLOR))
    fig_cohorts.add_trace(go.Bar(x=cohort_df['HOWLONG_bucket'], y=cohort_df['q9_all'], name='Cleanliness (Q9ALL)', marker_color=MUTED_GRAY))
    
    fig_cohorts.update_layout(title="Average Scores by Dwell Time", barmode='group')
    st.plotly_chart(fig_cohorts, use_container_width=True, key='cohort_chart')

    # Q. Benchmark vs Airport & Peer Areas (simplified)
    st.subheader("Area Benchmarking (vs. Airport Average)")
    benchmark_area = st.selectbox("Select Boarding Area to Benchmark", sorted(filtered_df['BAREA'].dropna().unique()))
    
    if benchmark_area:
        benchmark_df = filtered_df[filtered_df['BAREA'] == benchmark_area]
        other_areas_df = filtered_df[filtered_df['BAREA'] != benchmark_area]
        
        benchmark_means = benchmark_df[['Q7ALL', 'Q9ALL']].mean()
        other_means = other_areas_df[['Q7ALL', 'Q9ALL']].mean()
        
        benchmark_data = pd.DataFrame({
            'Metric': ['Q7ALL', 'Q9ALL'],
            'Benchmark Area': benchmark_means.values,
            'Other Areas Avg': other_means.values
        })
        
        fig_benchmark = px.bar(benchmark_data, x='Metric', y=['Benchmark Area', 'Other Areas Avg'], barmode='group',
                               color_discrete_map={'Benchmark Area': BRAND_COLOR, 'Other Areas Avg': MUTED_GRAY},
                               title=f"Scores for Area {benchmark_area} vs. Other Areas")
        fig_benchmark.update_layout(yaxis_title="Average Score")
        st.plotly_chart(fig_benchmark, use_container_width=True, key='benchmark_chart')

# --- Tab 6: About Me ---
with tab6:
    st.header("👤 About Me")
    
    profile_col, text_col = st.columns([1, 2])
    
    with profile_col:
        # Display the profile picture
        try:
            st.image('images/pp.png', caption="Syed Atef Alvi")
        except FileNotFoundError:
            st.warning("Profile picture not found at `images/pp.png`")

    with text_col:
        st.markdown("### Summary")
        st.markdown("Experienced Data Engineer with 10+ years of expertise in designing scalable pipelines, building robust data models, and delivering actionable insights. Proficient in Python, SQL, Airflow, Snowflake, and Tableau, with a focus on enhancing performance, ensuring data quality, and driving data-driven decisions. Skilled in cross-functional collaboration and creating efficient, well-governed data systems to support business goals and continuous improvement.")

        st.markdown("---")
        st.markdown("### 🛠️ Skills")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("🐍 Python")
            st.markdown("📊 Tableau")
        with col2:
            st.markdown("🐘 PostgreSQL")
            st.markdown("❄️ Snowflake")
        with col3:
            st.markdown("☁️ Salesforce")
            st.markdown("💧 dbt")
        with col4:
            st.markdown("💻 GitHub")
            st.markdown("🐳 Docker")

        st.markdown("---")
        st.markdown("### ✈️ About This Case Study")
        st.markdown("""
    This dashboard is built on the **2017 San Francisco International Airport (SFO) Customer Survey**, which collected feedback from travelers on airport facilities, cleanliness, safety, wayfinding, food & retail, and overall satisfaction.
    The goal of this case study is to transform raw survey data into **actionable insights**—highlighting passenger pain points, identifying quick operational wins, and uncovering long-term opportunities for infrastructure and service improvements.
    By combining data cleaning, statistical analysis, and visualization, this project demonstrates how analytics can guide decisions that improve **passenger flow, customer experience, and airport operations**.
    """)