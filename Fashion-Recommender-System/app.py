import datetime
import streamlit as st
import pandas as pd
from src.data_preprocessing import prepare_data
from src.recommender_logic import HybridFashionRecommender
from src.evaluation import get_comprehensive_metrics

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fashion Outfit Recommender", page_icon="👗", layout="centered")

# --- DATA LOADING ---
@st.cache_resource
def load_system():
    try:
        train, test, items = prepare_data('data/interactions.csv', 'data/items.csv')
        users = pd.read_csv('data/users.csv')
        recommender = HybridFashionRecommender(items, train)
        return recommender, items, users, test 
    except FileNotFoundError:
        return None, None, None, None

engine, items_df, users_df, test_df = load_system()

# Streamlit wrapper to cache the imported evaluation function at the global level
@st.cache_data(show_spinner=False)
def load_metrics_cached(_engine, test_data, users_data, items_data):
    return get_comprehensive_metrics(_engine, test_data, users_data, items_data)

# --- MAIN INTERFACE ---
st.title("👗 Fashion Outfit Recommender System")
st.markdown("### Hybrid: Content-Aware + Knowledge-Based")
st.markdown("#### Personalized Outfit Suggestions based on Season, Age, Gender, and Style")
st.divider()

if engine is None:
    st.error("❌ Dataset not found! Please run 'python data_generator.py' first.")
    st.stop()

# --- STEP 1: USER ATTRIBUTES ---
st.subheader("1. About You")

is_cold_start = st.toggle("🧪 Simulate New User (Demonstrate Cold-Start Handling)")

col1, col2, col3 = st.columns(3)

with col1:
    if is_cold_start:
        st.info("👤 Guest User")
        user_id = 999 
    else:
        user_id = st.number_input("User ID (1-60)", min_value=1, max_value=60, value=1)

# AUTOMATIC LOOKUP LOGIC
age_options = ["Teen", "Adult", "Senior"]
gender_options = ["Male", "Female"]

if not is_cold_start:
    user_record = users_df[users_df['user_id'] == user_id].iloc[0]
    
    actual_age_group = user_record['age_group']
    default_age_index = age_options.index(actual_age_group) if actual_age_group in age_options else 1
    
    # Check for gender, default to Female if patch wasn't run
    actual_gender = user_record['gender'] if 'gender' in user_record else 'Female'
    default_gender_index = gender_options.index(actual_gender) if actual_gender in gender_options else 1
else:
    default_age_index = 0
    default_gender_index = 0

with col2:
    if is_cold_start:
        age_grp = st.selectbox("Select Age Group", age_options, index=default_age_index)
    else:
        age_grp = st.selectbox("Age Group (Auto-filled)", age_options, index=default_age_index)

with col3:
    if is_cold_start:
        user_gender = st.selectbox("Select Gender", gender_options, index=default_gender_index)
    else:
        user_gender = st.selectbox("Gender (Auto-filled)", gender_options, index=default_gender_index)
        
# --- STYLE & CONTEXTUAL INPUTS ---
st.subheader("2. Your Style & Context")
c1, c2, c3 = st.columns(3)

with c1:
    pref_color = st.selectbox("Preferred Color", 
        ["Pastel Pink", "Mint Green", "Sky Blue", "Classic Black", "Navy Blue", "Dark Charcoal", "Forest Green"])
with c2:
    occ = st.selectbox("Occasion", ["Casual", "Formal", "Sports"])
with c3:
    # TIME-AWARE LOGIC: Auto-detect the current real-world season
    current_month = datetime.datetime.now().month
    if current_month in [12, 1, 2]:
        auto_season = "Winter"
    elif current_month in [6, 7, 8]:
        auto_season = "Summer"
    else:
        auto_season = "Spring/Autumn"
        
    season_options = ["Winter", "Summer", "Spring/Autumn"]
    default_season_index = season_options.index(auto_season)
    
    season = st.selectbox("Season (Auto-detected)", season_options, index=default_season_index)

# --- EXECUTION ---
if st.button("✨ Generate My Outfits", use_container_width=True):
    season_temp_map = {"Winter": -5, "Summer": 30, "Spring/Autumn": 18}
    current_temp = season_temp_map[season]

    # 🛠️ FIXED: Now correctly passing the gender argument!
    recommendations = engine.get_recommendations(
        user_id=user_id, 
        temp=current_temp, 
        occasion=occ, 
        age_group=age_grp,
        gender=user_gender,
        user_pref_color=pref_color,
        top_n=5
    )
    
    st.success(f"Top 5 personalized picks for User {user_id}:")
    
    for i, row in recommendations.iterrows():
        with st.container(border=True):
            col_a, col_b = st.columns([1, 5])
            with col_a:
                if row['color'] == pref_color and row['age_group'] == age_grp:
                    st.markdown("🎯 **Match**")
                else:
                    st.markdown("✨ **Style**")
            with col_b:
                st.markdown(f"**{row['name']}**")
                # 🛠️ FIXED: Added Target Gender to the caption for the demonstration
                st.caption(f"**Reason:** {row['reason']} (Target: {row['age_group']} | {row['target_gender']})")

# --- PERFORMANCE EVALUATION ---
st.divider()

with st.expander("📊 View Comprehensive Performance Metrics"):
    with st.spinner("Calculating Ranking and Prediction metrics against test split..."):
        metrics_df = load_metrics_cached(engine, test_df, users_df, items_df)
    
    st.table(metrics_df)
    st.info("💡 **Software Architecture Note:** This UI dynamically imports evaluation metrics from the `src/evaluation.py` module, demonstrating separation of concerns. It measures Ranking (Precision/Recall) and Rating Prediction (RMSE/MAE) against the 20% ground-truth test split.")