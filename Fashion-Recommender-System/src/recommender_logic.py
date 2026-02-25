import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. DICTIONARY: Ensuring cross-demographic harmony
COLOR_HARMONY = {
    'Pastel Pink': ['Pure White', 'Sky Blue', 'Mint Green'],
    'Mint Green': ['Lavender', 'Pure White', 'Peach'],
    'Sky Blue': ['Pure White', 'Pastel Pink', 'Slate Grey'],
    'Classic Black': ['Pure White', 'Slate Grey', 'Navy Blue', 'Dark Charcoal', 'Midnight Blue'],
    'Navy Blue': ['Classic Black', 'Pure White', 'Slate Grey'],
    'Pure White': ['Any'],
    'Dark Charcoal': ['Midnight Blue', 'Forest Green', 'Classic Black'],
    'Deep Maroon': ['Dark Charcoal', 'Forest Green', 'Classic Black'],
    'Forest Green': ['Deep Maroon', 'Midnight Blue', 'Chocolate Brown'],
    'Midnight Blue': ['Forest Green', 'Dark Charcoal', 'Classic Black'],
    'Chocolate Brown': ['Forest Green', 'Pure White']
}

class HybridFashionRecommender:
    def __init__(self, items_df, interactions_df):
        self.items_df = items_df
        self.interactions_df = interactions_df
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.items_df['metadata'])

    def get_recommendations(self, user_id, temp, occasion, age_group, gender, user_pref_color=None, top_n=5):
        # --- Knowledge-Based Filter (Hard Constraints) ---
        mask = (
            (self.items_df['occasion'].str.lower() == occasion.lower()) &
            (self.items_df['min_temp'] <= temp) &
            (self.items_df['max_temp'] >= temp)
        )
        
        # STRICT GENDER RULES
        if gender == 'Male':
            # 1. Filter out specific clothing items using Regex
            forbidden_clothes = 'skirt|bra|dress|tank top|blouse|leggings|bikini'
            mask = mask & (~self.items_df['name'].str.lower().str.contains(forbidden_clothes, regex=True, na=False))
            
            # 2. Filter out specific colors
            forbidden_colors = ['Pastel Pink', 'Lavender']
            mask = mask & (~self.items_df['color'].isin(forbidden_colors))

        filtered_items = self.items_df[mask].copy()

        # FALLBACK: If no items meet criteria
        if filtered_items.empty:
            fallback = self.items_df.sample(min(top_n, len(self.items_df))).copy()
            fallback['kb_score'] = 1.0
            fallback['reason'] = "General recommendation (Strict constraints limited exact matches)."
            return fallback

        kb_scores = []
        reasons = []
        np.random.seed(user_id) 

        for idx, row in filtered_items.iterrows():
            score = 1.0
            reason_parts = [f"Fits {row['occasion']} for {row['season']}"]
            
            # Age-Group Logic
            if row['age_group'] == age_group:
                score += 3.0  
                reason_parts.append(f"designed for {age_group}s")
            else:
                score -= 1.0  
            
            # Color Harmony 
            if user_pref_color:
                if row['color'] == user_pref_color:
                    score += 1.5  
                    reason_parts.append(f"exact {row['color']} match")
                elif row['color'] in COLOR_HARMONY.get(user_pref_color, []):
                    serendipity_factor = np.random.uniform(0.0, 0.2)
                    score += (1.2 + serendipity_factor)
                    reason_parts.append(f"✨ Stylist's pick: {row['color']} complements {user_pref_color}")
                    
            kb_scores.append(score)
            reasons.append(" | ".join(reason_parts))
            
        filtered_items['kb_score'] = kb_scores
        filtered_items['reason'] = reasons

        # --- STEP B: Content-Based Ranking (TF-IDF) ---
        user_favs = self.interactions_df[
            (self.interactions_df['user_id'] == user_id) & (self.interactions_df['rating'] >= 4)
        ]

        if user_favs.empty:
            return filtered_items.sort_values(by='kb_score', ascending=False).head(top_n)

        candidate_indices = filtered_items.index.tolist()
        fav_item_ids = user_favs['item_id'].values
        fav_indices = self.items_df[self.items_df['item_id'].isin(fav_item_ids)].index.tolist()
        
        if not fav_indices:
            return filtered_items.sort_values(by='kb_score', ascending=False).head(top_n)

        # Compute Similarity
        sim_scores = cosine_similarity(self.tfidf_matrix[candidate_indices], self.tfidf_matrix[fav_indices])
        filtered_items['content_score'] = sim_scores.mean(axis=1)
        
        # We divide kb_score by 5.5 (its approximate max) to scale it down between 0 and 1
        normalized_kb = filtered_items['kb_score'] / 5.5
        normalized_content = filtered_items['content_score']
        
        # Blend them smoothly, then scale back up to a 1-5 rating for the UI and Evaluator
        filtered_items['final_score'] = ((normalized_kb * 0.7) + (normalized_content * 0.3)) * 5.0
        
        return filtered_items.sort_values(by='final_score', ascending=False).head(top_n)