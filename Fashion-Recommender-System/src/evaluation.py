import pandas as pd
import numpy as np

def get_comprehensive_metrics(model, test_data, users, items, k=5):
    """Calculates both Ranking (Precision/Recall) and Prediction (RMSE/MAE) metrics."""
    test_relevant = test_data[test_data['rating'] >= 4]
    test_users = test_relevant['user_id'].unique()

    hits_hybrid, hits_random = 0, 0
    precision_hybrid, precision_random = 0.0, 0.0
    recall_hybrid, recall_random = 0.0, 0.0
    
    squared_errors_hybrid, absolute_errors_hybrid = [], []
    squared_errors_random, absolute_errors_random = [], []
    
    valid_users = 0

    for uid in test_users:
        # --- RANKING LOGIC ---
        # 🛠️ FIX 1: Define true_items before trying to use it
        true_items = test_relevant[test_relevant['user_id'] == uid]['item_id'].tolist()
        if not true_items: continue

        user_info = users[users['user_id'] == uid].iloc[0]
        age_grp = user_info['age_group']
        # Fetch gender safely
        gender = user_info['gender'] if 'gender' in user_info else 'Female'

        first_true_item_id = true_items[0]
        item_info = items[items['item_id'] == first_true_item_id].iloc[0]
        occ = item_info['occasion']
        temp = (item_info['min_temp'] + item_info['max_temp']) / 2 

        # 🛠️ FIX 2: Call get_recommendations ONLY ONCE, with the gender argument included
        recs = model.get_recommendations(user_id=uid, temp=temp, occasion=occ, age_group=age_grp, gender=gender, top_n=k)
        rec_items = recs['item_id'].tolist()
        
        random_recs = items.sample(k)['item_id'].tolist()

        # Ranking Metrics
        hybrid_hits_count = len(set(rec_items) & set(true_items))
        precision_hybrid += hybrid_hits_count / k
        recall_hybrid += hybrid_hits_count / len(true_items)
        if hybrid_hits_count > 0: hits_hybrid += 1

        random_hits_count = len(set(random_recs) & set(true_items))
        precision_random += random_hits_count / k
        recall_random += random_hits_count / len(true_items)
        if random_hits_count > 0: hits_random += 1

        # --- PREDICTION LOGIC ---
        all_user_test_interactions = test_data[test_data['user_id'] == uid]
        
        for _, interaction in all_user_test_interactions.iterrows():
            actual_rating = interaction['rating']
            test_item = interaction['item_id']
            
            if test_item in recs['item_id'].values:
                # Safely check which score column exists
                score_col = 'final_score' if 'final_score' in recs.columns else 'kb_score'
                raw_score = recs[recs['item_id'] == test_item][score_col].values[0]
            else:
                raw_score = 2.5 
                
            predicted_rating = np.clip(raw_score, 1.0, 5.0)
            random_predicted_rating = np.random.uniform(1.0, 5.0)
            
            squared_errors_hybrid.append((actual_rating - predicted_rating) ** 2)
            absolute_errors_hybrid.append(abs(actual_rating - predicted_rating))
            
            squared_errors_random.append((actual_rating - random_predicted_rating) ** 2)
            absolute_errors_random.append(abs(actual_rating - random_predicted_rating))

        valid_users += 1

    if valid_users == 0: return pd.DataFrame()
    
    # Compile Final DataFrame
    rmse_hybrid = np.sqrt(np.mean(squared_errors_hybrid))
    mae_hybrid = np.mean(absolute_errors_hybrid)
    rmse_random = np.sqrt(np.mean(squared_errors_random))
    mae_random = np.mean(absolute_errors_random)
    
    return pd.DataFrame({
        "Evaluation Metric": [
            "Precision@5 (Higher is better)", 
            "Recall@5 (Higher is better)", 
            "Hit Rate@5 (Higher is better)", 
            "RMSE (Lower is better)", 
            "MAE (Lower is better)"
        ],
        "Random Baseline": [
            f"{precision_random/valid_users:.3f}", f"{recall_random/valid_users:.3f}", f"{hits_random/valid_users:.3f}", f"{rmse_random:.3f}", f"{mae_random:.3f}"
        ],
        "Hybrid Model (Ours)": [
            f"{precision_hybrid/valid_users:.3f}", f"{recall_hybrid/valid_users:.3f}", f"{hits_hybrid/valid_users:.3f}", f"{rmse_hybrid:.3f}", f"{mae_hybrid:.3f}"
        ]
    })

if __name__ == "__main__":
    from src.data_preprocessing import prepare_data
    from src.recommender_logic import HybridFashionRecommender
    
    print("Loading datasets to run offline evaluation...")
    train, test, items = prepare_data('data/interactions.csv', 'data/items.csv')
    users = pd.read_csv('data/users.csv')
    engine = HybridFashionRecommender(items, train)
    
    results = get_comprehensive_metrics(engine, test, users, items)
    print("\n" + "="*50 + "\n EVALUATION RESULTS\n" + "="*50)
    print(results.to_string(index=False))