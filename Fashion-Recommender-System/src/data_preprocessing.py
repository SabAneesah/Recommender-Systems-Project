import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(interactions_path, items_path):
    # Load data
    interactions = pd.read_csv(interactions_path)
    items = pd.read_csv(items_path)
    
    # 1. Cleaning: Remove duplicates or nulls
    interactions.drop_duplicates(inplace=True)
    interactions.dropna(inplace=True)
    
    # 2. Train/Test Split (80/20)
    train_data, test_data = train_test_split(
        interactions, test_size=0.2, random_state=42
    )
    
    # 3. Feature Preprocessing
    items['metadata'] = (
        items['name'].astype(str) + " " +
        items['age_group'].astype(str) + " " + 
        items['target_gender'].astype(str) + " " +
        items['occasion'].astype(str) + " " + 
        items['season'].astype(str) + " " +
        items['color'].astype(str)
    ).str.lower()
    
    return train_data, test_data, items

if __name__ == "__main__":
    print("Data preprocessing module ready.")