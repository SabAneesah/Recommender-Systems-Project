import pandas as pd
import numpy as np
import os

def generate_logical_dataset():
    if not os.path.exists('data'):
        os.makedirs('data')

    np.random.seed(42)

    # --- 1. THE LOGIC ENGINE: Age & Color Knowledge ---
    age_color_rules = {
        'Teen': ['Pastel Pink', 'Mint Green', 'Sky Blue', 'Lavender', 'Peach', 'Light Yellow', 'Classic Black', 'Navy Blue', 'Pure White'],
        'Adult': ['Classic Black', 'Navy Blue', 'Pure White', 'Slate Grey', 'Burgundy'],
        'Senior': ['Dark Charcoal', 'Deep Maroon', 'Forest Green', 'Midnight Blue', 'Chocolate Brown']
    }

    wardrobe_rules = {
        'Winter': {
            'Formal': ['Wool Overcoat', 'Tweed Blazer', 'Velvet Evening Gown'],
            'Casual': ['Heavy Cardigan', 'Puffer Jacket', 'Fleece Hoodie', 'Thermal Flannel'],
            'Sports': ['Thermal Tracksuit', 'Windbreaker', 'Compression Leggings'],
            'temps': (-15, 10)
        },
        'Summer': {
            'Formal': ['Linen Suit', 'Silk Sundress', 'Lightweight Blazer'],
            'Casual': ['Cotton T-shirt', 'Denim Shorts', 'Tank-top', 'Mini Skirt'],
            'Sports': ['Sports Bra', 'Running Shorts', 'Breathable Tank'],
            'temps': (25, 40)
        },
        'Spring/Autumn': {
            'Formal': ['Cotton Blazer', 'Pencil Skirt', 'Button-up Shirt'],
            'Casual': ['Light Sweater', 'Jeans', 'Denim Jacket', 'Midi Dress'],
            'Sports': ['Yoga Pants', 'Long Sleeve Tech Tee', 'Joggers'],
            'temps': (11, 24)
        }
    }

    # --- 2. GENERATE 100 ITEMS ---
    item_list = []
    
    for i in range(1, 101):
        season = np.random.choice(list(wardrobe_rules.keys()))
        occ = np.random.choice(['Formal', 'Casual', 'Sports'])
        item_base = np.random.choice(wardrobe_rules[season][occ])
        t_min, t_max = wardrobe_rules[season]['temps']
        
        # RULE: Logic for Age Group assignment
        if item_base in ['Sports Bra', 'Mini Skirt', 'Fleece Hoodie']:
            age_group = np.random.choice(['Teen', 'Adult']) 
        elif item_base in ['Tweed Blazer', 'Wool Overcoat']:
            age_group = np.random.choice(['Adult', 'Senior']) 
        else:
            age_group = np.random.choice(['Teen', 'Adult', 'Senior'])
        
        # RULE: Logic for Color based on Age Group
        color = np.random.choice(age_color_rules[age_group])

        # 🌟 NEW RULE: Logic for Target Gender Assignment
        forbidden_male_items = ['skirt', 'bra', 'dress', 'gown', 'tank', 'leggings']
        is_female_clothing = any(sub in item_base.lower() for sub in forbidden_male_items)
        
        if is_female_clothing or color in ['Pastel Pink', 'Lavender']:
            target_gender = 'Female'
        else:
            target_gender = np.random.choice(['Unisex', 'Male'])
        
        item_list.append({
            'item_id': i,
            'name': f"{color} {item_base}",
            'age_group': age_group,
            'target_gender': target_gender,
            'occasion': occ,
            'season': season,
            'color': color,
            'min_temp': t_min,
            'max_temp': t_max
        })

    items_df = pd.DataFrame(item_list)
    
    # Feature Preprocessing: Add gender to metadata for TF-IDF to learn from
    items_df['metadata'] = (items_df['name'] + " " + items_df['occasion'] + " " + 
                            items_df['age_group'] + " " + items_df['season'] + " " + 
                            items_df['target_gender']).str.lower()

    # --- 3. GENERATE 60 USERS ---
    users_df = pd.DataFrame({
        'user_id': range(1, 61),
        'age_group': np.random.choice(list(age_color_rules.keys()), 60),
        'gender': np.random.choice(['Male', 'Female'], 60), # 🌟 NEW: Added User Gender
        'preferred_color_type': np.random.choice(['Pastel', 'Dark', 'Classic'], 60)
    })

    # --- 4. GENERATE 650 INTERACTIONS (With Logical Human Behavior) ---
    interactions = []
    for _ in range(650):
        u_id = np.random.randint(1, 61)
        i_id = np.random.randint(1, 101)
        
        user_gender = users_df[users_df['user_id'] == u_id]['gender'].values[0]
        user_age = users_df[users_df['user_id'] == u_id]['age_group'].values[0]
        
        item_gender = items_df[items_df['item_id'] == i_id]['target_gender'].values[0]
        item_age = items_df[items_df['item_id'] == i_id]['age_group'].values[0]
        
        # 🌟 LOGICAL RATING RULES: The AI can actually learn these patterns!
        if user_gender == 'Male' and item_gender == 'Female':
            rating = np.random.choice([1, 2])  # Incompatible gender
        elif user_age == item_age:
            rating = np.random.choice([4, 5])  # Perfect demographic match!
        else:
            rating = np.random.choice([2, 3])  # Meh, not really my style
            
        interactions.append({
            'user_id': u_id,
            'item_id': i_id,
            'rating': rating,
            'timestamp': pd.Timestamp('2026-01-01') + pd.Timedelta(hours=np.random.randint(0, 1000))
        })
        
    interactions_df = pd.DataFrame(interactions)

    # Save all three mandatory files 
    items_df.to_csv('data/items.csv', index=False)
    users_df.to_csv('data/users.csv', index=False)
    interactions_df.to_csv('data/interactions.csv', index=False)
    
    print(f"✅ Success: Dataset generated with Gender rules! ({len(items_df)} items, {len(users_df)} users, 650 interactions)")

if __name__ == "__main__":
    generate_logical_dataset()