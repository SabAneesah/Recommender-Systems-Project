# Fashion Outfit Recommender System
**Module:** Recommender Systems  
**Author:** [SABAR S.A.] | **Registration Number:** [215554D]

## 1. Project Overview
This project is a Context-Aware, Knowledge-Based Hybrid Recommender System designed for fashion e-commerce. It solves traditional collaborative filtering issues (like context-blindness and the Cold-Start problem) by combining hard physical constraints (Weather/Occasion/Gender) with stylistic domain expertise (Age Appropriateness/Color Harmony) and TF-IDF Content-Based filtering.

## 2. Libraries Used
Ensure the following Python libraries are installed before running the system:
* **`pandas`**: For data manipulation and tabular data handling.
* **`numpy`**: For numerical operations, random seed generation, and calculating error metrics.
* **`scikit-learn`**: For `TfidfVectorizer` (converting clothing metadata into feature vectors) and `cosine_similarity` (calculating content-based ranking).
* **`streamlit`**: For building the interactive web-based frontend UI.
* **`datetime`**: (Python built-in) For the Time-Aware season detection feature.

*Installation Command:*
`pip install pandas numpy scikit-learn streamlit`

##  3. How to Run the System

**Step 1: Generate the Synthetic Dataset**
The system requires a logical dataset. Run the data generator to create 60 users, 100 gender-and-age-appropriate items, and 650 logical interactions.
```
python data_generator.py
```
*(Expected Output: A ```data/`` folder is created containing users.csv, items.csv, and interactions.csv)*

**Step 2: Launch the Web Interface (Live Demo)**
Start the Streamlit server to interact with the UI.
```
streamlit run app.py
```
*(Expected Output: A browser window will automatically open showing the Fashion Recommender UI.)*

## 4. Expected Output for 3 Core Use Cases
When demonstrating the system, these three use cases highlight the advanced architecture of the algorithm:

**Use Case 1: The Standard Hybrid User (Content + Knowledge)**
Input: Select an existing User ID (e.g., User 1), Occasion: "Casual", Season: "Winter".

Expected Output: The system instantly filters out all Summer and Formal wear (Knowledge-Based). It then ranks the remaining Casual Winter outfits by comparing their TF-IDF metadata against the items User 1 previously rated highly (Content-Based). The top 5 personalized items are displayed with reasoning captions.

**Use Case 2: The Cold-Start Problem (Guest User)**
Input: Toggle "Simulate New User" to ON. Select Age: "Teen", Gender: "Female", Color: "Pastel Pink", Occasion: "Sports".

Expected Output: Because the user has zero interaction history, a standard algorithm would fail. Instead, this system gracefully falls back to the "AI Stylist" Knowledge Base. It successfully recommends 5 age-appropriate, female sports items (like a Breathable Tank or Sports Bra) in Pastel Pink or harmonious colors, proving the system is immune to the Cold-Start problem.

**Use Case 3: Strict Domain Constraints & Serendipity**
Input: Select Gender: "Male", Age: "Senior", Color: "Pastel Pink" or "Lavender".

Expected Output: The Knowledge-Base applies strict Regex gender constraints. It completely blocks female-specific clothing (skirts, dresses) and the forbidden colors (Pastel Pink) for the male profile. Since a direct match is blocked, the Serendipity Engine kicks in—it recommends age-appropriate, masculine outfits in harmonious alternative colors (like Dark Charcoal), providing a highly intelligent and domain-safe recommendation.