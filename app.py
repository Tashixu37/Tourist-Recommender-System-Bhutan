from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load data
data = pd.read_csv('/content/Data.csv', encoding='latin1')
data.columns = data.columns.str.strip().str.lower()
data['type'] = data['type'].str.lower()
data['features'] = data['type'] + ' ' + data['season_availability'] + ' ' + data['dzongkhag'] + ' ' + data['fee']

# Create feature matrix
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(data['features'])
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

def map_budget_to_fee(budget):
    if budget == 'low':
        return 'free'
    elif budget == 'medium':
        return 'medium fee'
    elif budget == 'high':
        return 'high fee'
    else:
        return 'free'

def recommend(attractions, attraction_type, season, dzongkhag, budget):
    fee_value = map_budget_to_fee(budget)
    
    # First, try to find attractions in the specified dzongkhag and of the chosen attraction type
    matched_indices = attractions[
        attractions['features'].str.contains(attraction_type, case=False) & 
        attractions['season_availability'].str.contains(season, case=False) &
        attractions['dzongkhag'].str.contains(dzongkhag, case=False)
    ].index
    
    # If no attractions found in the specified dzongkhag and attraction type, try to find in any dzongkhag
    if len(matched_indices) == 0:
        matched_indices = attractions[
            attractions['features'].str.contains(attraction_type, case=False) & 
            attractions['season_availability'].str.contains(season, case=False)
        ].index
    
    # If still no attractions found, try to find based on budget
    if len(matched_indices) == 0:
        matched_indices = attractions[
            attractions['features'].str.contains(fee_value, case=False)
        ].index
        
    if len(matched_indices) == 0:
        return pd.DataFrame(columns=['name', 'ratings', 'dzongkhag', 'fee'])
    else:
        idx = matched_indices[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    attraction_indices = [i[0] for i in sim_scores]
    return attractions.iloc[attraction_indices][['name', 'ratings', 'dzongkhag', 'fee']]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        attraction_type = request.form['attraction_type'].lower()
        season = request.form['season'].lower()
        dzongkhag = request.form['dzongkhag'].lower()
        budget = request.form['budget'].lower()
        recommendations = recommend(data, attraction_type, season, dzongkhag, budget)
        if not recommendations.empty:
            return render_template('recommendations.html', recommendations=recommendations.to_html())
        else:
            return "No attractions found matching your criteria."
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)