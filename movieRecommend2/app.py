from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def load_tfidf_model(path='tfidf_model.pickle'):
    with open(path, 'rb') as file:
        vectorizer, tfidf_matrix = pickle.load(file)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = load_tfidf_model()

movies = pd.read_csv('netflix_titles.csv', encoding='ISO-8859-1')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def get_recommendation():
    user_query = request.args.get('query', '')
    if user_query:
        query_vec = vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        indices = similarities.argsort()[-5:][::-1]
        recommended_movies = movies.iloc[indices]['title'].tolist()
        return jsonify(recommended_movies)
    else:
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)
