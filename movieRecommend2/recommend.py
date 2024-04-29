from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(query, tfidf_matrix, movies, vectorizer, top_n=5):

    query_vec = vectorizer.transform([query])

    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    indices = similarities.argsort()[-top_n:][::-1]

    return movies.iloc[indices]['title'].tolist()
