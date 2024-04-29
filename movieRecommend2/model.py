import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def create_tfidf_features(data_file, description_column='description', save_path='tfidf_model.pickle'):
    data = pd.read_csv(data_file, encoding='ISO-8859-1')
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    tfidf_matrix = vectorizer.fit_transform(data[description_column])
    
    with open(save_path, 'wb') as f:
        pickle.dump((vectorizer, tfidf_matrix), f)
    
    print("TF-IDF model and matrix saved to", save_path)

if __name__ == '__main__':
    data_file = 'netflix_titles.csv'
    create_tfidf_features(data_file)
