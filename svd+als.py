import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import json
import time
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

# Константы
MIN_RATINGS_PER_USER = 15
MIN_RATINGS_PER_MOVIE = 15
TOP_USERS = 50000
TOP_RECOMMENDATIONS = 100 
K_NEIGHBORS = 15
TEST_SIZE = 0.2
COLD_START_TOP_N = 100
SVD_COMPONENTS = 100
ALS_FACTORS = 100
DIVERSITY_FACTOR = 0.2

def load_and_prepare_data():
    ratings = pd.read_csv("cleaned_ratings.csv")
    movies = pd.read_csv("cleaned_movies.csv")

    # Фильтрация
    user_counts = ratings['userId'].value_counts()
    movie_counts = ratings['movieId'].value_counts()
    
    active_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index[:TOP_USERS]
    popular_movies = movie_counts[movie_counts >= MIN_RATINGS_PER_MOVIE].index
    
    ratings = ratings[ratings['userId'].isin(active_users) & 
                     ratings['movieId'].isin(popular_movies)]
    
    # Взвешивание рейтингов
    ratings['weighted_rating'] = ratings['rating'] * np.log1p(ratings.groupby('userId')['rating'].transform('count'))
    
    train, test = train_test_split(ratings, test_size=TEST_SIZE, random_state=42, 
                                  stratify=ratings['userId'])
    return train, test, movies

def build_sparse_matrix(df, user_mapper, movie_mapper, weighted=False):
    rows = df['userId'].map(user_mapper).values
    cols = df['movieId'].map(movie_mapper).values
    ratings = df['weighted_rating' if weighted else 'rating'].values
    
    matrix = csr_matrix((ratings, (rows, cols)), 
                       shape=(len(user_mapper), len(movie_mapper)),
                       dtype=np.float32)
    return matrix

def train_models(train_matrix):
    print("Training models...")
    
    # SVD
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
    user_factors_svd = svd.fit_transform(train_matrix)
    movie_factors_svd = svd.components_.T
    
    # ALS
    als_model = AlternatingLeastSquares(factors=ALS_FACTORS, regularization=0.01)
    als_model.fit(bm25_weight(train_matrix.T).T)
    user_factors_als = als_model.user_factors
    movie_factors_als = als_model.item_factors
    
    return {
        'svd': (user_factors_svd, movie_factors_svd),
        'als': (user_factors_als, movie_factors_als)
    }

def generate_recommendations(models, user_mapper, movie_inv_mapper, train_data, movies):
    user_factors_svd, movie_factors_svd = models['svd']
    user_factors_als, movie_factors_als = models['als']
    
    recommendations = {}
    movie_popularity = train_data['movieId'].value_counts(normalize=True).to_dict()
    
    for user_id, user_idx in user_mapper.items():
        svd_scores = user_factors_svd[user_idx] @ movie_factors_svd.T
        als_scores = user_factors_als[user_idx] @ movie_factors_als.T

        combined_scores = (
            0.5 * svd_scores + 
            0.5 * als_scores + 
            DIVERSITY_FACTOR * np.array([movie_popularity.get(movie_inv_mapper[i], 0) 
                                       for i in range(len(movie_inv_mapper))]))
        
        top_movies = np.argsort(-combined_scores)[:TOP_RECOMMENDATIONS]
        recommendations[user_id] = [movie_inv_mapper[m] for m in top_movies]
    
    return recommendations

def get_popular_movies(train_data, movies, n=TOP_RECOMMENDATIONS):  
    """Топ-N популярных фильмов"""
    popular = train_data.groupby('movieId').agg(
        count=('rating', 'count'),
        mean_rating=('rating', 'mean')
    )
    popular['score'] = popular['count'] * popular['mean_rating']
    return popular.nlargest(n, 'score').index.tolist()

def hybrid_recommendations(user_id, models, user_mapper, movie_inv_mapper,
                          train_data, movies, user_genres=None):
    if user_id not in user_mapper:
        if user_genres:
            genre_movies = movies[movies['genres'].str.contains('|'.join(user_genres))]
            popular_in_genre = train_data[train_data['movieId'].isin(genre_movies['movieId'])]
            return get_popular_movies(popular_in_genre, movies, TOP_RECOMMENDATIONS)
        return get_popular_movies(train_data, movies, TOP_RECOMMENDATIONS)
    
    user_idx = user_mapper[user_id]
    user_factors_svd, movie_factors_svd = models['svd']
    user_factors_als, movie_factors_als = models['als']
    
    svd_scores = user_factors_svd[user_idx] @ movie_factors_svd.T
    als_scores = user_factors_als[user_idx] @ movie_factors_als.T
    
    combined_scores = 0.6 * svd_scores + 0.4 * als_scores
    top_movies = np.argsort(-combined_scores)[:TOP_RECOMMENDATIONS]
    
    return [movie_inv_mapper[m] for m in top_movies]

def calculate_metrics(recommendations, test_data, all_movies, k=10):
    rec_dict = recommendations if isinstance(recommendations, dict) else \
              recommendations.groupby('userId')['movieId'].apply(list).to_dict()
    
    assert all(len(recs) == TOP_RECOMMENDATIONS for recs in rec_dict.values()), \
           "Не все пользователи имеют 100 рекомендаций"
    
    recommended_movies = set(mid for recs in rec_dict.values() for mid in recs)
    coverage = len(recommended_movies) / len(all_movies)
    
    test_user_movies = test_data.groupby('userId')['movieId'].apply(set)
    
    precisions, recalls = [], []
    for user_id, recs in rec_dict.items():
        if user_id in test_user_movies.index:
            actual = test_user_movies[user_id]
            recommended = set(recs[:k])
            tp = len(actual & recommended)
            
            precisions.append(tp / min(k, len(recs)))
            if len(actual) > 0:
                recalls.append(tp / len(actual))
    
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    
    metrics = {
        'coverage': coverage,
        f'precision@{k}': avg_precision,
        f'recall@{k}': avg_recall,
        'unique_users': len(rec_dict),
        'unique_movies': len(recommended_movies),
        'avg_recs_per_user': TOP_RECOMMENDATIONS,
        'top_recommended_movies': pd.Series(
            [mid for recs in rec_dict.values() for mid in recs]
        ).value_counts().nlargest(10).to_dict()
    }
    
    with open('3recommendation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Coverage: {coverage:.2%}, Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}")
    return metrics

def main():
    start_time = time.time()
    
    train, test, movies = load_and_prepare_data()
    
    unique_users = train['userId'].unique()
    unique_movies = train['movieId'].unique()
    
    user_mapper = {u: i for i, u in enumerate(unique_users)}
    movie_mapper = {m: i for i, m in enumerate(unique_movies)}
    movie_inv_mapper = {v: k for k, v in movie_mapper.items()}
    
    print("Building matrices...")
    train_matrix = build_sparse_matrix(train, user_mapper, movie_mapper)
    weighted_matrix = build_sparse_matrix(train, user_mapper, movie_mapper, weighted=True)
    
    models = train_models(weighted_matrix)
    
    print("Generating recommendations...")
    recommendations = generate_recommendations(models, user_mapper, movie_inv_mapper, train, movies)
    
    for user_id, recs in list(recommendations.items())[:5]:
        print(f"User {user_id} has {len(recs)} recommendations")
    
    all_movies = set(unique_movies)
    metrics = calculate_metrics(recommendations, test, all_movies)
    
    rec_df = pd.DataFrame([
        {'userId': u, 'movieId': m}
        for u, recs in recommendations.items()
        for m in recs
    ])
    rec_df.to_csv('3recommendations.csv', index=False)
    
    popular_movies = get_popular_movies(train, movies)
    with open('3cold_start_recommendations.json', 'w') as f:
        json.dump({'popular_movies': popular_movies, 'metrics': metrics}, f)
    
    print(f"Total time: {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    main()