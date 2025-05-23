import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import json
import time

# Константы
MIN_RATINGS_PER_USER = 20
MIN_RATINGS_PER_MOVIE = 20
TOP_USERS = 50000
TOP_RECOMMENDATIONS = 100
K_NEIGHBORS = 10
TEST_SIZE = 0.2
COLD_START_TOP_N = 50  
SVD_COMPONENTS = 50   
K_FOR_METRICS = 10    

def load_and_prepare_data():
    ratings = pd.read_csv("cleaned_ratings.csv")
    movies = pd.read_csv("cleaned_movies.csv")

    active_users = ratings['userId'].value_counts(
    )[ratings['userId'].value_counts() >= MIN_RATINGS_PER_USER].index[:TOP_USERS]
    popular_movies = ratings['movieId'].value_counts(
    )[ratings['movieId'].value_counts() >= MIN_RATINGS_PER_MOVIE].index
    ratings = ratings[ratings['userId'].isin(
        active_users) & ratings['movieId'].isin(popular_movies)]

    train, test = train_test_split(ratings, test_size=TEST_SIZE, random_state=42, stratify=ratings['userId'])
    return train, test, movies


def build_sparse_matrix(df, user_mapper, movie_mapper):
    rows = df['userId'].map(user_mapper).values
    cols = df['movieId'].map(movie_mapper).values
    ratings = df['rating'].values
    
    matrix = csr_matrix((ratings, (rows, cols)), 
                       shape=(len(user_mapper), len(movie_mapper)),
                       dtype=np.float32)
    
    return matrix

def train_svd_model(train_matrix):
    print("Training SVD model...")
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
    user_factors = svd.fit_transform(train_matrix)
    movie_factors = svd.components_.T
    return user_factors, movie_factors


def generate_svd_recommendations(user_factors, movie_factors, user_mapper, movie_inv_mapper):
    predictions = user_factors @ movie_factors.T
    recommendations = {}
    for user_id, user_idx in user_mapper.items():
        top_movies = np.argsort(-predictions[user_idx])[:TOP_RECOMMENDATIONS]
        recommendations[user_id] = [movie_inv_mapper[m] for m in top_movies]
    return recommendations


def get_popular_movies(train_data, n=COLD_START_TOP_N):
    return train_data['movieId'].value_counts().head(n).index.tolist()


def get_content_based_recommendations(user_genres, movies, n=TOP_RECOMMENDATIONS):
    if not user_genres:
        return get_popular_movies(movies, n)
    return movies[movies['genres'].str.contains('|'.join(user_genres))]['movieId'].head(n).tolist()


def hybrid_recommendations(user_id, user_factors, movie_factors, user_mapper, movie_inv_mapper,
                           train_data, movies, user_genres=None):
    if user_id not in user_mapper:
        if user_genres:
            return get_content_based_recommendations(user_genres, movies)
        return get_popular_movies(train_data)

    user_idx = user_mapper[user_id]
    predictions = user_factors[user_idx] @ movie_factors.T
    top_movies = np.argsort(-predictions)[:TOP_RECOMMENDATIONS]
    return [movie_inv_mapper[m] for m in top_movies]



def calculate_metrics(recommendations, test_data, all_movies, k=10):
    print("Calculating metrics...")

    if isinstance(recommendations, pd.DataFrame):
        rec_dict = recommendations.groupby('userId')['movieId'].apply(list).to_dict()
    else:
        rec_dict = recommendations

    recommended_movies = set(mid for recs in rec_dict.values() for mid in recs)
    coverage = len(recommended_movies) / len(all_movies)

    precision_scores = []
    recall_scores = []

    test_user_movies = test_data.groupby('userId')['movieId'].apply(set)

    for user_id, rec_movies in rec_dict.items():
        if user_id in test_user_movies.index and len(rec_movies) > 0:
            actual_movies = test_user_movies[user_id]
            top_k_recs = set(rec_movies[:k])
            
            true_positives = top_k_recs & actual_movies
            
            precision = len(true_positives) / min(k, len(rec_movies)) 
            precision_scores.append(precision)
            
            if len(actual_movies) > 0: 
                recall = len(true_positives) / len(actual_movies)
                recall_scores.append(recall)

    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0

    user_counts = pd.Series([len(recs) for recs in rec_dict.values()])
    all_rec_movies = [mid for recs in rec_dict.values() for mid in recs]
    movie_counts = pd.Series(all_rec_movies).value_counts()

    metrics = {
        'coverage': coverage,
        f'precision@{k}': avg_precision,
        f'recall@{k}': avg_recall,
        'unique_users': len(rec_dict),
        'unique_movies': len(recommended_movies),
        'avg_recs_per_user': user_counts.mean(),
        'most_recommended_movies': movie_counts.nlargest(10).to_dict(),
        'k_used': k
    }

    with open('recommendation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to recommendation_metrics.json")
    print(f"Coverage: {coverage:.2%}")
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"Unique users: {len(rec_dict)}")
    print(f"Unique movies recommended: {len(recommended_movies)}")
    
    return metrics


def main():
    start_time = time.time()

    train, test, movies = load_and_prepare_data()

    user_mapper = {u: i for i, u in enumerate(train['userId'].unique())}
    movie_mapper = {m: i for i, m in enumerate(train['movieId'].unique())}
    movie_inv_mapper = {v: k for k, v in movie_mapper.items()}

    train_matrix = build_sparse_matrix(train, user_mapper, movie_mapper)

    user_factors, movie_factors = train_svd_model(train_matrix)

    print("Generating recommendations...")
    recommendations = {}
    for user_id in train['userId'].unique():
        recommendations[user_id] = hybrid_recommendations(
            user_id, user_factors, movie_factors, user_mapper, movie_inv_mapper, train, movies
        )

    all_movies = set(train['movieId'].unique())
    metrics = calculate_metrics(recommendations, test, all_movies)

    popular_movies = get_popular_movies(train)
    with open('cold_start_recommendations.json', 'w') as f:
        json.dump({'popular_movies': popular_movies, 'metrics': metrics}, f)

    rec_df = pd.DataFrame([
        {'userId': u, 'movieId': m}
        for u, movies in recommendations.items()
        for m in movies
    ])
    rec_df.to_csv('recommendations.csv', index=False)

    print(f"Done! Time: {time.time()-start_time:.2f} sec")


if __name__ == "__main__":
    main()
