import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import time
from collections import defaultdict
import json

MIN_RATINGS_PER_USER = 20
MIN_RATINGS_PER_MOVIE = 20
TOP_USERS = 500000
TOP_RECOMMENDATIONS = 100
K_NEIGHBORS = 10
TEST_SIZE = 0.2
BATCH_SIZE = 1000 

def load_and_prepare_data():
    print("Loading data...")
    ratings = pd.read_csv("cleaned_ratings.csv")
    movies = pd.read_csv("cleaned_movies.csv")

    user_counts = ratings['userId'].value_counts()
    active_users = user_counts[user_counts >=
                               MIN_RATINGS_PER_USER].head(TOP_USERS).index
    ratings = ratings[ratings['userId'].isin(active_users)]

    movie_counts = ratings['movieId'].value_counts()
    popular_movies = movie_counts[movie_counts >= MIN_RATINGS_PER_MOVIE].index
    ratings = ratings[ratings['movieId'].isin(popular_movies)]

    print(
        f"Splitting data into train/test ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})...")
    train, test = train_test_split(
        ratings, test_size=TEST_SIZE, random_state=42, stratify=ratings['userId'])

    return train, test, movies


def build_sparse_matrix(df, user_mapper, movie_mapper):
    rows = [user_mapper[u] for u in df['userId']]
    cols = [movie_mapper[m] for m in df['movieId']]
    return csr_matrix((df['rating'], (rows, cols))), len(user_mapper), len(movie_mapper)


def batch_cosine_similarity(matrix, batch_size=BATCH_SIZE):
    n_users = matrix.shape[0]
    similarity = np.zeros((n_users, n_users), dtype=np.float32)

    for i in range(0, n_users, batch_size):
        i_end = min(i + batch_size, n_users)
        for j in range(0, n_users, batch_size):
            j_end = min(j + batch_size, n_users)
            similarity[i:i_end, j:j_end] = cosine_similarity(
                matrix[i:i_end], matrix[j:j_end])
        print(f"Processed {i_end}/{n_users} users")

    return similarity


def generate_recommendations_batch(train_matrix, similarity, user_mapper, movie_mapper):
    recommendations = {}
    n_users = train_matrix.shape[0]
    movie_ids = [k for k, v in sorted(
        movie_mapper.items(), key=lambda item: item[1])]

    for i in range(0, n_users, BATCH_SIZE):
        batch_users = list(range(i, min(i + BATCH_SIZE, n_users)))
        user_indices = {u: idx for idx, u in enumerate(batch_users)}

        sim_batch = similarity[batch_users]
        top_neighbors = np.argpartition(-sim_batch,
                                        K_NEIGHBORS, axis=1)[:, :K_NEIGHBORS]

        for user_idx in range(len(batch_users)):
            user_sim = sim_batch[user_idx]
            neighbors = top_neighbors[user_idx]

            seen_movies = set(train_matrix[batch_users[user_idx]].indices)
            rec_scores = defaultdict(float)

            for neighbor_idx in neighbors:
                if neighbor_idx == batch_users[user_idx]:
                    continue

                similarity_score = user_sim[neighbor_idx]
                for movie_idx in train_matrix[neighbor_idx].indices:
                    if movie_idx not in seen_movies:
                        rating = train_matrix[neighbor_idx, movie_idx]
                        rec_scores[movie_idx] += similarity_score * \
                            (rating - 3.0)

            top_recs = sorted(rec_scores.items(),
                              key=lambda x: -x[1])[:TOP_RECOMMENDATIONS]
            user_id = next(k for k, v in user_mapper.items()
                           if v == batch_users[user_idx])
            recommendations[user_id] = [movie_ids[movie_idx]
                                        for movie_idx, _ in top_recs]

        print(f"Generated recommendations for {i + len(batch_users)}/{n_users} users")

    return recommendations


def calculate_metrics(recommendations, test_data, all_movies, k=10):
    print("Calculating metrics...")

    if isinstance(recommendations, pd.DataFrame):
        rec_dict = recommendations.groupby(
            'userId')['movieId'].apply(list).to_dict()
    else:
        rec_dict = recommendations
 
    recommended_movies = set(mid for recs in rec_dict.values() for mid in recs)
    coverage = len(recommended_movies) / len(all_movies)

    precision_scores = []
    recall_scores = []

    test_user_movies = test_data.groupby('userId')['movieId'].apply(set)

    for user_id, rec_movies in rec_dict.items():
         if user_id in test_user_movies.index:
            actual_movies = test_user_movies[user_id]

            top_k_recs = set(rec_movies[:k])

            true_positives = top_k_recs & actual_movies

            precision = len(true_positives) / k
            precision_scores.append(precision)

            recall = len(true_positives) / len(actual_movies)
            recall_scores.append(recall)

    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0

    # Additional metrics
    user_counts = recommendations['userId'].value_counts()
    movie_counts = recommendations['movieId'].value_counts()

    metrics = {
        'coverage': coverage,
        'precision@k': avg_precision,
        'recall@k': avg_recall,
        'unique_users': len(user_counts),
        'unique_movies': len(recommended_movies),
        'avg_recs_per_user': user_counts.mean(),
        'most_recommended_movies': movie_counts.nlargest(10).to_dict(),
        'k_used_for_metrics': k
    }

    with open('recommendation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to recommendation_metrics.json")
    print(f"Coverage: {coverage:.2%}")
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"Unique users: {len(user_counts)}")
    print(f"Unique movies recommended: {len(recommended_movies)}")


def main():
    start_time = time.time()

    try:
        recommendations = pd.read_csv("batch_recommendations1.csv")
        movies = pd.read_csv("cleaned_movies.csv")
        print("Loaded existing recommendations, skipping model training...")

        train, test, _ = load_and_prepare_data()
        calculate_metrics(recommendations, test, movies)
        return
    except FileNotFoundError:
        print("No existing recommendations found, running full pipeline...")

    train, test, movies = load_and_prepare_data()

    user_mapper = {u: i for i, u in enumerate(train['userId'].unique())}
    movie_mapper = {m: i for i, m in enumerate(train['movieId'].unique())}

    print("Building matrices...")
    train_matrix, n_users, n_movies = build_sparse_matrix(
        train, user_mapper, movie_mapper)
    test_matrix, _, _ = build_sparse_matrix(test, user_mapper, movie_mapper)

    print("Calculating user similarities...")
    similarity = batch_cosine_similarity(train_matrix)

    print("Generating recommendations...")
    recommendations = generate_recommendations_batch(
        train_matrix, similarity, user_mapper, movie_mapper)

    print("Saving recommendations and calculating metrics...")
    results = []
    for user_id, movie_list in recommendations.items():
        for movie_id in movie_list:
            results.append({'userId': user_id, 'movieId': movie_id})

    recommendations_df = pd.DataFrame(results)
    recommendations_df.to_csv("batch_recommendations1.csv", index=False)

    calculate_metrics(recommendations_df, test, movies)

    print(f"Done! Total execution time: {time.time()-start_time:.2f} seconds")


if __name__ == "__main__":
    main()
