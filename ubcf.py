import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

ratings_df = pd.read_csv("cleaned_ratings.csv")
movies_df = pd.read_csv("cleaned_movies.csv")

user_counts = ratings_df['userId'].value_counts()
active_users = user_counts[user_counts >= 20].head(50000).index  
filtered_df = ratings_df[ratings_df['userId'].isin(active_users)]

movie_counts = filtered_df['movieId'].value_counts()
popular_movies = movie_counts[movie_counts >= 20].index
filtered_df = filtered_df[filtered_df['movieId'].isin(popular_movies)]

print(f"Number of active users: {len(active_users)}")
print(f"Number of movies: {len(popular_movies)}")
print(f"Size of the filtered data: {filtered_df.shape}")

user_movie_matrix = filtered_df.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

print(f"Size of user movie matrix: {user_movie_matrix.shape}")

def get_similar_users(user_id, k_neighbors=5):
    if user_id not in user_movie_matrix.index:
        print(f"User {user_id} not found in the matrix")
        return [] 
    
    target_ratings = user_movie_matrix.loc[user_id].values.reshape(1, -1)
    similarities = []
    
    for other_user_id in user_movie_matrix.index:
        if other_user_id != user_id:
            other_ratings = user_movie_matrix.loc[other_user_id].values.reshape(1, -1)
            similarity = cosine_similarity(target_ratings, other_ratings)[0][0]
            similarities.append((other_user_id, similarity))
    
    similar_users = sorted(similarities, key=lambda x: x[1], reverse=True)[:k_neighbors]
    return similar_users

def recommend_movie_ids(user_id, top_n=100, k_neighbors=5):
    similar_users = get_similar_users(user_id, k_neighbors)
    
    if not similar_users:
        print(f"No similar users found for {user_id}")
        return []

    weighted_scores = {}
    for neighbor_id, similarity in similar_users:
        neighbor_ratings = user_movie_matrix.loc[neighbor_id]
        for movie_id, rating in neighbor_ratings.items():
            if user_movie_matrix.loc[user_id, movie_id] == 0 and rating > 0:  
                weighted_scores.setdefault(movie_id, []).append(similarity * rating)
    
    avg_scores = {
        movie_id: sum(scores)/len(scores)
        for movie_id, scores in weighted_scores.items()
    }

    recommended = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(user_id, movie_id) for movie_id, _ in recommended]

for user_id in active_users:
    recommendations = recommend_movie_ids(user_id, top_n=100, k_neighbors=10)
    
    if recommendations:
        for uid, mid in recommendations:
            print(f"userId: {uid}, movieId: {mid}")
    else:
        print(f"No recommendations for userId: {user_id}")
