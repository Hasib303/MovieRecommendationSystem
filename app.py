import gradio as gr
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

def load_data():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, 
                         names=['movie_id', 'title', 'release_date', 'video_release_date', 
                                'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                'Thriller', 'War', 'Western'])
    
    user_item_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    
    return ratings, movies, user_item_matrix

ratings, movies, user_item_matrix = load_data()

def matrix_factorization_svd(user_id, N=10, k=50):
    R = user_item_matrix.values
    user_ratings_mean = np.mean(R, axis=1)
    R_normalized = R - user_ratings_mean.reshape(-1, 1)
    
    U, sigma, Vt = svds(R_normalized, k=k)
    sigma = np.diag(sigma)
    
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    predictions_df = pd.DataFrame(predicted_ratings, 
                                  index=user_item_matrix.index,
                                  columns=user_item_matrix.columns)
    
    user_predictions = predictions_df.loc[user_id]
    user_rated = user_item_matrix.loc[user_id]
    user_predictions = user_predictions[user_rated == 0]
    
    top_movies = user_predictions.sort_values(ascending=False).head(N)
    return top_movies.index.tolist()

def recommend_movies(user_id, N=10):
    if user_id not in user_item_matrix.index:
        return "User ID not found. Please enter a valid user ID (1-943)"

    movie_ids = matrix_factorization_svd(user_id, N)

    recommendations = []
    for idx, movie_id in enumerate(movie_ids, 1):
        movie_title = movies[movies['movie_id'] == movie_id]['title'].values
        if len(movie_title) > 0:
            recommendations.append(f"{idx}. {movie_title[0]}")
    return "\n".join(recommendations)

def get_user_history(user_id):
    if user_id not in user_item_matrix.index:
        return "User ID not found"
    
    user_ratings = ratings[ratings['user_id'] == user_id].sort_values('rating', ascending=False).head(10)
    
    history = []
    for _, row in user_ratings.iterrows():
        movie_title = movies[movies['movie_id'] == row['item_id']]['title'].values
        if len(movie_title) > 0:
            history.append(f"★ {row['rating']:.0f}/5 - {movie_title[0]}")
    
    return "\n".join(history) if history else "No rating history found"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
    )
    
    with gr.Row():
        with gr.Column():
            user_input = gr.Number(label="Enter User ID", value=196, precision=0)
            num_recs = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="Number of Recommendations")
            recommend_btn = gr.Button("Get Recommendations", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(label="Recommended Movies", lines=15)
    
    with gr.Row():
        history_btn = gr.Button("View User's Rating History")
        history_output = gr.Textbox(label="User's Top Rated Movies", lines=10)
    
    gr.Markdown(
    )
    
    recommend_btn.click(
        fn=recommend_movies,
        inputs=[user_input, num_recs],
        outputs=output
    )
    
    history_btn.click(
        fn=get_user_history,
        inputs=user_input,
        outputs=history_output
    )

demo.launch()