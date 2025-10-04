# MovieLens Recommendation System

## Project Overview
A comprehensive movie recommendation system built using collaborative filtering, matrix factorization, and deep learning techniques on the MovieLens 100K dataset.

## Features
- **Matrix Factorization with SVD**: Advanced recommendation using Singular Value Decomposition
- **Item-Based Collaborative Filtering**: Similarity-based recommendations
- **Neural Collaborative Filtering (NCF)**: Deep learning approach with user/item embeddings
- **Interactive Interface**: Easy-to-use Gradio web interface

## Dataset
- **Source**: MovieLens 100K Dataset
- **Size**: 100,000 ratings from 943 users on 1,682 movies
- **Rating Scale**: 1-5 stars

## Methods Implemented

### 1. Item-Based Collaborative Filtering
Uses cosine similarity between items to find similar movies and recommend based on user's past preferences.

**How it works**:
- Calculate similarity between all movie pairs
- For each unrated movie, find similar movies the user rated
- Predict rating as weighted average based on similarity

### 2. Matrix Factorization (SVD)
Decomposes the user-item rating matrix into lower-dimensional representations to predict missing ratings.

**How it works**:
- Decompose rating matrix: R ≈ U × Σ × V^T
- U: User factors (943 × 50)
- V: Movie factors (1682 × 50)
- Predict missing ratings using learned factors

### 3. Neural Collaborative Filtering (NCF)
Deep learning approach combining matrix factorization with multi-layer perceptron.

**Architecture**:
- **GMF Path**: Generalized Matrix Factorization (element-wise product)
- **MLP Path**: Multi-Layer Perceptron [64→32→16]
- **Embeddings**: 50-dimensional learned representations
- **Output**: Combined prediction from both paths

**Advantages**:
- Learns non-linear user-item interactions
- More accurate rating predictions (RMSE: 0.92)
- Provides interpretable embeddings
- Can find similar users/movies in latent space

## Evaluation Metrics
- **Precision@K**: Measures the proportion of relevant items in top-K recommendations
- **Recall@K**: Measures the proportion of relevant items that were recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain - measures ranking quality

## Performance Results

| Method | Precision@10 | Recall@10 | NDCG@10 | 
|--------|-------------|-----------|---------|
| Item-Based CF | 0.0380 | 0.0209 | 0.0367 | 
| **SVD** | 0.1220 | 0.1150 | 0.1578 | 
| Neural (NCF) | 0.1800 | 0.0751 | 0.1820 |
**Best Overall Method**: **Neural (NCF)**

## Why Neural (NCF) Works Best

1. **Dimensionality Reduction**: SVD captures latent features effectively, reducing noise in sparse data
2. **Generalization**: Better handles the cold-start problem and generalizes well to unseen data
3. **Scalability**: More efficient for large-scale recommendations 
4. **Pattern Recognition**: Identifies hidden patterns in user preferences beyond simple similarity
5. **Optimal for Dataset Size**: For 100K ratings, embedding method is more efficient

## When to Use Each Method

### Use **SVD** when:
- Need fast, real-time recommendations
- For best ranking quality
- Production deployment is priority

### Use **Neural (NCF)** when:
- Dataset is very large 
- Need embedding representations
- Want to incorporate side information
- Rating prediction accuracy is critical
- Requires longer training time

### Use **Item-Based CF** when:
- Need explainable recommendations
- Cold-start for new items
- Simple, interpretable system required

## Usage

### Main Recommendation Function
```python
def recommend_movies(user_id, N=10, model=ncf_model):
    if user_id not in user2idx:
        print(f"User {user_id} not found")
        return []
    
    user_idx = user2idx[user_id]
    
    user_train_movies = train_data[train_data['user_id'] == user_id]['item_id'].values
    rated_movie_indices = [movie2idx[m] for m in user_train_movies if m in movie2idx]
    
    all_movie_indices = list(range(num_movies))
    
    unrated_movie_indices = [m for m in all_movie_indices if m not in rated_movie_indices]
    
    if len(unrated_movie_indices) == 0:
        return []
    
    user_indices = np.array([user_idx] * len(unrated_movie_indices))
    movie_indices = np.array(unrated_movie_indices)
    
    predictions = model.predict([user_indices, movie_indices], verbose=0)
    predictions = predictions.flatten()
    
    predictions = predictions * 4 + 1
    
    top_indices = np.argsort(predictions)[::-1][:N]
    
    recommendations = []
    for idx in top_indices:
        movie_idx = unrated_movie_indices[idx]
        movie_id = idx2movie[movie_idx]
        movie_title = movies[movies['movie_id'] == movie_id]['title'].values
        predicted_rating = predictions[idx]
        
        if len(movie_title) > 0:
            recommendations.append((movie_id, movie_title[0], predicted_rating))
    
    return recommendations
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the MovieLens 100K dataset from:
   - https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset/data
   - Place `u.data` and `u.item` files in the project directory

4. Run the main implementation:
```bash
python movie_recommendation_system.ipynb
```

5. (Optional) Train neural model:
```bash
python neural_collaborative_filtering.ipynb
```

6. Run the web application:
```bash
python gradio_app.ipynb
```

## Project Structure
```
├── movie_recommendation_system.py  # Main implementation (CF + SVD)
├── neural_collaborative_filtering.py  # Neural enhancement
├── gradio_app.py                          # Gradio web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── u.data                         # Ratings data (download required)
├── u.item                         # Movie metadata (download required)
```

## Technical Details

### Data Processing
- Created user-item rating matrix (943 × 1682)
- Handled missing values with zero-filling
- Data sparsity: ~93.7%
- Train-test split: 80-20

### SVD Configuration
- **Latent factors (k)**: 50
- **Normalization**: User-mean centering
- **Reconstruction**: U × Σ × V^T + user_mean

### Item-Based CF Configuration
- **Similarity metric**: Cosine similarity
- **Weighting**: Similarity-weighted average
- **Threshold**: Top-20 similar items

### Neural Model Architecture
```
Input: User ID, Movie ID

User Embedding (50d) ──┐
                       ├──> Element-wise Product ──┐
Movie Embedding (50d) ─┘                          │
                                                  ├──> Concatenate ──> Dense ──> Output
User Embedding (50d) ──┐                          │
                       ├──> Concatenate ──> [64→32→16] MLP ─┘
Movie Embedding (50d) ─┘

Loss: MSE
Optimizer: Adam (lr=0.001)
Epochs: 10
Batch Size: 256
```

### Deployment
- **Platform**: Hugging Face Spaces
- **Framework**: Gradio
- **Interface**: Web-based interactive UI
- **Methods Available**: SVD, Neural (NCF)
