from .network_embed import setup_paths, load_data
from typing import List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_titles(movies: List[str], metadata_df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Find similar movie titles using TF-IDF and cosine similarity.
    
    Args:
        movies: List of movie titles to find matches for
        metadata_df: DataFrame containing movie metadata with 'original_title' column
        threshold: Minimum similarity score (0-1) to consider a match, default set to 0.9 for high precision
    
    Returns:
        DataFrame with columns: query_title, best_matching_title, best_matching_id, score
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(lowercase=True, analyzer='char_wb', ngram_range=(2, 3))
    
    # Combine all titles and fit vectorizer
    all_titles = list(metadata_df['original_title']) + movies
    tfidf_matrix = vectorizer.fit_transform(all_titles)
    
    # Split matrices for database titles and query titles
    db_matrix = tfidf_matrix[:len(metadata_df)]
    query_matrix = tfidf_matrix[len(metadata_df):]
    
    # Calculate similarities
    similarities = cosine_similarity(query_matrix, db_matrix)
    
    results = []
    for idx, movie in enumerate(movies):
        # Get similarities for current movie
        movie_similarities = similarities[idx]
        
        # Find matches above threshold
        matches = np.where(movie_similarities >= threshold)[0]
        for match_idx in matches:
            matched_title = metadata_df.iloc[match_idx]['original_title']
            score = movie_similarities[match_idx]
            results.append({
                'query_title': movie,
                'best_matching_title': matched_title,
                'best_matching_id': metadata_df.iloc[match_idx]['id'],
                'score': round(float(score), 3)
            })
    
    return pd.DataFrame(results).sort_values('score', ascending=False)

def main():
    # Load metadata
    _, DATA_DIR, _ = setup_paths()
    _, metadata_df = load_data(DATA_DIR)
    
    # Find similar movies
    movies = ["Inception", "John Wick", "Avengers", "Django"]
    similar_movies_df = find_similar_titles(movies, metadata_df)
    
    # Display results
    print("\nMatching results:")
    print(similar_movies_df.to_string(index=False))

if __name__ == "__main__":
    main()