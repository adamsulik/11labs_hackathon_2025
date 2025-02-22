from .movies_fuzzy_matching import find_similar_titles
from .explore_similarities import load_resources
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class MovieMatcher:
    def __init__(self):
        self.embeddings, self.metadata_df = load_resources()
    
    def match_titles(self, query_titles, threshold=80):
        """Find similar titles using fuzzy matching"""
        return find_similar_titles(query_titles, self.metadata_df)

    def find_similar_by_content(self, movie_id, n_similar=5):
        """Find similar movies based on content embeddings"""
        if movie_id not in self.embeddings:
            return pd.DataFrame()
        
        # Get embedding for target movie
        target_embedding = self.embeddings[movie_id]
        
        # Calculate similarities with all other movies
        similarities = []
        for other_id, other_embedding in self.embeddings.items():
            if other_id != movie_id:
                sim = cosine_similarity(
                    target_embedding.reshape(1, -1),
                    other_embedding.reshape(1, -1)
                )[0][0]
                similarities.append((other_id, sim))
        
        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:n_similar]
        
        # Create DataFrame with results
        similar_movies = []
        for sim_id, sim_score in top_similar:
            movie = self.metadata_df[self.metadata_df['id'] == sim_id].iloc[0]
            similar_movies.append({
                'movie_id': sim_id,
                'title': movie['title'],
                'release_date': movie['release_date'],
                'vote_average': movie['vote_average'],
                'similarity_score': sim_score
            })
        
        return pd.DataFrame(similar_movies)

    def find_movies(self, query_titles, threshold=80, n_similar=5):
        """Complete pipeline: match titles and find similar movies"""
        # First find matching titles
        matched_df = self.match_titles(query_titles, threshold)
        
        # For each matched movie, find similar ones
        all_similar_movies = []
        for _, row in matched_df.iterrows():
            similar_df = self.find_similar_by_content(row['best_matching_id'], n_similar)
            if not similar_df.empty:
                similar_df['query_title'] = row['query_title']
                similar_df['matched_title'] = row['best_matching_title']
                all_similar_movies.append(similar_df)
        
        if all_similar_movies:
            return pd.concat(all_similar_movies, ignore_index=True)
        return pd.DataFrame()


if __name__ == "__main__":
    # Initialize matcher
    matcher = MovieMatcher()
    
    # Example usage
    movies = ["Inception", "John Wick", "Avengers", "Django"]
    
    # First show matched titles
    matched_titles = matcher.match_titles(movies)
    print("\nTitle matching results:")
    print(matched_titles.to_string(index=False))
    
    # Then show similar movies
    similar_movies_df = matcher.find_movies(movies)
    print("\nSimilar movies results:")
    print(similar_movies_df.to_string(index=False))

