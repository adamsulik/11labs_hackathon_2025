import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

def load_resources():
    """Load all necessary resources"""
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    OUTPUTS_DIR = BASE_DIR / 'outputs'
    
    # Load embeddings
    with open(OUTPUTS_DIR / 'movie_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load metadata
    metadata_df = pd.read_csv(DATA_DIR / 'movies_metadata.csv')
    print(f"\nInitial number of movies in metadata: {len(metadata_df)}")
    
    # Clean and convert columns
    metadata_df['id'] = pd.to_numeric(metadata_df['id'], errors='coerce')
    metadata_df['popularity'] = pd.to_numeric(metadata_df['popularity'], errors='coerce')
    
    # Drop rows with invalid IDs or popularity
    metadata_df = metadata_df.dropna(subset=['id', 'popularity'])
    metadata_df['id'] = metadata_df['id'].astype(int)
    print(f"Movies after cleaning: {len(metadata_df)}")
    
    # Filter metadata to only include movies we have embeddings for
    available_ids = set(embeddings.keys())
    metadata_df = metadata_df[metadata_df['id'].isin(available_ids)].copy()
    
    print(f"Movies with available embeddings: {len(available_ids)}")
    print(f"Movies in final dataset: {len(metadata_df)}")
    print("-"*40)
    
    return embeddings, metadata_df

def find_similar_movies(movie_id, embeddings, metadata_df, top_k=5):
    """Find top_k similar movies to the given movie_id"""
    if movie_id not in embeddings:
        print(f"Error: Movie ID {movie_id} not found in embeddings!")
        print("Please choose from the available movies using options 2 or 3 in the menu.")
        return
    
    # Get target movie info
    target_movie = metadata_df[metadata_df['id'] == movie_id].iloc[0]
    print("\n" + "="*80)
    print(f"Finding similar movies to:")
    print(f"Title: {target_movie['title']}")
    print(f"Release Date: {target_movie['release_date']}")
    print(f"Vote Average: {target_movie['vote_average']}")
    print("="*80)
    
    # Get embedding for target movie
    target_embedding = embeddings[movie_id]
    
    # Calculate similarities with all other movies
    similarities = []
    for other_id, other_embedding in embeddings.items():
        if other_id != movie_id:
            sim = cosine_similarity(
                target_embedding.reshape(1, -1),
                other_embedding.reshape(1, -1)
            )[0][0]
            similarities.append((other_id, sim))
    
    # Sort by similarity and get top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_k} Most Similar Movies:")
    print("-"*80)
    for movie_id, sim in similarities[:top_k]:
        movie = metadata_df[metadata_df['id'] == movie_id].iloc[0]
        print(f"\nSimilarity Score: {sim:.3f}")
        print(f"Title: {movie['title']}")
        print(f"Release Date: {movie['release_date']}")
        print(f"Vote Average: {movie['vote_average']}")
        print("-"*40)

def print_sample_movies(metadata_df, n=10, sort_by='popularity'):
    """Print sample movies sorted by specified criterion"""
    print(f"\nSample Movies with Available Embeddings (sorted by {sort_by}):")
    print("-"*80)
    
    try:
        # Ensure numeric sorting column
        if sort_by not in metadata_df.columns or not pd.api.types.is_numeric_dtype(metadata_df[sort_by]):
            print(f"Warning: Cannot sort by {sort_by}, defaulting to ID")
            sort_by = 'id'
        
        # Sort by specified criterion and get samples
        sorted_movies = metadata_df.sort_values(by=sort_by, ascending=False).head(n)
        
        # Print movies in a formatted way
        for _, movie in sorted_movies.iterrows():
            print(f"ID: {int(movie['id']):6d} | "
                  f"Title: {movie['title']:<50} | "
                  f"Year: {str(movie['release_date'])[:4]} | "
                  f"Popularity: {float(movie['popularity']):.1f}")
    
    except Exception as e:
        print(f"Error displaying movies: {str(e)}")
        print("Showing unsorted sample instead:")
        for _, movie in metadata_df.head(n).iterrows():
            print(f"ID: {int(movie['id']):6d} | Title: {movie['title']}")

def interactive_search():
    """Interactive function to search for similar movies"""
    embeddings, metadata_df = load_resources()
    
    while True:
        print("\nInteractive Movie Similarity Search")
        print("="*40)
        print("1. Search by movie ID")
        print("2. List popular movies with available embeddings")
        print("3. List random movies with available embeddings")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            try:
                movie_id = int(input("Enter movie ID: "))
                top_k = int(input("Enter number of similar movies to find (default 5): ") or 5)
                find_similar_movies(movie_id, embeddings, metadata_df, top_k)
            except ValueError:
                print("Please enter valid numbers!")
        
        elif choice == '2':
            n_samples = int(input("How many movies to show? (default 10): ") or 10)
            print_sample_movies(metadata_df, n=n_samples, sort_by='popularity')
        
        elif choice == '3':
            n_samples = int(input("How many movies to show? (default 10): ") or 10)
            print_sample_movies(metadata_df.sample(n=n_samples), n=n_samples, sort_by='popularity')
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    interactive_search()
