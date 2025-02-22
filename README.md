# 11labs_hackathon_2025
Contributors: Adam Sulik, Michał Stachowicz

data: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data

Data files should be stored in `data/` directory in the main project

For MacOS users: to freely download sample network from pyg library it requires to setup ssl cerificates first: `bash install_certs.sh`.

See and run notebook `notebooks/sage_embedding_example` to validate if PyG was set up properly.

## Using MovieMatcher

The `MovieMatcher` class provides an easy way to find similar movies based on both title matching and content similarity.

### Basic Usage

```python
from network_embedding import MovieMatcher

# Initialize the matcher
matcher = MovieMatcher()

# Example list of movies to search for
movies = ["Inception", "John Wick", "Avengers", "Django"]

# Get title matches
matched_titles = matcher.match_titles(movies)
print("\nTitle matching results:")
print(matched_titles)

# Expected output:
# query_title best_matching_title  best_matching_id  score
#   Inception           Inception             27205  1.000
#   John Wick           John Wick            245891  1.000
#      Django              Django             10772  1.000
#      Django              Django            436334  1.000
#    Avengers        The Avengers              9320  0.951
#    Avengers        The Avengers             24428  0.951

# Find similar movies
similar_movies = matcher.find_movies(movies)
print("\nSimilar movies results:")
print(similar_movies)

# Expected output will include:
# - movie_id: ID of the similar movie
# - title: Title of the similar movie
# - release_date: Release date
# - vote_average: Average vote score
# - similarity_score: How similar the movie is
# - query_title: Original title that was searched
# - matched_title: The exact match found in database
```

### Available Methods

1. `match_titles(query_titles, threshold=80)`: Find similar titles using fuzzy matching
2. `find_similar_by_content(movie_id, n_similar=5)`: Find similar movies based on content embeddings
3. `find_movies(query_titles, threshold=80, n_similar=5)`: Complete pipeline that combines both title matching and content similarity