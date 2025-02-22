from .network_embed import setup_paths, load_data
from .movies_fuzzy_matching import find_similar_titles
from .movie_matcher import MovieMatcher

__all__ = [
    'setup_paths',
    'load_data',
    'find_similar_movies',
    'MovieMatcher'
]
