from network_embedding import MovieMatcher
from typing import List


def process_suggested_df(sugggested_df):
    movie_ids = sugggested_df['movie_id'].tolist()
    titles_with_desc = sugggested_df[['title', 'overview']].to_markdown(index=False)
    return movie_ids, titles_with_desc

movie_tools = [{
    "type": "function",
    "function": {
        "name": "suggest_movies",
        "description": "Find similar movies based on given movie titles. Returns a pandas dataframe of recommended movies with their details.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_titles": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of movie titles to find recommendations for",
                }
            },
            "required": ["query_titles"],
            "additionalProperties": False
        },
        "strict": True
    }
}]

