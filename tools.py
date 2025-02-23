from network_embedding import MovieMatcher
from typing import List


def process_suggested_df(sugggested_df):
    movie_ids = sugggested_df['movie_id'].tolist()
    titles_with_desc = sugggested_df[['title', 'overview']].to_markdown(index=False)
    return movie_ids, titles_with_desc



