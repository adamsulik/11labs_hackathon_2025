# %%
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# %% Imports and setup
def setup_paths():
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    OUTPUTS_DIR = BASE_DIR / 'outputs'
    OUTPUTS_DIR.mkdir(exist_ok=True)
    return BASE_DIR, DATA_DIR, OUTPUTS_DIR

def load_data(data_dir):
    """Load credits and metadata dataframes"""
    credits_df = pd.read_csv(data_dir / 'credits.csv')
    metadata_df = pd.read_csv(data_dir / 'movies_metadata.csv')
    metadata_df['id'] = pd.to_numeric(metadata_df['id'], errors='coerce')
    # drop rows with invalid IDs
    metadata_df = metadata_df.dropna(subset=['id'])
    metadata_df['id'] = metadata_df['id'].astype(int)
    return credits_df, metadata_df

def extract_cast(cast_str):
    """Extract cast IDs from JSON string"""
    try:
        cast = json.loads(cast_str.replace("'", '"'))
        return set(actor['id'] for actor in cast)
    except:
        return set()

def extract_genres(genres_str):
    """Extract genre names from JSON string"""
    try:
        genres = json.loads(genres_str.replace("'", '"'))
        return [g['name'] for g in genres]
    except:
        return []

def is_valid_overview(overview):
    """Check if overview is valid (string with length > 1)"""
    if not isinstance(overview, str):
        return False
    if len(overview.strip()) <= 1:
        return False
    return True

def create_movie_cast_dict(credits_df):
    """Create dictionary mapping movie IDs to their cast"""
    movie_cast = {}
    for _, row in credits_df.iterrows():
        movie_id = row['id']
        cast = extract_cast(row['cast'])
        if cast:
            movie_cast[movie_id] = cast
    return movie_cast

def create_movie_info_dict(metadata_df):
    """Create dictionary with movie metadata, filtering for valid overviews"""
    movie_info = {}
    for _, row in metadata_df.iterrows():
        try:
            movie_id = int(row['id'])
            if is_valid_overview(row['overview']):
                movie_info[movie_id] = {
                    'title': row['title'],
                    'overview': row['overview'],
                    'genres': extract_genres(row['genres'])
                }
        except:
            continue
    return movie_info

def create_movie_network(movie_cast, movie_info):
    """Create network of movies connected by shared cast members"""
    G = nx.Graph()
    
    # Add nodes
    for movie_id in movie_cast:
        if movie_id in movie_info:
            G.add_node(movie_id, **movie_info[movie_id])
    
    # Add edges (without weights)
    movies = list(set(movie_cast.keys()) & set(movie_info.keys()))
    for i in range(len(movies)):
        for j in range(i+1, len(movies)):
            movie1, movie2 = movies[i], movies[j]
            shared_cast = movie_cast[movie1].intersection(movie_cast[movie2])
            if shared_cast:
                G.add_edge(movie1, movie2)
    
    return G

def print_network_stats(G):
    """Print basic statistics about the network"""
    print(f"Number of movies (nodes): {G.number_of_nodes()}")
    print(f"Number of connections (edges): {G.number_of_edges()}")

def print_example_connections(G, num_examples=5):
    """Print example connections between movies"""
    for edge in list(G.edges(data=True))[:num_examples]:
        movie1, movie2, _ = edge
        print(f"\nConnected movies:")
        print(f"- {G.nodes[movie1]['title']} (Genres: {', '.join(G.nodes[movie1]['genres'])})")
        print(f"- {G.nodes[movie2]['title']} (Genres: {', '.join(G.nodes[movie2]['genres'])})")

def create_genre_features(G):
    """Convert genres to one-hot encoding for all nodes"""
    # Get all unique genres
    mlb = MultiLabelBinarizer()
    # Extract genres from all nodes
    genres_list = [G.nodes[node]['genres'] for node in G.nodes()]
    # Fit and transform to one-hot encoding
    genre_features = mlb.fit_transform(genres_list)
    return torch.FloatTensor(genre_features), mlb.classes_

def create_pytorch_geometric_data(G, node_features):
    """Convert NetworkX graph to PyTorch Geometric Data without edge weights"""
    # Create node mapping first (movie_id to index)
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    
    # Create edge index using the mapping
    edges = list(G.edges())
    edge_index = []
    
    for i, j in edges:
        # Add both directions for undirected graph
        edge_index.extend([[node_mapping[i], node_mapping[j]], 
                          [node_mapping[j], node_mapping[i]]])
    
    # Convert to tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    return Data(x=node_features, edge_index=edge_index), node_mapping

class MovieSAGE(torch.nn.Module):
    """GraphSAGE model for movie embedding"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

def create_movie_embeddings(G, hidden_dim=64, out_dim=32, num_epochs=100):
    """Create movie embeddings using GraphSAGE"""
    # Create features and convert graph to PyG Data
    node_features, genre_labels = create_genre_features(G)
    data, node_mapping = create_pytorch_geometric_data(G, node_features)
    
    # Initialize model
    model = MovieSAGE(in_channels=node_features.size(1),
                     hidden_channels=hidden_dim,
                     out_channels=out_dim)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train the model
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Simple reconstruction loss
        edge_index = data.edge_index
        
        # Compute similarity between connected nodes
        loss = -torch.mean(torch.sum(out[edge_index[0]] * out[edge_index[1]], dim=1))
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
    
    # Generate embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
    
    # Convert embeddings to dictionary with original movie IDs
    reverse_mapping = {idx: movie_id for movie_id, idx in node_mapping.items()}
    movie_embeddings = {reverse_mapping[idx]: emb.numpy() for idx, emb in enumerate(embeddings)}
    
    return movie_embeddings, genre_labels

def save_embeddings(embeddings, genre_labels, output_dir):
    """Save embeddings and genre labels to files"""
    with open(output_dir / 'movie_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    with open(output_dir / 'genre_labels.pkl', 'wb') as f:
        pickle.dump(genre_labels, f)

def load_embeddings(output_dir):
    """Load embeddings and genre labels from files"""
    with open(output_dir / 'movie_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    with open(output_dir / 'genre_labels.pkl', 'rb') as f:
        genre_labels = pickle.load(f)
    return embeddings, genre_labels

def find_similar_movies(movie_id, G, embeddings, n=5):
    """Find n most similar movies to given movie_id"""
    if movie_id not in embeddings:
        return []
    
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
    
    # Sort by similarity and get top n
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [(movie_id, sim, G.nodes[movie_id]['title'], G.nodes[movie_id]['genres']) 
            for movie_id, sim in similarities[:n]]

def print_sample_nodes(G, metadata_df, n=5):
    """Print sample nodes with their metadata information"""
    print("\nSample Movies in the Network:")
    print("="*80)
    
    sample_nodes = list(G.nodes())[:n]
    for node_id in sample_nodes:
        node_data = G.nodes[node_id]
        print(f"\nMovie ID: {node_id}")
        # Find corresponding row in metadata
        meta_row = metadata_df[metadata_df['id'] == node_id].iloc[0]
        print(f"\nMovie ID: {node_id}")
        print(f"Title: {node_data['title']}")
        print(f"Release Date: {meta_row['release_date']}")
        print(f"Vote Average: {meta_row['vote_average']}")
        print(f"Popularity: {meta_row['popularity']}")
        print(f"Genres: {', '.join(node_data['genres'])}")
        print("-"*40)

def main():
    """Main function to orchestrate the movie network creation and analysis"""
    BASE_DIR, DATA_DIR, OUTPUTS_DIR = setup_paths()
    credits_df, metadata_df = load_data(DATA_DIR)
    
    movie_cast = create_movie_cast_dict(credits_df)
    movie_info = create_movie_info_dict(metadata_df)
    
    print(f"Number of movies with valid overviews: {len(movie_info)}")
    
    G = create_movie_network(movie_cast, movie_info)
    print_network_stats(G)
    
    # Print sample nodes with metadata
    print('---')
    print_sample_nodes(G, metadata_df)
    print('---')
    
    # Continue with existing code...
    print_example_connections(G)
    
    # Create and save embeddings
    movie_embeddings, genre_labels = create_movie_embeddings(G)
    save_embeddings(movie_embeddings, genre_labels, OUTPUTS_DIR)
    print(f"\nCreated and saved {len(movie_embeddings)} movie embeddings with {len(genre_labels)} genre features")
    
    # Detailed analysis of similar movies to Toy Story
    target_movie_id = 862
    if target_movie_id in G.nodes:
        target_movie = G.nodes[target_movie_id]
        print("\n" + "="*80)
        print(f"Finding similar movies to:")
        print(f"Title: {target_movie['title']}")
        print(f"Genres: {', '.join(target_movie['genres'])}")
        print(f"Overview: {target_movie['overview'][:200]}...")
        print("="*80)
        
        similar_movies = find_similar_movies(target_movie_id, G, movie_embeddings)
        print("\nTop 5 Most Similar Movies:")
        print("-"*80)
        for movie_id, sim, title, genres in similar_movies:
            print(f"\nSimilarity Score: {sim:.3f}")
            print(f"Title: {title}")
            print(f"Genres: {', '.join(genres)}")
            print(f"Overview: {G.nodes[movie_id]['overview'][:200]}...")
            print("-"*40)
    
    return G, movie_embeddings, genre_labels

# %% Run the analysis
if __name__ == "__main__":
    G, embeddings, genres = main()

