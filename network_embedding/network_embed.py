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
import ast  # Add to imports at top
from datetime import datetime

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
    """Extract cast IDs from string representation of Python list"""
    try:
        cast_list = ast.literal_eval(cast_str)
        return set(actor['id'] for actor in cast_list)
    except (ValueError, SyntaxError, TypeError):
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

def is_valid_date(date_str):
    """Check if date is valid and after 2010"""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        return date.year >= 2010
    except:
        return False

def create_movie_cast_dict(credits_df):
    """Create dictionary mapping movie IDs to their cast"""
    movie_cast = {}
    for _, row in credits_df.iterrows():
        movie_id = row['id']
        cast = extract_cast(row['cast'])
        if cast:  # Only add movies with non-empty cast
            movie_cast[movie_id] = cast
    return movie_cast

def create_movie_info_dict(metadata_df):
    """Create dictionary with movie metadata, filtering for valid overviews and recent movies"""
    movie_info = {}
    for _, row in metadata_df.iterrows():
        try:
            movie_id = int(row['id'])
            if (is_valid_overview(row['overview']) and 
                is_valid_date(row['release_date'])):
                movie_info[movie_id] = {
                    'title': row['title'],
                    'overview': row['overview'],
                    'genres': extract_genres(row['genres']),
                    'release_date': row['release_date']
                }
        except:
            continue
    return movie_info

def create_movie_network(movie_cast, movie_info, min_shared_cast=1):
    """Create network of movies connected by shared cast members.
    
    Args:
        movie_cast (dict): Dictionary of movie_id to set of actor IDs
        movie_info (dict): Dictionary of movie_id to movie metadata
        min_shared_cast (int): Minimum number of shared actors to create an edge
    """
    G = nx.Graph()
    
    # Pre-filter movies that appear in both dictionaries
    valid_movies = set(movie_cast.keys()) & set(movie_info.keys())
    
    # Add nodes first
    for movie_id in valid_movies:
        G.add_node(movie_id, **movie_info[movie_id])
    
    # Create edges more efficiently
    # Convert to list for indexing
    movie_list = list(valid_movies)
    
    # Create dictionary of actor_id to set of movies they appear in
    actor_to_movies = {}
    for movie_id in valid_movies:
        for actor_id in movie_cast[movie_id]:
            if actor_id not in actor_to_movies:
                actor_to_movies[actor_id] = set()
            actor_to_movies[actor_id].add(movie_id)
    
    # Create edges based on shared actors
    edges_to_add = set()
    for actor_movies in actor_to_movies.values():
        if len(actor_movies) > 1:  # Actor appears in multiple movies
            # Create edges between all pairs of movies this actor appears in
            for movie1 in actor_movies:
                for movie2 in actor_movies:
                    if movie1 < movie2:  # Avoid duplicate edges
                        edges_to_add.add((movie1, movie2))
    
    # Filter edges by minimum shared cast threshold
    if min_shared_cast > 1:
        filtered_edges = []
        for movie1, movie2 in edges_to_add:
            shared_cast = len(movie_cast[movie1] & movie_cast[movie2])
            if shared_cast >= min_shared_cast:
                filtered_edges.append((movie1, movie2))
        edges_to_add = filtered_edges
    
    # Add all edges at once
    G.add_edges_from(edges_to_add)
    
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

def sample_negative_nodes(edge_index, num_nodes, num_negative_samples=5):
    """Sample negative nodes that aren't connected to source nodes"""
    device = edge_index.device
    batch_size = edge_index.size(1)
    
    # Create set of existing edges for fast lookup
    existing_edges = set((i.item(), j.item()) for i, j in edge_index.t())
    
    negative_edges = []
    for idx in range(batch_size):
        source = edge_index[0, idx].item()
        negatives = []
        while len(negatives) < num_negative_samples:
            # Sample random node
            neg = torch.randint(0, num_nodes, (1,)).item()
            # Check if this creates a new negative edge
            if neg != source and (source, neg) not in existing_edges:
                negatives.append(neg)
        negative_edges.append(negatives)
    
    return torch.tensor(negative_edges, device=device)

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
    
    def contrastive_loss(self, z, edge_index, num_nodes):
        """Compute contrastive loss with negative sampling"""
        # Get positive pairs (connected nodes)
        source_nodes = edge_index[0]
        pos_nodes = edge_index[1]
        
        # Sample negative nodes for each source node
        neg_nodes = sample_negative_nodes(edge_index, num_nodes)
        
        # Compute positive similarity
        pos_similarity = torch.sum(z[source_nodes] * z[pos_nodes], dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_similarity))
        
        # Compute negative similarity
        neg_loss = 0
        for i in range(neg_nodes.size(1)):
            neg_similarity = torch.sum(z[source_nodes] * z[neg_nodes[:, i]], dim=1)
            neg_loss += -torch.log(torch.sigmoid(-neg_similarity))
        
        return (pos_loss + neg_loss).mean()

def create_movie_embeddings(G, hidden_dim=64, out_dim=32, num_epochs=100):
    """Create movie embeddings using GraphSAGE with contrastive loss"""
    # Create features and convert graph to PyG Data
    node_features, genre_labels = create_genre_features(G)
    data, node_mapping = create_pytorch_geometric_data(G, node_features)
    
    # Initialize model
    model = MovieSAGE(in_channels=node_features.size(1),
                     hidden_channels=hidden_dim,
                     out_channels=out_dim)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    model.train()
    num_nodes = data.x.size(0)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Get node embeddings
        z = model(data.x, data.edge_index)
        
        # Compute contrastive loss
        loss = model.contrastive_loss(z, data.edge_index, num_nodes)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
    
    # Generate final embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
    
    # Convert to dictionary with original movie IDs
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

def track_data_reduction(credits_df, metadata_df):
    """Track and print data reduction at each step"""
    print("\nTracking Data Reduction:")
    print(f"1. Initial data sizes:")
    print(f"   - Credits dataset: {len(credits_df)} movies")
    print(f"   - Metadata dataset: {len(metadata_df)} movies")
    
    # Check ID conversion losses
    metadata_df['id'] = pd.to_numeric(metadata_df['id'], errors='coerce')
    invalid_ids = metadata_df['id'].isna().sum()
    print(f"\n2. Invalid IDs in metadata: {invalid_ids}")
    print(f"   Remaining after ID cleanup: {len(metadata_df.dropna(subset=['id']))}")
    
    # Check release date
    valid_date = metadata_df['release_date'].apply(is_valid_date)
    print(f"\n3. Movies before 2010: {(~valid_date).sum()}")
    print(f"   Movies from 2010 onwards: {valid_date.sum()}")
    
    # Check cast parsing losses
    valid_cast = credits_df['cast'].apply(lambda x: len(extract_cast(x)) > 0)
    print(f"\n4. Movies without valid cast data: {(~valid_cast).sum()}")
    print(f"   Movies with valid cast: {valid_cast.sum()}")
    
    # Check overview losses
    valid_overview = metadata_df['overview'].apply(is_valid_overview)
    print(f"\n5. Movies without valid overview: {(~valid_overview).sum()}")
    print(f"   Movies with valid overview: {valid_overview.sum()}")
    
    # Check genre losses
    valid_genres = metadata_df['genres'].apply(lambda x: len(extract_genres(x)) > 0)
    print(f"\n6. Movies without valid genres: {(~valid_genres).sum()}")
    print(f"   Movies with valid genres: {valid_genres.sum()}")
    
    # Intersection losses
    movie_cast = create_movie_cast_dict(credits_df)
    movie_info = create_movie_info_dict(metadata_df)
    common_movies = set(movie_cast.keys()) & set(movie_info.keys())
    print(f"\n7. Final intersection of valid movies (2010+): {len(common_movies)}")
    print(f"   Lost in intersection: {len(movie_cast) - len(common_movies)} from cast dataset")
    print(f"                         {len(movie_info) - len(common_movies)} from info dataset")

def main():
    """Main function to orchestrate the movie network creation and analysis"""
    BASE_DIR, DATA_DIR, OUTPUTS_DIR = setup_paths()
    credits_df, metadata_df = load_data(DATA_DIR)
    
    # Add tracking before network creation
    track_data_reduction(credits_df, metadata_df)
    
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
    movie_embeddings, genre_labels = create_movie_embeddings(G, num_epochs=200)
    save_embeddings(movie_embeddings, genre_labels, OUTPUTS_DIR)
    print(f"\nCreated and saved {len(movie_embeddings)} movie embeddings with {len(genre_labels)} genre features")
    
    # Detailed analysis of similar movies to Toy Story
    target_movie_id = 27205
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

