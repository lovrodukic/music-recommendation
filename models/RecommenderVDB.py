import faiss
import json
import os
import requests
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.utils import map_features


# Constants for embedding dimensions and model initialization
EMBEDDING_DIM_OLLAMA = 4096
EMBEDDING_DIM_SENTENCE_TRANSFORMER = 384
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
OLLAMA_URL = 'http://localhost:11434/api/embeddings'

# Audio feature columns to use
FEATURES = [
    'danceability', 'energy', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Path for index data file
INDEX_NAME = 'models/index_embedded'


class RecommenderVDB:
    def __init__(
        self,
        model_name=MODEL_NAME,
        ollama_url=OLLAMA_URL,
        use_ollama=False,
    ):
        """
        Initialize the RecommenderVDB instance

        Parameters:
        - model_name (str): The name of the sentence transformer model to use
        - ollama_url (str): The URL for the Ollama embedding service
        - use_ollama (bool): Whether to use Ollama for generating embeddings
        """

        # FAISS index configuration
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url
        self.embedding_dim = (
            EMBEDDING_DIM_OLLAMA if use_ollama else
            EMBEDDING_DIM_SENTENCE_TRANSFORMER
        )
        
        # Model and FAISS index initialization
        self.model = SentenceTransformer(model_name) if not use_ollama else None
        self.index_name = INDEX_NAME
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Data store and preprocessing
        self.data = None
        self.scaler = MinMaxScaler()
        with open("models/feature_mappings.json", "r") as f:
            self.feature_mappings = json.load(f)


    def get_data_by_id(self, _id):
        """
        Retrieve song metadata by song ID
        """
        result = self.data[self.data['id'] == _id]
    
        if result.empty:
            raise ValueError(f"No song found with the ID '{_id}'")
        
        return result.iloc[0]


    def load_data(self, data_path):
        """
        Load the dataset containing song metadata and textual representations
        """
        # Data preprocessing
        self.data = pd.read_csv(data_path).drop(
            columns=['key', 'loudness', 'time_signature'],
            errors='ignore'
        )
        self.data = self.data.drop_duplicates(
            subset=['name', 'artists'], keep='first'
        )
        self.data['artists'] = self.data['artists'].apply(
            lambda a: a.split(';') if isinstance(a, str) else []
        )
        self.data['tempo'] = self.scaler.fit_transform(
            self.data[['tempo']]
        )

        if not set(FEATURES).issubset(self.data.columns):
            missing = set(FEATURES) - set(self.data.columns)
            raise ValueError(f"Missing required numerical features: {missing}")


    def _generate_embedding(self, representation):
        """
        Generate textual embedding based on selected model
        """
        if self.use_ollama:
            try:
                response = requests.post(
                    self.ollama_url,
                    json={'model': 'llama2', 'prompt': representation}
                )
                response.raise_for_status()
                return np.array(response.json()['embedding'], dtype='float32')
            except requests.RequestException as e:
                raise ValueError(f"Error generating Ollama embedding: {e}")
        else:
            return self.model.encode(representation)


    def _create_textual_representation(self, row):
        """
        Create textual representation to use for embedding
        """
        feature_mappings = self.feature_mappings

        if row.name % 1000 == 0:
            print(f"Created textual representation for {row.name} rows")

        metadata = (
            f"\nTrack: {row['name']}\n"
            f"Album: {row['album']}\n"
            f"Artists: {', '.join(row['artists'])}\n"
            f"Genre: {row['genre']}\n"
        )

        feature_phrases = [
            f"{feature}: "
            f"{map_features(feature, row[feature], feature_mappings)} "
            f"({row[feature]:.2f})"
            for feature in FEATURES
        ]

        return ', '.join(feature_phrases + [metadata])


    def _build_textual_index(self, batch_size=1000):
        """
        Build the FAISS index for textual embeddings
        """
        self.data.reset_index(drop=True, inplace=True)
        num_rows = len(self.data)
        print(f"Building textual index for {num_rows} rows")
        
        self.data['textual_representation'] = self.data.apply(
            self._create_textual_representation,
            axis=1
        )

        for start_idx in range(0, len(self.data), batch_size):
            end_idx = min(start_idx + batch_size, num_rows)
            print(f"Processing rows {start_idx} to {end_idx}...")

            # Prepare batch embeddings
            batch_embeddings = np.zeros(
                (end_idx - start_idx, self.embedding_dim),
                dtype='float32'
            )

            for i, (_, row) in enumerate(
                self.data.iloc[start_idx:end_idx].iterrows()
            ):
                try:
                    # Normalize embeddings
                    textual_embedding = self._generate_embedding(
                        row['textual_representation']
                    )
                    textual_embedding /= np.linalg.norm(textual_embedding)

                    batch_embeddings[i] = textual_embedding

                except ValueError as e:
                    print(f"Skipping row {start_idx + i} due to error: {e}")

            # Add batch embeddings to FAISS index
            self.index.add(batch_embeddings)
        
        faiss.write_index(self.index, self.index_name)
        print(f"Textual index saved to {self.index_name}")


    def build_index(self, batch_size=1000):
        """
        Build both FAISS textual and numerical indexes as needed
        """
        if self.data is None:
            raise ValueError("Data not loaded")

        # Build textual index
        if not os.path.exists(self.index_name):
            self._build_textual_index(batch_size)
        else:
            print(f"Index already exists at {self.index_name}")


    def load_index(self):
        """
        Loads previously saved FAISS indexes for features and textual embeddings
        """
        # Load the textual index
        if os.path.exists(self.index_name):
            print(f"Loading textual index from {self.index_name}...")
            self.index = faiss.read_index(self.index_name)
            print(f"Textual index loaded successfully")
        else:
            raise ValueError(f"Textual index file not found")


    def _filter(self, seed, recommended_songs):
        """
        Filter and prioritize recommended songs
        """
        seed_id = seed['id']
        seed_genre = seed['genre']
        
        recommended_songs = recommended_songs[
            recommended_songs['id'] != seed_id
        ].copy()

        recommended_songs['genre_match'] = (
            recommended_songs['genre'] == seed_genre
        ).astype(int)

        recommended_songs = recommended_songs.sort_values(
            by=['genre_match', 'numerical_distance'], ascending=[False, True]
        ).drop(columns=['genre_match'])

        # Limit songs per artist to ensure diversity
        def cap_songs_per_artist(recommended_songs, max_per_artist=5):
            artist_counts = defaultdict(int)
            capped_songs = []

            for _, row in recommended_songs.iterrows():
                if all(
                    artist_counts[artist] < max_per_artist
                    for artist in row['artists']
                ):
                    capped_songs.append(row)
                    for artist in row['artists']:
                        artist_counts[artist] += 1

            return pd.DataFrame(capped_songs)

        recommended_songs = cap_songs_per_artist(recommended_songs)

        return recommended_songs


    def recommend(self, seed, n_recommendations=5, n_search=100):
        """
        Recommend songs similar to a given seed
        """
        if not set(FEATURES).issubset(seed.keys()):
            raise ValueError(f"Seed is missing some features")
        
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Textual index is not loaded or empty")

        seed['textual_representation'] = (
            self._create_textual_representation(seed)
        )

        textual_embedding = self._generate_embedding(
            seed['textual_representation']
        ).reshape(1, -1)
        textual_embedding /= np.linalg.norm(textual_embedding)

        # Search textual FAISS index
        D_text, I_text = self.index.search(textual_embedding, n_search)
        D_text = D_text.flatten()
        I_text = I_text.flatten()
        D_text = 1.0 - D_text

        # Get numerical audio features of the seed song
        numerical_features_seed = np.array(
            [seed[feature] for feature in FEATURES]
        ).reshape(1, -1).astype('float32')

        # Compute distances between the numerical features
        numerical_features_songs = self.data.iloc[I_text][FEATURES].values
        differences = numerical_features_songs - numerical_features_seed
        numerical_distances = np.linalg.norm(differences, axis=1)

        recommended_songs = self.data.iloc[I_text].copy()
        recommended_songs['index'] = I_text
        recommended_songs['textual_distance'] = D_text
        recommended_songs['numerical_distance'] = numerical_distances

        recommended_songs = self._filter(seed, recommended_songs)
        top_songs = recommended_songs.head(n_recommendations)
        
        # Retrieve song metadata
        recommendations = [
            {
                'artists': row['artists'],
                'name': row['name'],
                'score': row['numerical_distance']
             }
            for _, row in top_songs.iterrows()
        ]

        return recommendations
