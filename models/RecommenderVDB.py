import ast
import faiss
import os
import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler


EMBEDDING_DIM_OLLAMA = 4096
EMBEDDING_DIM_SENTENCE_TRANSFORMER = 384

FEATURES = [
    'danceability', 'energy', 'mode',  'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]

FEATURE_INDEX_NAME = 'models/index'
TEXTUAL_INDEX_NAME = 'models/index_embedded'

class RecommenderVDB:
    def __init__(
        self,
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        ollama_url='http://localhost:11434/api/embeddings',
        use_ollama=False,
        use_textual_embeddings=False
    ):
        # Data store and preprocessing
        self.data = None
        self.scaler = MinMaxScaler()

        # FAISS index configuration
        self.use_ollama = use_ollama
        self.use_textual_embeddings = use_textual_embeddings
        self.ollama_url = ollama_url
        self.feature_dim = len(FEATURES)
        self.embedding_dim = (
            EMBEDDING_DIM_OLLAMA if use_ollama else
            EMBEDDING_DIM_SENTENCE_TRANSFORMER
        ) if use_textual_embeddings else 0
        
        # Model and indexes
        self.model = (
            SentenceTransformer(model_name) if use_textual_embeddings 
            and not use_ollama else None
        )

        self.feature_index_name = FEATURE_INDEX_NAME
        self.feature_index = faiss.IndexFlatL2(self.feature_dim)

        self.textual_index_name = TEXTUAL_INDEX_NAME
        self.textual_index = (
            faiss.IndexFlatL2(self.embedding_dim)
            if self.use_textual_embeddings
            else None
        )

    def get_song_data_by_id(self, _id):
        """
        Retrieve song metadata by song ID
        """
        result = self.data[self.data['id'] == _id]
    
        if result.empty:
            raise ValueError(f"No song found with the ID '{_id}'")
        
        return result.iloc[0]
    
    def get_song_id_by_name_artist(self, name, artist):
        """
        Retrieve song ID name and artist
        """
        result = self.data[
            (self.data['name'].str.lower() == name.lower()) &
            (self.data['artists'].apply(
                lambda artists: artist.lower() in [a.lower() for a in artists]
                )
            )
        ]
        
        if result.empty:
            raise ValueError(
                f"No song found with the name '{name}' and artist '{artist}'"
            )
        
        return result['id'].iloc[0]

    def get_songs_by_artist(self, artist):
        """
        Retrieve all songs and their IDs for a given artist.
        """
        result = self.data[
            self.data['artists'].apply(
                lambda artists: artist.lower() in [a.lower() for a in artists]
            )
        ]
        
        if result.empty:
            raise ValueError(f"No songs found for the artist '{artist}'")
        
        return result[['id', 'name']]
    
    def load_data(self, data_path):
        """
        Load the dataset containing song metadata and textual representations
        """
        self.data = pd.read_csv(data_path).drop(
            columns=['album_id', 'artist_ids', 'track_number',
                     'disc_number', 'duration_ms', 'time_signature', 'year',
                     'release_date', 'key', 'loudness'],
            errors='ignore'
        )
        self.data = self.data.drop_duplicates(
            subset=['name', 'artists'], keep='first'
        )
        self.data['artists'] = self.data['artists'].apply(ast.literal_eval)
        
        if self.use_textual_embeddings:
            def create_textual_representation(row):
                return (
                    f"Track: {row['name']},\n"
                    f"Album: {row['album']},\n"
                    f"Artists: {row['artists']},\n"
                    f"Explicit: {row['explicit']}"
                )
            
            self.data['textual_representation'] = self.data.apply(
                create_textual_representation,
                axis=1
            )

        if not set(FEATURES).issubset(self.data.columns):
            missing = set(FEATURES) - set(self.data.columns)
            raise ValueError(f"Missing required numerical features: {missing}")

        self.data['tempo'] = self.scaler.fit_transform(
            self.data[['tempo']]
        )
    
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

    def _build_textual_index(self, batch_size=1000):
        """
        Build the FAISS index for textual embeddings
        """
        if not self.use_textual_embeddings:
            return

        num_rows = len(self.data)
        print(f"Building textual index for {num_rows} rows")

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
                    batch_embeddings[i] = self._generate_embedding(
                        row['textual_representation']
                    )
                except ValueError as e:
                    print(f"Skipping row {start_idx + 1} due to error: {e}")

            # Add batch embeddings to FAISS index
            self.textual_index.add(batch_embeddings)
        
        faiss.write_index(self.textual_index, self.textual_index_name)
        print(f"Textual index saved to {self.textual_index_name}")

    def _build_feature_index(self):
        """
        Build the FAISS index for numerical audio features
        """
        if os.path.exists(self.feature_index_name):
            print(f"Feature index already exists at {self.feature_index_name}")
            self.feature_index = faiss.read_index(self.feature_index_name)
            return
        
        print("Building feature index...")
        features = self.data[FEATURES].values.astype('float32')
        self.feature_index.add(features)
        faiss.write_index(self.feature_index, self.feature_index_name)

    def build_index(self, batch_size=1000):
        """
        Build both FAISS textual and numerical indexes as needed
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Build numerical index
        self._build_feature_index()

        # Build textual index if enabled
        if self.use_textual_embeddings:
            self._build_textual_index(batch_size)
    
    def load_index(self):
        """
        Loads previously saved FAISS indexes for features and textual embeddings
        """
        # Load the feature index
        if os.path.exists(self.feature_index_name):
            print(f"Loading numerical index from {self.feature_index_name}...")
            self.feature_index = faiss.read_index(self.feature_index_name)
            print(f"Numerical index loaded successfully")
        else:
             raise ValueError(f"Numerical index file not found")
    
        # Load the textual index if enabled
        if self.use_textual_embeddings:
            if os.path.exists(self.textual_index_name):
                print(
                    f"Loading textual index from {self.textual_index_name}..."
                )
                self.textual_index = faiss.read_index(self.textual_index_name)
                print(f"Textual index loaded successfully")
            else:
                raise ValueError(f"Textual index file not found")

    def recommend(self, seed, n_recommendations=5):
        """
        Recommend songs similar to a given favorite song.
        """
        if self.use_textual_embeddings:
             if self.textual_index is None or self.textual_index.ntotal == 0:
                raise ValueError("Textual index is not loaded or empty")
        else:
            if self.feature_index is None or self.feature_index.ntotal == 0:
                raise ValueError("Feature index is not loader or empty")
        
        if self.use_textual_embeddings:
            if 'textual_representation' not in seed:
                raise ValueError("Textual representation is missing")
            
            # Generate embedding for the seed song
            seed_embedding = self._generate_embedding(
                seed['textual_representation']
            ).reshape(1, -1)

            index = self.textual_index
        else:
            if not set(FEATURES).issubset(seed.keys()):
                raise ValueError(f"Seed is missing some features")

            # Use preprocessed features from the seed song
            seed_features = np.array(
                [seed[feature] for feature in FEATURES]
            ).reshape(1, -1).astype('float32')

            seed_embedding = seed_features
            index = self.feature_index

        # FAISS search for top recommendations (+ 1 to avoid duplicate)
        _, I = index.search(seed_embedding, n_recommendations + 1)
        indices = I.flatten()
        
        recommended_ids = self.data.iloc[indices]['id'].values
        seed_id = seed['id']
        filtered_ids = [
            song_id for song_id in recommended_ids if song_id != seed_id
        ]
        filtered_ids = filtered_ids[:n_recommendations]

        # Retrieve song metadata
        recommendations = []
        for song_id in filtered_ids:
            row = self.data[self.data['id'] == song_id].iloc[0]
            recommendations.append({
                'artists': row['artists'],
                'name': row['name']
            })

        return recommendations
