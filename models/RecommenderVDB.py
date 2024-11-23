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


EMBEDDING_DIM_OLLAMA = 4096
EMBEDDING_DIM_SENTENCE_TRANSFORMER = 384

FEATURES = [
    'danceability', 'energy', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]

FEATURE_INDEX_NAME = 'models/index'
TEXTUAL_INDEX_NAME = 'models/index_embedded'
ARTISTS_DATA = 'data/artists.dat'

w_n, w_e = 1.0, 1.0


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
        with open("models/feature_mappings.json", "r") as f:
            self.feature_mappings = json.load(f)

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
            faiss.IndexFlatIP(self.embedding_dim)
            if self.use_textual_embeddings
            else None
        )


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

        # Load popular artists
        # artists_df = pd.read_csv(ARTISTS_DATA, delimiter='\t')
        # if 'name' not in artists_df.columns:
        #     raise ValueError("The 'artists.dat' file must contain a 'name' column.")

        # self.popular_artists = set(artists_df['name'].str.lower().str.strip())


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
        if not self.use_textual_embeddings:
            return
        
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


    def _filter(self, seed, combined_scores):
        seed_id = seed['id']
        seed_genre = seed['genre']
        recommended_songs = pd.DataFrame([
            {
                'id': idx,
                'score': score,
                'artists': self.data.iloc[idx].get('artists', None),
                'name': self.data.iloc[idx].get('name', None),
                'genre': self.data.iloc[idx].get('genre', None),
                'popularity': self.data.iloc[idx].get('popularity', None),
            }
            for idx, score in combined_scores.items()
            if self.data.iloc[idx]['id'] != seed_id
        ])

        recommended_songs['genre_match'] = (
            recommended_songs['genre'] == seed_genre
        ).astype(int)
        recommended_songs = recommended_songs.sort_values(
            by=['genre_match', 'score'], ascending=[False, False]
        ).drop(columns=['genre_match'])

        # recommended_songs = recommended_songs.sort_values(
        #     by=['score'], ascending=[False]
        # )

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
        if self.feature_index is None or self.feature_index.ntotal == 0:
            raise ValueError("Feature index is not loader or empty")
        
        if not set(FEATURES).issubset(seed.keys()):
            raise ValueError(f"Seed is missing some features")
        
        if self.use_textual_embeddings:
            if self.textual_index is None or self.textual_index.ntotal == 0:
                raise ValueError("Textual index is not loaded or empty")

            seed['textual_representation'] = (
                self._create_textual_representation(seed)
            )
        
        numerical_features = np.array(
            [seed[feature] for feature in FEATURES]
        ).reshape(1, -1).astype('float32')

        # Search numerical FAISS index
        D_num, I_num = self.feature_index.search(numerical_features, n_search)
        D_num = D_num.flatten()
        I_num = I_num.flatten()

        # Initialize combined distances and indices
        combined_scores = defaultdict(float)

        if not self.use_textual_embeddings:
            for idx, distance in zip(I_num, D_num):
                combined_scores[idx] = w_n * distance

        if self.use_textual_embeddings:
            textual_embedding = self._generate_embedding(
                seed['textual_representation']
            ).reshape(1, -1)
            textual_embedding /= np.linalg.norm(textual_embedding)

            # Search textual FAISS index
            D_text, I_text = self.textual_index.search(
                textual_embedding, n_search
            )
            D_text = D_text.flatten()
            I_text = I_text.flatten()
            D_text = 1.0 - D_text

            for idx, distance in zip(I_text, D_text):
                combined_scores[idx] += w_e * distance

        # for score in combined_scores.items():
        #     print(f"{score}")

        recommended_songs = self._filter(seed, combined_scores)
        top_songs = recommended_songs.head(n_recommendations)
        
        # Retrieve song metadata
        recommendations = [
            {
                'artists': row['artists'],
                'name': row['name'],
                'score': row['score']
             }
            for _, row in top_songs.iterrows()
        ]

        return recommendations
