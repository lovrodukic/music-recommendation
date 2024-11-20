import faiss
import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler


class RecommenderTransformer:
    EMBEDDING_DIM_OLLAMA = 4096
    EMBEDDING_DIM_SENTENCE_TRANSFORMER = 384
    NUMERICAL_DIM = 11

    def __init__(
        self,
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        ollama_url='http://localhost:11434/api/embeddings',
        use_ollama=False
    ):
        self.data = None
        self.embeddings = None
        self.scaler = MinMaxScaler()

        # FAISS index configuration
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url
        self.embedding_dim = (
            self.EMBEDDING_DIM_OLLAMA if use_ollama else
            self.EMBEDDING_DIM_SENTENCE_TRANSFORMER
        )
        self.numerical_dim = self.NUMERICAL_DIM
        self.faiss_index = faiss.IndexFlatL2(
            self.embedding_dim + self.numerical_dim
        )
        self.model = SentenceTransformer(model_name) if not use_ollama else None

    def get_song_by_id(self, song_id):
        return self.data.iloc[song_id]

    def load_data(self, data_path):
        """
        Load the dataset containing song metadata and textual representations
        """
        self.data = pd.read_csv(data_path).drop(
            columns=['id', 'album_id', 'artist_ids', 'track_number',
                     'disc_number', 'duration_ms', 'time_signature', 'year',
                     'release_date'],
            errors='ignore'
        )
        
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

        required_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        if not set(required_features).issubset(self.data.columns):
            missing = set(required_features) - set(self.data.columns)
            raise ValueError(f"Missing required numerical features: {missing}")

        columns_to_normalize = ['key', 'loudness', 'tempo']
        self.data[columns_to_normalize] = self.scaler.fit_transform(
            self.data[columns_to_normalize]
        )
    
    def generate_embedding(self, representation):
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

    def build_index(self, index_name, batch_size=1000):
        """
        Build the FAISS index using using song textual representations
        """
        if (
            self.data is None or
            'textual_representation' not in self.data.columns
        ):
            raise ValueError(
                "Data not loaded or missing 'textual_representation'"
            )
        
        num_rows = len(self.data)
        print(f"Total rows: {num_rows}")

        for start_idx in range(0, len(self.data), batch_size):
            end_idx = min(start_idx + batch_size, num_rows)
            print(f"Processing songs {start_idx} to {end_idx}...")

            # Prepare batch embeddings
            batch_embeddings = np.zeros(
                (end_idx - start_idx, self.embedding_dim + self.NUMERICAL_DIM),
                dtype='float32'
            )

            for i, row in enumerate(
                self.data.iloc[start_idx:end_idx].iterrows()
            ):
                song_idx, song_row = row

                # Generate textual embedding
                try:
                    textual_embedding = self.generate_embedding(
                        song_row['textual_representation']
                    )
                except ValueError as e:
                    print(f"Skipping song {song_idx} due to error: {e}")
                    continue

                # Extract numerical features
                numerical_features = song_row[
                    ['danceability', 'energy', 'key', 'loudness', 'mode',
                     'speechiness', 'acousticness', 'instrumentalness',
                     'liveness', 'valence', 'tempo']
                ].values.astype('float32')

                # Combine textual and numerical embeddings
                batch_embeddings[i] = np.concatenate(
                    [textual_embedding, numerical_features]
                )

            # Add batch embeddings to FAISS index
            self.faiss_index.add(batch_embeddings)

        faiss.write_index(self.faiss_index, f"models/{index_name}")
        print(f"Saved FAISS index to models/{index_name}")
    
    def load_index(self, index_file_path):
        """
        Load a previously saved FAISS index
        """
        try:
            self.faiss_index = faiss.read_index(index_file_path)
            print(f"FAISS index set successfully to {index_file_path}")
        except Exception as e:
            raise ValueError(
                f"Failed to load FAISS index from '{index_file_path}': {e}"
            )

    def recommend(self, seed, n_recommendations=5):
        """
        Recommend songs similar to a given favorite song.
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            raise ValueError("FAISS index is not loaded or empty")
        
        # Generate embedding for the favorite song
        seed_text = seed['textual_representation']
        seed_embedding = np.array(
            self.model.encode(seed_text)
        ).reshape(1, -1)

        # FAISS search for top recommendations (+ 1 to avoid duplicate)
        _, I = self.faiss_index.search(seed_embedding, n_recommendations + 1)
        indices = I.flatten()
        seed_index = seed.name
        filtered_indices = [
            idx for idx in indices if idx != seed_index
        ][:n_recommendations]

        # Retrieve song metadata
        recommendations = []
        for idx in filtered_indices:
            song_row = self.data.iloc[idx]
            recommendations.append({
                'artist': song_row['artist'],
                'song': song_row['song']
            })

        return recommendations
