import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class RecommenderVector:
    def __init__(
        self,
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        embedding_dim=384
    ):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = embedding_dim
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.data = None
        self.embeddings = None


    def get_song_by_id(self, song_id):
        return self.data.iloc[song_id]


    def load_data(self, data_path):
        """
        Load the dataset containing song metadata and textual representations
        """
        self.data = pd.read_csv(data_path).drop(columns=['link'])
        
        def create_textual_representation(row):
            representation = (
                f"Artist: {row['artist']},\nSong: {row['song']},\nText: "
                f"{row['text']}"
            )
            return representation
        
        self.data['textual_representation'] = self.data.apply(
            create_textual_representation,
            axis=1
        )
    

    def build_index(self, index_name):
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

        # Generate embeddings for the dataset
        embeddings = np.zeros(
            (len(self.data['textual_representation']),
             self.embedding_dim),
             dtype='float32'
        )

        for i, representation in enumerate(self.data['textual_representation']):
            # Check progress
            if i % 500 == 0: print(f"Processed {i} instances")

            # Generate embedding locally
            embedding = self.model.encode(representation)
            embeddings[i] = np.array(embedding)

        self.faiss_index.add(embeddings)
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
