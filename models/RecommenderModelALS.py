import pickle
from implicit import als
import scipy.sparse as sp
import pandas as pd


class RecommenderModelALS:
    def __init__(self, factors=200, regularization=0.05, iterations=50):
        self.model = als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations
        )

        self.artists = None
        self.user_item_matrix = None
    
    def load_data(self, user_artists_path, artists_path):
        """
        Load user-artist interactions and artist metadata
        """
        # Load artist data
        self.artists = pd.read_csv(artists_path, sep='\t')
        self.artists = self.artists.set_index('id')

        # Load user artist interaction data
        user_artists = pd.read_csv(user_artists_path, sep='\t')
        user_artists.set_index(['userID', 'artistID'], inplace=True)
        coo = sp.coo_matrix(
            (
                user_artists.weight.astype(float),
                (
                    user_artists.index.get_level_values(0),
                    user_artists.index.get_level_values(1),
                ),
            )
        )
        self.user_item_matrix = coo.tocsr()

    def train(self):
        """
        Train ALS model
        """
        if self.user_item_matrix is None:
            raise ValueError("User artist matrix not loaded")

        self.model.fit(self.user_item_matrix)

    def recommend(self, user_id, n_recommendations=5):
        """
        Get recommendations and scores for top n artists for a given user
        """
        if self.user_item_matrix is None or self.artists is None:
            raise ValueError("Data not loaded")
        
        # Generate recommendations
        recommended_items, scores = self.model.recommend(
            userid=user_id,
            user_items=self.user_item_matrix[n_recommendations],
            N=n_recommendations
        )

        # Map recommended artist IDs to their names
        recommendations = [
            self.artists.loc[artist_id, 'name'] for artist_id in recommended_items
        ]

        return recommendations, scores

    def save_model(self, file_path):
        """
        Save the trained model to a file
        """
        with open(file_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)

    def load_model(self, file_path):
        """
        Load a pre-trained model from a file
        """
        with open(file_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
