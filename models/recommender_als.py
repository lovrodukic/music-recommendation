import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import numpy as np
import pandas as pd
import scipy

from implicit import als


def load_user_artists(user_artists_file):
    """
    Return a CSR matrix of user_artistst.dat
    """
    user_artists = pd.read_csv(user_artists_file, sep='\t')
    user_artists.set_index(['userID', 'artistID'], inplace=True)
    coo = scipy.sparse.coo_matrix(
        (
            user_artists.weight.astype(float),
            (
                user_artists.index.get_level_values(0),
                user_artists.index.get_level_values(1),
            ),
        )
    )

    return coo.tocsr()

def load_artists(artists_file):
    """
    Load artists and return in a dataframe format
    """
    artists = pd.read_csv(artists_file, sep='\t')
    artists = artists.set_index('id')

    return artists

def get_als_recommendations(model, user_id, user_artists, n_recommendations=5):
    """
    Generate top n recommendations for a specific user using the ALS model.
    """
    recommended_items, scores = model.recommend(
        userid=user_id,
        user_items=user_artists[n_recommendations],
        N=n_recommendations
    )

    recommendations = [
        artists.loc[artist_id, 'name'] for artist_id in recommended_items
    ]

    return recommendations, scores

def mean_percentile_ranking(model, user_artists, k=10):
    """
    Calculate Mean Percentile Ranking (MPR) for the ALS model
    """
    num_users = user_artists.shape[0]
    mpr_sum = 0
    num_evaluated_users = 0

    user_items = user_artists

    for user_id in range(num_users):
        user_interacted_items = user_items[user_id].indices
        if len(user_interacted_items) == 0:
            continue

        # Get all items ranked by the model for this user
        recommended_items, scores = model.recommend(
            userid=user_id,
            user_items=user_items,
            N=user_items.shape[1],
            filter_already_liked_items=False
        )

        # Calculate the rank of each item the user interacted with
        ranks = np.argsort(np.argsort(-scores))
        user_mpr = np.mean([
            ranks[item] / len(scores) 
            for item in user_interacted_items 
            if item in recommended_items
        ])

        mpr_sum += user_mpr
        num_evaluated_users += 1

    # Calculate the average MPR across all users
    avg_mpr = mpr_sum / num_evaluated_users if num_evaluated_users > 0 else 0

    return avg_mpr

def map_at_k(model, user_artists, k=5):
    """
    Calculate Mean Average Precision
    """
    num_users = user_artists.shape[0]
    map_sum = 0
    num_evaluated_users = 0

    user_items = user_artists

    for user_id in range(num_users):
        user_interacted_items = user_items[user_id].indices
        if len(user_interacted_items) == 0:
            continue

        # Generate top-k recommendations for the user
        recommended_items, _ = model.recommend(
            userid=user_id,
            user_items=user_items[k],
            N=k
        )
        relevant_items_set = set(user_interacted_items)

        # Calculate Average Precision
        hits = 0
        sum_precision = 0
        for i, item in enumerate(recommended_items):
            if item in relevant_items_set:
                hits += 1
                sum_precision += hits / (i + 1)

        if len(relevant_items_set) > 0:
            average_precision = sum_precision / min(len(relevant_items_set), k)
        else:
            average_precision = 0

        map_sum += average_precision
        num_evaluated_users += 1

    avg_map = map_sum / num_evaluated_users if num_evaluated_users > 0 else 0

    return avg_map


if __name__ == '__main__':
    # Load data
    user_artists = load_user_artists('/content/datasets/user_artists.dat')
    artists = load_artists('/content/datasets/artists.dat')
    print(f"Sparse matrix shape: {user_artists.shape}")
    print(f"Dataframe shape: {artists.shape}")

    # Train model
    model = als.AlternatingLeastSquares(
        factors=200,
        regularization=0.05,
        iterations=50
    )
    model.fit(user_artists)
    print("Training complete.")

    user_id = 2
    recommendations, scores = get_als_recommendations(
        model, user_id, user_artists, n_recommendations=5
    )

    for (artist, score) in zip(recommendations, scores):
        print(f"{artist}: {score}")

    # Model evaluation
    avg_mpr = mean_percentile_ranking(model, user_artists)
    avg_map = map_at_k(model, user_artists, k=5)
    print(f"Mean Percentile Ranking (MPR): {avg_mpr:.4f}")
    print(f"Mean Average Precision (MAP): {avg_map:.4f}")
