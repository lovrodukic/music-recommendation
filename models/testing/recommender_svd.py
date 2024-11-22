import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from collections import defaultdict
from surprise import SVD, Dataset, Reader
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split


def load_user_artists(user_artists_file):
    """
    Load the user-artists interactions data
    """
    return pd.read_csv(user_artists_file, sep='\t')

def load_artists(artists_file):
    """
    Load the artists data
    """
    return pd.read_csv(artists_file, sep='\t')

def normalize_weights(user_artists):
    user_artists['log_weight'] = np.log1p(user_artists['weight'])
    min_wt = user_artists['log_weight'].min()
    max_wt = user_artists['log_weight'].max()

    # Normalize the log-transformed weights to the range [1, 5]
    user_artists['normalized_weight'] = 1 + 4 * (
        (user_artists['log_weight'] - min_wt) / (max_wt - min_wt)
    )

    return user_artists

def transform_data(user_artists):
    """
    Transforms the data into SVD format
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        user_artists[['userID', 'artistID', 'normalized_weight']],
        reader
    )
    print("Data loaded into Surprise format")

    trainset, testset = train_test_split(data, test_size=0.2)

    return trainset, testset

def plot_error():
    """
    Plot error distribution for dataset
    """
    # Extract actual and predicted values from the predictions
    actual = [pred.r_ui for pred in predictions]
    predicted = [pred.est for pred in predictions]
    errors = np.array(actual) - np.array(predicted)

    # Plot error distribution
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=errors, color='skyblue', density_norm='width', inner=None)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.xlim(-0.75, 0.75)

    plt.title('Error Distribution', fontsize=16)
    plt.xlabel('Error', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()

def plot_ape():
    """
    Plot absolute percent error
    """
    actual = np.array([pred.r_ui for pred in predictions])
    predicted = np.array([pred.est for pred in predictions])

    # Calculate Absolute Percentage Error (APE)
    ape = np.abs((actual - predicted) / actual) * 100

    plt.figure(figsize=(10, 6))
    sns.violinplot(x=ape, color='coral', inner=None)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.xlim(0, 100)
    plt.title('Absolute Percentage Error Distribution', fontsize=16)
    plt.xlabel('Error (%)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()

def precision_recall_at_k(predictions, k, threshold):
    """
    Calculate precision and recall (and F1 score)
    """
    user_est_true = defaultdict(list)
    for pred in predictions:
        # (predicted, actual)
        user_est_true[pred.uid].append((pred.est, pred.r_ui))

    precisions = []
    recalls = []
    f1_scores = []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        # Calculate the number of relevant items based on a rating threshold
        n_relevant = sum(1 for (pred, actual) in top_k if actual >= threshold)
        n_relevant_total = sum(
            1 for (pred, actual) in user_ratings if actual >= threshold
        )

        precision = n_relevant / k if k > 0 else 0
        recall = n_relevant / n_relevant_total if n_relevant_total > 0 else 0
        precisions.append(precision)
        recalls.append(recall)

        f1 = (2 * (precision * recall) /
              (precision + recall) if precision + recall > 0 else 0)
        f1_scores.append(f1)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    return avg_precision, avg_recall, avg_f1

def get_recommendations(user_id, n_recommendations=5, threshold=0.75):
    """
    Generate top n artist recommendations for a given user
    """
    all_artists = user_artists['artistID'].unique()
    predictions = [
        model.predict(user_id, artist_id) for artist_id in all_artists
    ]
    filtered_predictions = [
        pred for pred in predictions if pred.est >= threshold
    ]
    filtered_predictions.sort(key=lambda k: k.est, reverse=True)

    top_artists = [
        pred.iid for pred in filtered_predictions[:n_recommendations]
    ]
    recommended = artists[artists['id'].isin(top_artists)]['name'].tolist()

    return recommended


if __name__ == '__main__':
    # Load data
    user_artists = load_user_artists('/content/datasets/user_artists.dat')
    artists = load_artists('/content/datasets/artists.dat')

    print("User-Artists Interactions:")
    print(user_artists.head())
    print("Artists Data:")
    print(artists.head())

    # Quick info about the datasets
    user_artists.info()
    artists.info()

    # Check for missing values
    print("\nMissing values in user_artists:\n", user_artists.isnull().sum())
    print("\nMissing values in artists:\n", artists.isnull().sum())

    # Remove duplicates in user_artists
    initial_count = len(user_artists)
    user_artists.drop_duplicates(inplace=True)
    print(f"\nRemoved {initial_count - len(user_artists)} duplicates.")

    user_artists = normalize_weights(user_artists)

    # Check the range after normalization
    print("Log-transformed and normalized weight range:",
          user_artists['normalized_weight'].min(),
          "-",
          user_artists['normalized_weight'].max())

    # Obtain training set and test set
    trainset, testset = transform_data(user_artists)
    print(f"Training set size: {trainset.n_ratings}")
    print(f"Test set size: {len(testset)}")

    # Train model
    model = SVD(
        n_factors=150,
        n_epochs=40,
        reg_all=0.015,
        lr_all=0.005
    )
    model.fit(trainset)
    print("Training complete.")

    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    # Plots
    plot_error()
    plot_ape()

    # Print precision, recall, F1 score for varying k and threshold
    for k in [5, 10, 20]:
        for thresh in [0.5, 1.0, 2.0]:
            precision, recall, f1 = precision_recall_at_k(
                predictions, k, thresh
            )

            print(
                f"k={k}, threshold={thresh} -> Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, F1 Score: {f1:.4f}"
            )

    user_id = 2
    recommendations = get_recommendations(user_id)
    print(f"Top recommendations for user {user_id}: {recommendations}")
