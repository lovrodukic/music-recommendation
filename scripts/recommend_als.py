import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.recommender_system import RecommenderSystem


USER_ARTISTS_PATH = './data/user_artists.dat'
ARTISTS_PATH = './data/artists.dat'

def parse_cli():
    parser = argparse.ArgumentParser(
        description="Generate recommendations for a user"
    )
    parser.add_argument(
        'model',
        type=str,
        help="The name of the model to use"
    )
    parser.add_argument(
        'user_id',
        type=int,
        help="The user ID for which recommendations are generated"
    )
    parser.add_argument(
        '-n', 
        type=int, 
        default=5, 
        help="Number of recommendations to generate (default: 5)"
    )

    return parser.parse_args()

def main(model, user_id, n_recommendations=5):
    recommender = RecommenderSystem()

    # Load data
    print("Loading data...")
    recommender.load_data(USER_ARTISTS_PATH, ARTISTS_PATH)
    
    # Load the trained model
    print(f"Loading model from models/{model}...")
    recommender.load_model(f"./models/{model}")

    # Get recommendations
    print(f"Generating recommendations for user {user_id}...")
    recommendations, scores = recommender.recommend(user_id, n_recommendations)

    # Display the recommendations
    print("\nTop Recommendations:")
    for i, (artist, score) in enumerate(zip(recommendations, scores), start=1):
        print(f"{i}. {artist} (Score: {score:.4f})")


if __name__ == '__main__':
    args = parse_cli()
    main(args.model, args.user_id, n_recommendations=args.n)
