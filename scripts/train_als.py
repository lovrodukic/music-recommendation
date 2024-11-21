import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.RecommenderModelALS import RecommenderModelALS


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

    return parser.parse_args()

def main(model):
    recommender = RecommenderModelALS()

    # Load data
    print("Loading data...")
    recommender.load_data(USER_ARTISTS_PATH, ARTISTS_PATH)

    # Train the model
    print("Training model...")
    recommender.train()

    # Save the trained model
    print(f"Saving model to models/{model}...")
    recommender.save_model(f"./models/{model}")
    print("Model training and saving completed.")


if __name__ == '__main__':
    args = parse_cli()
    main(args.model)
