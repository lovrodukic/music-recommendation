import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.recommender_transformer import RecommenderTransformer


DATA_PATH = './data/tracks_features.csv'

def parse_cli():
    parser = argparse.ArgumentParser(
        description="Generate recommendations for a seed item"
    )
    parser.add_argument(
        'index',
        type=str,
        help="The name of the index to use"
    )

    return parser.parse_args()

def main(index):
    recommender =  RecommenderTransformer(use_ollama=True)

    # Load data
    print("Loading data...")
    recommender.load_data(DATA_PATH)

    # Build the FAISS index
    print(f"Building FAISS index at models/{index}...")
    recommender.build_index(index)

    print("Training completed successfully.")


if __name__ == '__main__':
    args = parse_cli()
    main(args.index)
