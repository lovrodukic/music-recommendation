import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.RecommenderVDB import RecommenderVDB
from models.utils import get_song_id_by_name_artist


DATA_PATH = './data/dataset.csv'


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Generate recommendations for a seed item"
    )
    parser.add_argument(
        'song_id',
        type=str,
        help="The song ID for which recommendations are generated"
    )
    parser.add_argument(
        '-n', 
        type=int, 
        default=5, 
        help="Number of recommendations to generate (default: 5)"
    )

    return parser.parse_args()


def main(seed_song_id, n_recommendations=5):
    recommender = RecommenderVDB(
        use_textual_embeddings=True, use_ollama=False
    )

    # Load data
    print("Loading data...")
    recommender.load_data(DATA_PATH)

    # Load the pre-trained FAISS index
    print(f"Loading FAISS indexes...")
    recommender.load_index()

    # Get recommendations
    seed_song = recommender.get_data_by_id(seed_song_id)
    recommendations = recommender.recommend(seed_song, n_recommendations)
    print(
        f"\nTop songs based on {seed_song['name']} "
        f"by {', '.join(seed_song['artists'])}:"
    )
    for i, rec in enumerate(recommendations, start=1):
        song_name = rec['name']
        artists = ', '.join(rec['artists'])
        print(f"{i:<4} {song_name} -- {artists}")


if __name__ == '__main__':
    args = parse_cli()
    main(args.song_id, n_recommendations=args.n)
