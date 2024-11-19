import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.recommender_vector import RecommenderVector


DATA_PATH = './data/music.csv'
INDEX_PATH = './models/index_transformer'

def parse_cli():
    parser = argparse.ArgumentParser(
        description="Generate recommendations for a seed item"
    )
    parser.add_argument(
        'song_id',
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

def main(seed_song_id, n_recommendations=5):
    recommender = RecommenderVector()

    # Load data
    print("Loading data...")
    recommender.load_data(DATA_PATH)

    # Load the pre-trained FAISS index
    print(f"Loading FAISS index from {INDEX_PATH}...")
    recommender.load_index(INDEX_PATH)

    # Get recommendations
    seed_song = recommender.get_song_by_id(seed_song_id)
    recommendations = recommender.recommend(seed_song, n_recommendations)
    print(
        f"\nTop {n_recommendations} Recommendations based on {seed_song.song} "
        f"by {seed_song.artist}:"
    )
    for i, rec in enumerate(recommendations, start=1):
        print(f"{i}. {rec['song']} - {rec['artist']}")


if __name__ == '__main__':
    args = parse_cli()
    main(args.song_id, n_recommendations=args.n)
