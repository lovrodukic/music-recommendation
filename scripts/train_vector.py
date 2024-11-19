import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.recommender_vector import RecommenderVector


DATA_PATH = './data/music.csv'
INDEX_NAME = 'index'

def main():
    recommender =  RecommenderVector()

    # Load data
    print("Loading data...")
    recommender.load_data(DATA_PATH)

    # Build the FAISS index
    print("Building FAISS index...")
    recommender.build_index(INDEX_NAME)

    print("Training completed successfully.")


if __name__ == '__main__':
    main()
