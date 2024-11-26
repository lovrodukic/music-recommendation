import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.RecommenderVDB import RecommenderVDB


DATA_PATH = './data/dataset.csv'


def main():
    recommender = RecommenderVDB(use_ollama=False)

    # Load data
    print("Loading data...")
    recommender.load_data(DATA_PATH)

    # Build the FAISS index
    recommender.build_index()

    print("Building completed successfully.")


if __name__ == '__main__':
    main()
