import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.recommender_system import RecommenderSystem


USER_ARTISTS_PATH = './data/user_artists.dat'
ARTISTS_PATH = './data/artists.dat'
MODEL_PATH = './models/als_model.pkl'

def main():
    recommender = RecommenderSystem()

    # Load data
    print("Loading data...")
    recommender.load_data(USER_ARTISTS_PATH, ARTISTS_PATH)

    # Train the model
    print("Training model...")
    recommender.train()

    # Save the trained model
    print(f"Saving model to {MODEL_PATH}...")
    recommender.save_model(MODEL_PATH)
    print("Model training and saving completed.")

if __name__ == '__main__':
    main()
