# Music Recommendation System

This project implements a music recommendation system designed to provide
personalized artist and song recommendations. It leverages collaborative
filtering and mood-based recommendations through embeddings. The system consists
of a collaborative filtering model using Alternating Least Squares (ALS) and a
vector database (VDB) model for mood-based filtering via embeddings.

The ALS model utilizes the Last.fm dataset consisting of user-artist
interactions to recommend artists for a user based on their artist interactions
and other users' interactions. The VDB class creates textual representations
of songs based on their audio features to build an embedding index, and then
utilizes the index to recommend similar songs based on an input song.

These two models can be used to create a recommendation pipeline that recommends
artists based on a certain user's listening activity. The songs from these
artists can be used to search the embedding index for songs with similar "mood"
or audio features based on a dataset obtained using the Spotify API. The
overall goal is to combine collaborative filtering with natural language to
recommend music.

## Project Structure

The project is organized as follows:

- `data/`: Contains the datasets used for training and testing the
  recommendation model.
- `models/`: Contains the models:
  - `RecommenderModelALS.py`: Collaborative filtering model built using the
    Alternating Least Squares (ALS) method for artist recommendations. The model
    is saved as a `.pkl` file.
  - `RecommenderVDB.py`: Class for creating embedding indexes for improved,
    mood-based recommendations using a vector database (VDB). The index is
    saved in the directory after the build script is run.
- `notebooks/`: Contains Jupyter notebooks used for initial experimentation and
  testing before the project was finalized.
- `scripts/`: Python scripts that utilize the classes from `models/` to perform
  tasks such as training the models and making song recommendations.
- `terraform/`: Infrastructure configuration to set up an EC2 instance for using
  Ollama to improve embeddings for mood-based recommendations.

## Running the Recommendation System

The project includes scripts and classes for training and making
recommendations:

### 1. Training the Models

To train the models, you can run the following scripts from the project root:

- **Collaborative Filtering Model (ALS)**:
    ```
    python3 scripts/train_als.py
    ```

- **VDB Model for Mood-based Recommendations**:
    ```
    python3 scripts/build_vdb.py
    ```

These scripts use the data in the `data/` directory to train the models and save
the trained model and built index for later use.

### 2. Recommendation Pipeline

Once the models are trained, you can use them to create a recommendation
pipeline in `scripts/`. Use the capabilities of the classes in `models/` to
build a refined recommendation pipeline, including any filtering steps and any
fine-tuning of parameters. The directory contains some sample scripts that
leverage the ALS model and VDB index (`recommend_als.py` & `recommend_vdb.py`).
More scripts, with the naming convention `scripts/recommend.py` can be added to
the directory.

### 3. Making Recommendations

Once the recommender is created, you can use it run it like this:

- **Recommending Songs or Artists**:
    ```
    python3 scripts/recommend.py
    ```
