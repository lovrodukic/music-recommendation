import ast
import pandas as pd
from typing import List, Tuple


data = pd.read_csv('data/dataset.csv')
data['artists'] = data['artists'].apply(
    lambda x: x.split('; ') if isinstance(x, str) else []
)


def get_song_id_by_name_artist(name: str, artist: str) -> str:
    """
    Retrieve song ID name and artist from tracks_features.csv
    """
    result = data[
        (data['name'].str.lower() == name.lower()) &
        (data['artists'].apply(
            lambda artists: artist.lower() in [a.lower() for a in artists]
            )
        )
    ]
    
    if result.empty:
        raise ValueError(
            f"No song found with the name '{name}' and artist '{artist}'"
        )
    
    return result['id'].iloc[0]


def get_songs_by_artist(artist: str) -> List[Tuple[str, str]]:
    """
    Retrieve all songs and their IDs for a given artist from dataset
    """
    result = data[
        data['artists'].apply(
            lambda artists: artist.lower() in [a.lower() for a in artists]
        )
    ]
    
    if result.empty:
        raise ValueError(f"No songs found for the artist '{artist}'")

    result = result.drop_duplicates(subset='name', keep='first')
    
    return list(result[['id', 'name']].itertuples(index=False, name=None))
